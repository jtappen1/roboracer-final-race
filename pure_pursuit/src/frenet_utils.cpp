#include "final_race_pure_pursuit/frenet_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "final_race_pure_pursuit/mpc_utils.hpp"

namespace mpc {

double wrapAngle(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

double angleDiff(double target, double source) {
  return wrapAngle(target - source);
}

ReferenceFrenetPath::ReferenceFrenetPath(
    const Eigen::VectorXd &cx, const Eigen::VectorXd &cy,
    const std::optional<Eigen::VectorXd> &cyaw,
    const std::optional<Eigen::VectorXd> &cv)
    : cx_(cx), cy_(cy), cv_(cv) {
  if (cx_.size() != cy_.size()) {
    throw std::runtime_error(
        "Centerline x/y must be 1D arrays with matching length.");
  }
  if (cx_.size() < 3) {
    throw std::runtime_error("Need at least three waypoints to build a Frenet path.");
  }
  if (cv_ && cv_->size() != cx_.size()) {
    throw std::runtime_error("Reference speed size must match centerline.");
  }
  if (cyaw && cyaw->size() != cx_.size()) {
    throw std::runtime_error("Reference yaw size must match centerline.");
  }

  const Eigen::Index n = cx_.size();
  points_.resize(n, 2);
  points_.col(0) = cx_;
  points_.col(1) = cy_;

  next_idx_.resize(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    next_idx_(i) = static_cast<int>((i + 1) % n);
  }

  seg_vec_.resize(n, 2);
  for (Eigen::Index i = 0; i < n; ++i) {
    seg_vec_.row(i) = points_.row(next_idx_(i)) - points_.row(i);
  }

  seg_len_ = seg_vec_.rowwise().norm();
  seg_len_ = seg_len_.array().max(1e-6);
  seg_len_sq_ = seg_len_.array().square();

  cum_s_ = Eigen::VectorXd::Zero(n + 1);
  for (Eigen::Index i = 0; i < n; ++i) {
    cum_s_(i + 1) = cum_s_(i) + seg_len_(i);
  }
  total_length_ = cum_s_(n);

  if (cyaw) {
    cyaw_ = unwrapAngles(*cyaw);
  } else {
    cyaw_.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      cyaw_(i) = std::atan2(seg_vec_(i, 1), seg_vec_(i, 0));
    }
  }

  kappa_.resize(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const int next = next_idx_(i);
    kappa_(i) = angleDiff(cyaw_(next), cyaw_(i)) / seg_len_(i);
  }
}

double ReferenceFrenetPath::wrapS(double s) const {
  if (total_length_ <= 0.0) {
    return 0.0;
  }
  const double wrapped = std::fmod(s, total_length_);
  return wrapped < 0.0 ? wrapped + total_length_ : wrapped;
}

double ReferenceFrenetPath::forwardDistance(double s_from, double s_to) const {
  return wrapS(s_to - s_from);
}

double ReferenceFrenetPath::signedSDelta(double s_from, double s_to) const {
  double delta = wrapS(s_to) - wrapS(s_from);
  if (delta > 0.5 * total_length_) {
    delta -= total_length_;
  } else if (delta < -0.5 * total_length_) {
    delta += total_length_;
  }
  return delta;
}

double ReferenceFrenetPath::unwrapSValue(double reference_s,
                                         double wrapped_s) const {
  return reference_s + signedSDelta(reference_s, wrapped_s);
}

Eigen::VectorXd ReferenceFrenetPath::unwrapSSequence(
    const Eigen::VectorXd &s_values,
    const std::optional<double> &reference_s) const {
  if (s_values.size() == 0) {
    return s_values;
  }

  Eigen::VectorXd unwrapped = Eigen::VectorXd::Zero(s_values.size());
  if (reference_s) {
    unwrapped(0) = unwrapSValue(*reference_s, s_values(0));
  } else {
    unwrapped(0) = s_values(0);
  }

  for (Eigen::Index i = 1; i < s_values.size(); ++i) {
    const double prev_wrapped = wrapS(unwrapped(i - 1));
    unwrapped(i) =
        unwrapped(i - 1) + signedSDelta(prev_wrapped, s_values(i));
  }
  return unwrapped;
}

FrenetProjection ReferenceFrenetPath::segmentSample(int seg_idx,
                                                    double seg_t) const {
  const int n = static_cast<int>(cx_.size());
  seg_idx = ((seg_idx % n) + n) % n;
  seg_t = clamp(seg_t, 0.0, 1.0);
  const int nxt = next_idx_(seg_idx);

  const double base_x = (1.0 - seg_t) * cx_(seg_idx) + seg_t * cx_(nxt);
  const double base_y = (1.0 - seg_t) * cy_(seg_idx) + seg_t * cy_(nxt);
  const double yaw =
      cyaw_(seg_idx) + seg_t * angleDiff(cyaw_(nxt), cyaw_(seg_idx));
  const double kappa =
      (1.0 - seg_t) * kappa_(seg_idx) + seg_t * kappa_(nxt);
  const double s = cum_s_(seg_idx) + seg_t * seg_len_(seg_idx);

  FrenetProjection projection;
  projection.s = s;
  projection.d = 0.0;
  projection.seg_idx = seg_idx;
  projection.seg_t = seg_t;
  projection.x = base_x;
  projection.y = base_y;
  projection.yaw = yaw;
  projection.kappa = kappa;
  return projection;
}

FrenetProjection ReferenceFrenetPath::sampleCenterline(double s) const {
  const double s_wrapped = wrapS(s);
  auto begin = cum_s_.data();
  auto end = begin + cum_s_.size();
  int seg_idx = static_cast<int>(std::upper_bound(begin, end, s_wrapped) - begin - 1);
  seg_idx = std::min(std::max(seg_idx, 0), static_cast<int>(cx_.size() - 1));
  const double seg_s = s_wrapped - cum_s_(seg_idx);
  const double seg_t = seg_s / seg_len_(seg_idx);
  return segmentSample(seg_idx, seg_t);
}

double ReferenceFrenetPath::sampleSpeed(double s) const {
  if (!cv_) {
    return 0.0;
  }
  const FrenetProjection sample = sampleCenterline(s);
  const int nxt = next_idx_(sample.seg_idx);
  return (1.0 - sample.seg_t) * (*cv_)(sample.seg_idx) +
         sample.seg_t * (*cv_)(nxt);
}

FrenetProjection ReferenceFrenetPath::projectXY(double x, double y) const {
  const Eigen::Vector2d point(x, y);
  Eigen::MatrixXd diff = (-points_).rowwise() + point.transpose();

  Eigen::VectorXd dots(points_.rows());
  for (Eigen::Index i = 0; i < points_.rows(); ++i) {
    dots(i) = diff.row(i).dot(seg_vec_.row(i)) / seg_len_sq_(i);
  }

  Eigen::VectorXd t = dots.array().max(0.0).min(1.0);
  Eigen::MatrixXd projections(points_.rows(), 2);
  for (Eigen::Index i = 0; i < points_.rows(); ++i) {
    projections.row(i) = points_.row(i) + seg_vec_.row(i) * t(i);
  }

  Eigen::MatrixXd residual =
      (-projections).rowwise() + point.transpose();
  Eigen::VectorXd dist_sq = residual.rowwise().squaredNorm();
  Eigen::Index seg_idx = 0;
  dist_sq.minCoeff(&seg_idx);
  const double seg_t = t(seg_idx);

  FrenetProjection sample = segmentSample(static_cast<int>(seg_idx), seg_t);
  const Eigen::Vector2d normal(-std::sin(sample.yaw), std::cos(sample.yaw));
  sample.d = (point - Eigen::Vector2d(sample.x, sample.y)).dot(normal);
  return sample;
}

std::tuple<double, double, double, double> ReferenceFrenetPath::frenetToCartesian(
    double s, double d) const {
  const FrenetProjection sample = sampleCenterline(s);
  const double x = sample.x - std::sin(sample.yaw) * d;
  const double y = sample.y + std::cos(sample.yaw) * d;
  return {x, y, sample.yaw, sample.kappa};
}

double ReferenceFrenetPath::totalLength() const { return total_length_; }

}  // namespace mpc

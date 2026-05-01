#ifndef FINAL_RACE_PURE_PURSUIT_FRENET_UTILS_HPP
#define FINAL_RACE_PURE_PURSUIT_FRENET_UTILS_HPP

#include <optional>
#include <tuple>

#include <Eigen/Core>

namespace mpc {

double wrapAngle(double angle);
double angleDiff(double target, double source);

struct FrenetProjection {
  double s = 0.0;
  double d = 0.0;
  int seg_idx = 0;
  double seg_t = 0.0;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double kappa = 0.0;
};

class ReferenceFrenetPath {
 public:
  ReferenceFrenetPath(const Eigen::VectorXd &cx, const Eigen::VectorXd &cy,
                      const std::optional<Eigen::VectorXd> &cyaw = std::nullopt,
                      const std::optional<Eigen::VectorXd> &cv = std::nullopt);

  double wrapS(double s) const;
  double forwardDistance(double s_from, double s_to) const;
  double signedSDelta(double s_from, double s_to) const;
  double unwrapSValue(double reference_s, double wrapped_s) const;
  Eigen::VectorXd unwrapSSequence(const Eigen::VectorXd &s_values,
                                  const std::optional<double> &reference_s =
                                      std::nullopt) const;

  FrenetProjection sampleCenterline(double s) const;
  double sampleSpeed(double s) const;
  FrenetProjection projectXY(double x, double y) const;
  std::tuple<double, double, double, double> frenetToCartesian(double s,
                                                               double d) const;

  double totalLength() const;

 private:
  FrenetProjection segmentSample(int seg_idx, double seg_t) const;

  Eigen::VectorXd cx_;
  Eigen::VectorXd cy_;
  std::optional<Eigen::VectorXd> cv_;
  Eigen::MatrixXd points_;
  Eigen::MatrixXd seg_vec_;
  Eigen::VectorXi next_idx_;
  Eigen::VectorXd seg_len_;
  Eigen::VectorXd seg_len_sq_;
  Eigen::VectorXd cum_s_;
  Eigen::VectorXd cyaw_;
  Eigen::VectorXd kappa_;
  double total_length_ = 0.0;
};

}  // namespace mpc

#endif  // FINAL_RACE_PURE_PURSUIT_FRENET_UTILS_HPP

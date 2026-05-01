#include "final_race_pure_pursuit/frenet_overtake_planner.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <Eigen/LU>

namespace mpc {

namespace {

double clipValue(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

Eigen::VectorXd clipVector(const Eigen::VectorXd &values, double lo, double hi) {
  Eigen::VectorXd clipped(values.size());
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    clipped(i) = clipValue(values(i), lo, hi);
  }
  return clipped;
}

Eigen::VectorXd linspace(double start, double step, int count) {
  Eigen::VectorXd out(count);
  for (int i = 0; i < count; ++i) {
    out(i) = start + step * static_cast<double>(i);
  }
  return out;
}

double vectorMax(const Eigen::VectorXd &values) {
  return values.size() == 0 ? 0.0 : values.maxCoeff();
}

}  // namespace

Eigen::VectorXd gradient(const Eigen::VectorXd &values, double dt) {
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(values.size());
  if (values.size() == 0) {
    return grad;
  }
  if (values.size() == 1) {
    return grad;
  }

  grad(0) = (values(1) - values(0)) / dt;
  for (Eigen::Index i = 1; i < values.size() - 1; ++i) {
    grad(i) = (values(i + 1) - values(i - 1)) / (2.0 * dt);
  }
  grad(values.size() - 1) =
      (values(values.size() - 1) - values(values.size() - 2)) / dt;
  return grad;
}

QuinticPolynomial::QuinticPolynomial(double p0, double v0, double a0, double p1,
                                     double v1, double a1, double duration) {
  duration_ = std::max(duration, 1e-3);
  a0_ = p0;
  a1_ = v0;
  a2_ = 0.5 * a0;

  const double t = duration_;
  Eigen::Matrix3d a;
  a << std::pow(t, 3), std::pow(t, 4), std::pow(t, 5), 3.0 * std::pow(t, 2),
      4.0 * std::pow(t, 3), 5.0 * std::pow(t, 4), 6.0 * t, 12.0 * std::pow(t, 2),
      20.0 * std::pow(t, 3);

  Eigen::Vector3d b;
  b << p1 - (a0_ + a1_ * t + a2_ * std::pow(t, 2)),
      v1 - (a1_ + 2.0 * a2_ * t), a1 - (2.0 * a2_);

  const Eigen::Vector3d coeffs = a.fullPivLu().solve(b);
  a3_ = coeffs(0);
  a4_ = coeffs(1);
  a5_ = coeffs(2);
}

double QuinticPolynomial::clampT(double t) const {
  return clipValue(t, 0.0, duration_);
}

double QuinticPolynomial::calcPoint(double t) const {
  t = clampT(t);
  return a0_ + a1_ * t + a2_ * std::pow(t, 2) + a3_ * std::pow(t, 3) +
         a4_ * std::pow(t, 4) + a5_ * std::pow(t, 5);
}

double QuinticPolynomial::calcFirstDerivative(double t) const {
  t = clampT(t);
  return a1_ + 2.0 * a2_ * t + 3.0 * a3_ * std::pow(t, 2) +
         4.0 * a4_ * std::pow(t, 3) + 5.0 * a5_ * std::pow(t, 4);
}

double QuinticPolynomial::calcSecondDerivative(double t) const {
  t = clampT(t);
  return 2.0 * a2_ + 6.0 * a3_ * t + 12.0 * a4_ * std::pow(t, 2) +
         20.0 * a5_ * std::pow(t, 3);
}

FrenetOvertakePlanner::FrenetOvertakePlanner(
    const Eigen::VectorXd &ref_x, const Eigen::VectorXd &ref_y,
    const Eigen::VectorXd &ref_yaw, const Eigen::VectorXd &ref_v, double dt,
    int horizon_steps, double max_speed, double min_speed,
    double track_half_width)
    : reference_(ref_x, ref_y, ref_yaw, ref_v) {
  config_.horizon_steps = horizon_steps;
  config_.dt = dt;
  config_.max_speed = max_speed;
  config_.min_speed = min_speed;
  config_.track_half_width = track_half_width;
}

void FrenetOvertakePlanner::reset() {
  obstacles_.clear();
  prev_by_id_.clear();
  last_selected_d_.reset();
  last_selected_side_ = 0;
  last_mode_ = "nominal";
}

void FrenetOvertakePlanner::updateObstacles(
    const overtake_msgs::msg::TrackedObstacleArray &obstacle_array_msg,
    double stamp_sec) {
  for (const auto &dead_id : obstacle_array_msg.dead_ids) {
    prev_by_id_.erase(static_cast<int>(dead_id));
  }

  if (obstacle_array_msg.obstacles.empty()) {
    obstacles_.clear();
    prev_by_id_.clear();
    return;
  }

  std::unordered_map<int, ObstacleTrack> new_by_id;
  for (const auto &obs : obstacle_array_msg.obstacles) {
    const int obs_id = static_cast<int>(obs.id);
    const double x = obs.pose.position.x;
    const double y = obs.pose.position.y;
    const FrenetProjection projection = reference_.projectXY(x, y);
    const double yaw = quatToYaw(obs.pose.orientation);
    const double radius =
        obs.radius > 0.0 ? static_cast<double>(obs.radius) : config_.obstacle_radius;

    const double vx_world = obs.velocity.x;
    const double vy_world = obs.velocity.y;
    const double measured_vs =
        vx_world * std::cos(projection.yaw) + vy_world * std::sin(projection.yaw);
    const double measured_vd =
        -vx_world * std::sin(projection.yaw) + vy_world * std::cos(projection.yaw);
    const bool has_measured_velocity =
        std::hypot(vx_world, vy_world) > 1e-3;

    double vs = has_measured_velocity ? measured_vs : 0.0;
    double vd = has_measured_velocity ? measured_vd : 0.0;
    const auto prev_it = prev_by_id_.find(obs_id);
    if (prev_it != prev_by_id_.end()) {
      const ObstacleTrack &prev = prev_it->second;
      const double dt = std::max(stamp_sec - prev.stamp, 1e-3);
      const double ds = reference_.signedSDelta(prev.s, projection.s);
      const double raw_vs = ds / dt;
      const double raw_vd = (projection.d - prev.d) / dt;
      if (has_measured_velocity) {
        vs = 0.35 * prev.vs + 0.25 * raw_vs + 0.40 * measured_vs;
        vd = 0.35 * prev.vd + 0.25 * raw_vd + 0.40 * measured_vd;
      } else {
        vs = 0.55 * prev.vs + 0.45 * raw_vs;
        vd = 0.55 * prev.vd + 0.45 * raw_vd;
      }
    }

    ObstacleTrack track;
    track.id = obs_id;
    track.x = x;
    track.y = y;
    track.s = projection.s;
    track.d = projection.d;
    track.yaw = yaw;
    track.vs = vs;
    track.vd = vd;
    track.radius = radius;
    track.stamp = stamp_sec;
    new_by_id[obs_id] = track;
  }

  prev_by_id_ = new_by_id;
  obstacles_.clear();
  obstacles_.reserve(prev_by_id_.size());
  for (const auto &pair : prev_by_id_) {
    obstacles_.push_back(pair.second);
  }
}

std::pair<Eigen::MatrixXd, PlanInfo> FrenetOvertakePlanner::plan(
    const State &vehicle_state, const Eigen::MatrixXd &nominal_ref_path) {
  const int short_count = config_.horizon_steps + 1;
  const double eval_horizon =
      std::max(config_.planning_time, config_.horizon_steps * config_.dt);
  const int count = static_cast<int>(std::floor(eval_horizon / config_.dt + 1e-6)) + 1;
  const Eigen::VectorXd times = linspace(0.0, config_.dt, count);

  const FrenetProjection ego_proj =
      reference_.projectXY(vehicle_state.x, vehicle_state.y);
  Eigen::VectorXd nominal_s;
  Eigen::VectorXd nominal_v;
  std::tie(nominal_s, nominal_v) =
      buildNominalProgress(ego_proj.s, vehicle_state.v, times);

  const auto predicted_obstacles = predictObstacles(times, ego_proj.s);
  const auto threats = findThreats(predicted_obstacles, nominal_s);

  const bool need_rejoin =
      std::abs(ego_proj.d) > 0.08 ||
      (last_mode_ != "nominal" && last_selected_d_.has_value());

  if (threats.empty() && !need_rejoin) {
    last_mode_ = "nominal";
    last_selected_d_.reset();
    last_selected_side_ = 0;
    PlanInfo info;
    info.mode = "nominal";
    info.reason = "clear";
    return {nominal_ref_path, info};
  }

  const PredictedObstacle *primary = nullptr;
  if (!threats.empty()) {
    primary = &*std::min_element(
        threats.begin(), threats.end(),
        [](const PredictedObstacle &a, const PredictedObstacle &b) {
          return a.ahead_now < b.ahead_now;
        });
  }

  std::vector<PlannerCandidate> candidates;
  candidates.push_back(
      buildFollowCandidate(ego_proj, vehicle_state.v, nominal_s, primary, times,
                           short_count));

  if (threats.empty() && need_rejoin) {
    candidates.push_back(buildRejoinCandidate(ego_proj, vehicle_state.v, nominal_s,
                                              times, short_count));
  } else if (primary != nullptr) {
    for (int side : {-1, 1}) {
      for (const double target_abs_d : config_.lateral_targets) {
        for (const double depart_time : config_.depart_times) {
          for (const double rejoin_time : config_.rejoin_times) {
            auto candidate = buildOvertakeCandidate(
                ego_proj, vehicle_state.v, nominal_s, nominal_v, *primary, times,
                short_count, side, target_abs_d, depart_time, rejoin_time);
            if (candidate) {
              candidates.push_back(*candidate);
            }
          }
        }
      }
    }
  }

  PlannerCandidate *best = nullptr;
  double best_cost = std::numeric_limits<double>::infinity();
  for (auto &candidate : candidates) {
    candidate.cost =
        scoreCandidate(candidate, primary, predicted_obstacles, nominal_s);
    if (candidate.cost < best_cost) {
      best_cost = candidate.cost;
      best = &candidate;
    }
  }

  if (best == nullptr || !std::isfinite(best->cost)) {
    last_mode_ = "nominal";
    last_selected_d_.reset();
    last_selected_side_ = 0;
    PlanInfo info;
    info.mode = "nominal";
    info.reason = "fallback";
    return {nominal_ref_path, info};
  }

  last_mode_ = best->mode;
  last_selected_d_ = best->d_traj;
  last_selected_side_ = best->side;

  PlanInfo info;
  info.mode = best->mode;
  info.label = best->label;
  info.cost = best->cost;
  info.side = best->side;
  return {best->ref_path, info};
}

std::vector<FrenetOvertakePlanner::PredictedObstacle>
FrenetOvertakePlanner::predictObstacles(const Eigen::VectorXd &times,
                                        double ego_s) const {
  std::vector<PredictedObstacle> predictions;
  for (const auto &obs : obstacles_) {
    const double ahead_now = reference_.forwardDistance(ego_s, obs.s);
    if (ahead_now > config_.activation_lookahead + 4.0) {
      continue;
    }

    PredictedObstacle pred;
    pred.track = obs;
    pred.ahead_now = ahead_now;
    pred.s = Eigen::VectorXd(times.size());
    pred.d = Eigen::VectorXd(times.size());
    pred.sigma_s = Eigen::VectorXd(times.size());
    pred.sigma_d = Eigen::VectorXd(times.size());

    const double base_s = ego_s + ahead_now;
    for (Eigen::Index i = 0; i < times.size(); ++i) {
      pred.s(i) = base_s + obs.vs * times(i);
      pred.d(i) = obs.d + obs.vd * times(i);
      pred.sigma_s(i) = config_.uncertainty_base_s +
                        config_.uncertainty_growth_s * times(i) +
                        0.10 * std::abs(obs.vs);
      pred.sigma_d(i) = config_.uncertainty_base_d +
                        config_.uncertainty_growth_d * times(i) +
                        0.10 * std::abs(obs.vd);
    }

    predictions.push_back(pred);
  }
  return predictions;
}

std::vector<FrenetOvertakePlanner::PredictedObstacle>
FrenetOvertakePlanner::findThreats(
    const std::vector<PredictedObstacle> &predicted_obstacles,
    const Eigen::VectorXd &nominal_s) const {
  std::vector<PredictedObstacle> threats;
  const Eigen::VectorXd nominal_d = Eigen::VectorXd::Zero(nominal_s.size());

  for (const auto &obs : predicted_obstacles) {
    if (obs.ahead_now <= 0.0 || obs.ahead_now > config_.activation_lookahead) {
      continue;
    }

    const double nominal_step =
        nominal_s.size() > 1 ? nominal_s(1) - nominal_s(0) : 0.0;
    if (obs.track.vs >
        std::max(0.0, nominal_step) / std::max(config_.dt, 1e-3)) {
      continue;
    }

    const Eigen::ArrayXd rel_s = nominal_s.array() - obs.s.array();
    const Eigen::ArrayXd rel_d = nominal_d.array() - obs.d.array();
    const Eigen::ArrayXd metric =
        (rel_s / (config_.activation_clearance_s + obs.sigma_s.array())).square() +
        (rel_d / (config_.activation_clearance_d + obs.sigma_d.array())).square();
    if ((metric < 1.5).any()) {
      threats.push_back(obs);
    }
  }

  return threats;
}

PlannerCandidate FrenetOvertakePlanner::buildFollowCandidate(
    const FrenetProjection &ego_proj, double ego_speed,
    const Eigen::VectorXd &nominal_s, const PredictedObstacle *primary,
    const Eigen::VectorXd &times, int short_count) const {
  double s_goal = nominal_s(nominal_s.size() - 1);
  if (primary != nullptr) {
    s_goal = std::min(s_goal, primary->s(primary->s.size() - 1) - config_.follow_gap);
  }
  s_goal = std::max(ego_proj.s + 0.15, s_goal);

  const QuinticPolynomial s_poly(
      ego_proj.s, std::max(ego_speed, 0.20), 0.0, s_goal,
      std::min(std::max(0.0, 0.5 * ego_speed), config_.max_speed), 0.0,
      times(times.size() - 1));

  Eigen::VectorXd s_traj(times.size());
  for (Eigen::Index i = 0; i < times.size(); ++i) {
    s_traj(i) = s_poly.calcPoint(times(i));
  }

  if (primary != nullptr) {
    const Eigen::VectorXd limit = primary->s.array() - config_.follow_gap;
    s_traj = s_traj.array().min(limit.array());
    s_traj(0) = ego_proj.s;
    for (Eigen::Index i = 1; i < s_traj.size(); ++i) {
      s_traj(i) = std::max(s_traj(i), s_traj(i - 1));
    }
  }

  const Eigen::VectorXd d_traj =
      buildCenterRejoinProfile(ego_proj.d, times, std::min(1.0, times(times.size() - 1)));
  const Eigen::MatrixXd ref_path =
      sdToRefPath(s_traj.head(short_count), d_traj.head(short_count));

  PlannerCandidate candidate;
  candidate.label = "follow_center";
  candidate.mode = "follow";
  candidate.side = 0;
  candidate.target_d = 0.0;
  candidate.s_traj = s_traj;
  candidate.d_traj = d_traj;
  candidate.ref_path = ref_path;
  candidate.rejoin_done = true;
  return candidate;
}

PlannerCandidate FrenetOvertakePlanner::buildRejoinCandidate(
    const FrenetProjection &ego_proj, double ego_speed,
    const Eigen::VectorXd &nominal_s, const Eigen::VectorXd &times,
    int short_count) const {
  const double s_goal = std::max(ego_proj.s + 0.20, nominal_s(nominal_s.size() - 1));
  const QuinticPolynomial s_poly(
      ego_proj.s, std::max(ego_speed, 0.20), 0.0, s_goal,
      std::min(config_.max_speed, std::max(ego_speed, 0.4)), 0.0,
      times(times.size() - 1));

  Eigen::VectorXd s_traj(times.size());
  for (Eigen::Index i = 0; i < times.size(); ++i) {
    s_traj(i) = s_poly.calcPoint(times(i));
  }

  const Eigen::VectorXd d_traj =
      buildCenterRejoinProfile(ego_proj.d, times, std::min(1.2, times(times.size() - 1)));
  const Eigen::MatrixXd ref_path =
      sdToRefPath(s_traj.head(short_count), d_traj.head(short_count));

  PlannerCandidate candidate;
  candidate.label = "rejoin_center";
  candidate.mode = "rejoin";
  candidate.side = 0;
  candidate.target_d = 0.0;
  candidate.s_traj = s_traj;
  candidate.d_traj = d_traj;
  candidate.ref_path = ref_path;
  candidate.rejoin_done = true;
  return candidate;
}

std::optional<PlannerCandidate> FrenetOvertakePlanner::buildOvertakeCandidate(
    const FrenetProjection &ego_proj, double ego_speed,
    const Eigen::VectorXd &nominal_s, const Eigen::VectorXd &nominal_v,
    const PredictedObstacle &primary, const Eigen::VectorXd &times,
    int short_count, int side, double target_abs_d, double depart_time,
    double rejoin_time) const {
  const double target_d = static_cast<double>(side) * target_abs_d;
  const double usable_limit =
      config_.track_half_width - config_.vehicle_half_width - 0.03;
  if (std::abs(target_d) > usable_limit) {
    return std::nullopt;
  }

  const double nominal_end =
      nominal_s(nominal_s.size() - 1) + config_.overtaking_speed_gain * times(times.size() - 1);
  const double pass_requirement =
      primary.s(primary.s.size() - 1) + config_.pass_front_gap + config_.post_pass_progress;
  const double max_reachable =
      ego_proj.s + config_.max_speed * times(times.size() - 1) * 0.98;
  const double s_goal = std::min(max_reachable, std::max(nominal_end, pass_requirement));
  if (s_goal <= ego_proj.s + 0.10) {
    return std::nullopt;
  }

  const double end_speed = std::min(
      config_.max_speed,
      std::max(nominal_v(nominal_v.size() - 1),
               ego_speed + config_.overtaking_speed_gain));

  const QuinticPolynomial s_poly(ego_proj.s, std::max(ego_speed, 0.25), 0.0,
                                 s_goal, end_speed, 0.0,
                                 times(times.size() - 1));

  Eigen::VectorXd s_traj(times.size());
  Eigen::VectorXd sdot(times.size());
  for (Eigen::Index i = 0; i < times.size(); ++i) {
    s_traj(i) = s_poly.calcPoint(times(i));
    sdot(i) = s_poly.calcFirstDerivative(times(i));
  }

  if ((sdot.array() < -0.05).any() ||
      vectorMax(sdot) > config_.max_speed * 1.25) {
    return std::nullopt;
  }

  std::optional<int> pass_idx = std::nullopt;
  for (Eigen::Index i = 0; i < s_traj.size(); ++i) {
    if (s_traj(i) >= primary.s(i) + config_.pass_front_gap) {
      pass_idx = static_cast<int>(i);
      break;
    }
  }

  const double pass_time =
      pass_idx ? times(*pass_idx) : times(times.size() - 1);
  const double depart_end =
      std::min(depart_time, std::max(0.45, pass_time - 0.25));
  if (depart_end >= times(times.size() - 1)) {
    return std::nullopt;
  }

  const double rejoin_start = std::max(depart_end + 0.20, pass_time + 0.25);
  const bool rejoin_done =
      rejoin_start + rejoin_time <= times(times.size() - 1) + 1e-6;

  const Eigen::VectorXd d_traj = buildOvertakeLateralProfile(
      ego_proj.d, target_d, times, depart_end, rejoin_start, rejoin_time);
  const Eigen::MatrixXd ref_path =
      sdToRefPath(s_traj.head(short_count), d_traj.head(short_count));

  std::ostringstream label;
  label.setf(std::ios::fixed);
  label.precision(2);
  label << "overtake_" << (side > 0 ? "left" : "right") << "_" << target_abs_d
        << "_" << depart_time;

  PlannerCandidate candidate;
  candidate.label = label.str();
  candidate.mode = "overtake";
  candidate.side = side;
  candidate.target_d = target_d;
  candidate.s_traj = s_traj;
  candidate.d_traj = d_traj;
  candidate.ref_path = ref_path;
  candidate.pass_idx = pass_idx;
  candidate.rejoin_done = rejoin_done;
  return candidate;
}

Eigen::VectorXd FrenetOvertakePlanner::buildCenterRejoinProfile(
    double current_d, const Eigen::VectorXd &times, double duration) const {
  const QuinticPolynomial poly(current_d, 0.0, 0.0, 0.0, 0.0, 0.0, duration);
  Eigen::VectorXd d_values(times.size());
  for (Eigen::Index i = 0; i < times.size(); ++i) {
    d_values(i) = poly.calcPoint(times(i));
  }
  return d_values;
}

Eigen::VectorXd FrenetOvertakePlanner::buildOvertakeLateralProfile(
    double current_d, double target_d, const Eigen::VectorXd &times,
    double depart_end, double rejoin_start, double rejoin_time) const {
  const QuinticPolynomial depart_poly(current_d, 0.0, 0.0, target_d, 0.0, 0.0,
                                      depart_end);
  const double rejoin_duration = std::max(rejoin_time, 1e-3);
  const QuinticPolynomial rejoin_poly(target_d, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      rejoin_duration);

  Eigen::VectorXd d_values = Eigen::VectorXd::Zero(times.size());
  for (Eigen::Index i = 0; i < times.size(); ++i) {
    const double t = times(i);
    if (t <= depart_end) {
      d_values(i) = depart_poly.calcPoint(t);
    } else if (t < rejoin_start) {
      d_values(i) = target_d;
    } else if (t <= rejoin_start + rejoin_duration) {
      d_values(i) = rejoin_poly.calcPoint(t - rejoin_start);
    } else {
      d_values(i) = 0.0;
    }
  }
  return d_values;
}

Eigen::MatrixXd FrenetOvertakePlanner::sdToRefPath(
    const Eigen::VectorXd &s_traj, const Eigen::VectorXd &d_traj) const {
  Eigen::VectorXd x_vals(s_traj.size());
  Eigen::VectorXd y_vals(s_traj.size());
  Eigen::VectorXd base_speed(s_traj.size());

  for (Eigen::Index i = 0; i < s_traj.size(); ++i) {
    double yaw_unused = 0.0;
    double kappa_unused = 0.0;
    std::tie(x_vals(i), y_vals(i), yaw_unused, kappa_unused) =
        reference_.frenetToCartesian(s_traj(i), d_traj(i));
    base_speed(i) = reference_.sampleSpeed(s_traj(i));
  }

  Eigen::VectorXd dx(s_traj.size());
  Eigen::VectorXd dy(s_traj.size());
  if (s_traj.size() > 0) {
    for (Eigen::Index i = 0; i < s_traj.size() - 1; ++i) {
      dx(i) = x_vals(i + 1) - x_vals(i);
      dy(i) = y_vals(i + 1) - y_vals(i);
    }
    dx(s_traj.size() - 1) = 0.0;
    dy(s_traj.size() - 1) = 0.0;
    if (s_traj.size() > 1) {
      dx(s_traj.size() - 1) = dx(s_traj.size() - 2);
      dy(s_traj.size() - 1) = dy(s_traj.size() - 2);
    }
  }

  Eigen::VectorXd yaw(s_traj.size());
  for (Eigen::Index i = 0; i < s_traj.size(); ++i) {
    yaw(i) = std::atan2(dy(i), dx(i));
  }
  yaw = unwrapAngles(yaw);

  Eigen::VectorXd speed = gradient(s_traj, config_.dt);
  for (Eigen::Index i = 0; i < speed.size(); ++i) {
    speed(i) = std::max(speed(i), 0.0);
  }

  Eigen::VectorXd speed_limit(s_traj.size());
  for (Eigen::Index i = 0; i < base_speed.size(); ++i) {
    speed_limit(i) = clipValue(base_speed(i) + config_.overtaking_speed_gain,
                               config_.min_speed, config_.max_speed);
  }

  for (Eigen::Index i = 0; i < speed.size(); ++i) {
    speed(i) = std::min(speed(i), speed_limit(i));
  }
  speed = clipVector(speed, config_.min_speed, config_.max_speed);

  Eigen::MatrixXd ref_path(4, s_traj.size());
  ref_path.row(0) = x_vals.transpose();
  ref_path.row(1) = y_vals.transpose();
  ref_path.row(2) = speed.transpose();
  ref_path.row(3) = yaw.transpose();
  return ref_path;
}

double FrenetOvertakePlanner::scoreCandidate(
    const PlannerCandidate &candidate, const PredictedObstacle *primary,
    const std::vector<PredictedObstacle> &predicted_obstacles,
    const Eigen::VectorXd &nominal_s) const {
  const double usable_limit =
      config_.track_half_width - config_.vehicle_half_width;
  const Eigen::ArrayXd wall_margin =
      usable_limit - candidate.d_traj.array().abs();
  if ((wall_margin < 0.0).any()) {
    return config_.collision_penalty;
  }

  double clearance_cost = 0.0;
  for (const auto &obs : predicted_obstacles) {
    const Eigen::ArrayXd rel_s = candidate.s_traj.array() - obs.s.array();
    const Eigen::ArrayXd rel_d = candidate.d_traj.array() - obs.d.array();
    const Eigen::ArrayXd safety_s =
        config_.collision_clearance_s + obs.sigma_s.array();
    const Eigen::ArrayXd safety_d =
        config_.collision_clearance_d + obs.sigma_d.array();
    const Eigen::ArrayXd metric =
        (rel_s / safety_s).square() + (rel_d / safety_d).square();
    if ((metric < 1.0).any()) {
      return config_.collision_penalty;
    }
    clearance_cost +=
        (1.0 / (metric - 1.0).max(0.05)).sum();
  }

  const Eigen::VectorXd d_dot = gradient(candidate.d_traj, config_.dt);
  const Eigen::VectorXd d_ddot = gradient(d_dot, config_.dt);
  const Eigen::VectorXd d_jerk = gradient(d_ddot, config_.dt);
  const Eigen::VectorXd s_dot = gradient(candidate.s_traj, config_.dt);
  const Eigen::VectorXd s_ddot = gradient(s_dot, config_.dt);

  const double wall_cost =
      (wall_margin.unaryExpr([this](double value) {
         return std::pow(std::max(0.0, config_.wall_buffer - value), 2);
       })).sum() /
      std::max(std::pow(config_.wall_buffer, 2), 1e-6);

  const double smooth_cost =
      d_jerk.array().square().sum() * config_.dt +
      0.25 * s_ddot.array().square().sum() * config_.dt;

  double efficiency_cost =
      std::max(0.0, nominal_s(nominal_s.size() - 1) -
                        candidate.s_traj(candidate.s_traj.size() - 1));
  if (candidate.mode == "overtake" && !candidate.pass_idx.has_value()) {
    efficiency_cost += config_.incomplete_pass_penalty;
  }

  double rejoin_cost = std::pow(candidate.d_traj(candidate.d_traj.size() - 1), 2) +
                       std::pow(d_dot(d_dot.size() - 1), 2);
  if (candidate.mode == "overtake" && !candidate.rejoin_done) {
    rejoin_cost += config_.incomplete_rejoin_penalty;
  }
  if (primary != nullptr && candidate.pass_idx.has_value()) {
    const double end_gap =
        candidate.s_traj(candidate.s_traj.size() - 1) -
        primary->s(primary->s.size() - 1);
    rejoin_cost += std::pow(std::max(0.0, config_.pass_front_gap - end_gap), 2);
  }

  double temporal_cost = 0.0;
  if (last_selected_d_) {
    const Eigen::Index count = std::min<Eigen::Index>(
        {config_.temporal_points, candidate.d_traj.size(), last_selected_d_->size()});
    if (count > 0) {
      temporal_cost +=
          (candidate.d_traj.head(count) - last_selected_d_->head(count))
              .array()
              .square()
              .mean();
    }
    if (last_selected_side_ != 0 && candidate.side != 0 &&
        last_selected_side_ != candidate.side) {
      temporal_cost += config_.side_flip_penalty;
    }
  }

  const double total_cost =
      config_.clearance_weight * clearance_cost +
      config_.wall_weight * wall_cost +
      config_.smooth_weight * smooth_cost +
      config_.efficiency_weight * efficiency_cost +
      config_.rejoin_weight * rejoin_cost +
      config_.temporal_weight * temporal_cost;
  return total_cost;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
FrenetOvertakePlanner::buildNominalProgress(double ego_s, double current_speed,
                                            const Eigen::VectorXd &times) const {
  Eigen::VectorXd s_profile = Eigen::VectorXd::Zero(times.size());
  Eigen::VectorXd v_profile = Eigen::VectorXd::Zero(times.size());

  s_profile(0) = ego_s;
  v_profile(0) =
      clipValue(std::max(current_speed, reference_.sampleSpeed(ego_s)),
                config_.min_speed, config_.max_speed);

  for (Eigen::Index i = 1; i < times.size(); ++i) {
    const double ref_speed = reference_.sampleSpeed(s_profile(i - 1));
    v_profile(i) =
        clipValue(std::max(ref_speed, config_.min_speed), config_.min_speed,
                  config_.max_speed);
    s_profile(i) = s_profile(i - 1) + v_profile(i - 1) * config_.dt;
  }

  return {s_profile, v_profile};
}

double FrenetOvertakePlanner::quatToYaw(
    const geometry_msgs::msg::Quaternion &quat_msg) {
  const double siny_cosp =
      2.0 * (quat_msg.w * quat_msg.z + quat_msg.x * quat_msg.y);
  const double cosy_cosp =
      1.0 - 2.0 * (quat_msg.y * quat_msg.y + quat_msg.z * quat_msg.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace mpc

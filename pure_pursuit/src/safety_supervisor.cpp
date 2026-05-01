#include "final_race_pure_pursuit/safety_supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mpc {

void SafetySupervisor::setParams(
    double ego_radius,
    double margin,
    double emergency_ttc,
    double emergency_distance,
    double slow_ttc,
    double slow_distance,
    double stop_hold_time) {
  ego_radius_ = ego_radius;
  margin_ = margin;
  emergency_ttc_ = emergency_ttc;
  emergency_distance_ = emergency_distance;
  slow_ttc_ = slow_ttc;
  slow_distance_ = slow_distance;
  stop_hold_time_ = stop_hold_time;
}

SafetyDecision SafetySupervisor::evaluate(
    const EgoSafetyState& ego,
    const std::vector<ObstacleSafetyState>& obstacles,
    double now_sec,
    bool planner_failed) {
  SafetyDecision decision;

  // Hold stop for a short time once emergency stop is triggered.
  if (stop_until_sec_ > now_sec) {
    decision.action = SafetyAction::STOP;
    decision.target_speed = 0.0;
    decision.reason = "holding_stop";
    return decision;
  }

  const double ego_vx = ego.speed * std::cos(ego.yaw);
  const double ego_vy = ego.speed * std::sin(ego.yaw);

  bool has_slow_risk = false;
  double slow_target_speed = ego.speed;

  for (const auto& obs : obstacles) {
    const double rx = obs.x - ego.x;
    const double ry = obs.y - ego.y;
    const double dist = std::hypot(rx, ry);

    if (dist < 1e-6) {
      continue;
    }

    const double rvx = obs.vx - ego_vx;
    const double rvy = obs.vy - ego_vy;

    const double closing_speed = -1.0 * (rx * rvx + ry * rvy) / dist;
    const double safety_radius = ego_radius_ + obs.radius + margin_;

    double ttc = std::numeric_limits<double>::infinity();
    if (closing_speed > 1e-3) {
      ttc = std::max(0.0, (dist - safety_radius) / closing_speed);
    }

    if (dist < decision.min_distance) {
      decision.min_distance = dist;
      decision.ttc = ttc;
      decision.obstacle_id = obs.id;
    }

    const bool emergency =
        (ttc < emergency_ttc_) ||
        (dist - safety_radius < emergency_distance_);

    if (emergency) {
      stop_until_sec_ = now_sec + stop_hold_time_;
      decision.action = SafetyAction::STOP;
      decision.target_speed = 0.0;
      decision.obstacle_id = obs.id;
      decision.min_distance = dist;
      decision.ttc = ttc;
      decision.reason = "emergency_collision_risk";
      return decision;
    }

    const bool slow_risk =
        planner_failed &&
        ((ttc < slow_ttc_) || (dist - safety_radius < slow_distance_));

    if (slow_risk) {
      has_slow_risk = true;

      const double obs_speed = std::hypot(obs.vx, obs.vy);
      const double target = std::max(0.3, obs_speed - 0.2);
      slow_target_speed = std::min(slow_target_speed, target);
    }
  }

  if (has_slow_risk) {
    decision.action = SafetyAction::SLOW_DOWN;
    decision.target_speed = slow_target_speed;
    decision.reason = "planner_failed_slow_down";
    return decision;
  }

  decision.action = SafetyAction::NONE;
  decision.reason = "clear";
  return decision;
}

}  // namespace mpc
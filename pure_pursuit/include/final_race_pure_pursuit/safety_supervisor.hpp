#pragma once

#include <string>
#include <vector>
#include <optional>

namespace mpc {

struct EgoSafetyState {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double speed = 0.0;
};

struct ObstacleSafetyState {
  int id = -1;
  double x = 0.0;
  double y = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double radius = 0.25;
};

enum class SafetyAction {
  NONE,
  SLOW_DOWN,
  STOP
};

struct SafetyDecision {
  SafetyAction action = SafetyAction::NONE;
  double target_speed = 0.0;
  int obstacle_id = -1;
  double min_distance = 1e9;
  double ttc = 1e9;
  std::string reason;
};

class SafetySupervisor {
 public:
  SafetySupervisor() = default;

  void setParams(
      double ego_radius,
      double margin,
      double emergency_ttc,
      double emergency_distance,
      double slow_ttc,
      double slow_distance,
      double stop_hold_time);

  SafetyDecision evaluate(
      const EgoSafetyState& ego,
      const std::vector<ObstacleSafetyState>& obstacles,
      double now_sec,
      bool planner_failed);

 private:
  double ego_radius_ = 0.25;
  double margin_ = 0.10;

  double emergency_ttc_ = 0.5;
  double emergency_distance_ = 0.20;

  double slow_ttc_ = 2.0;
  double slow_distance_ = 1.0;

  double stop_hold_time_ = 2.0;
  double stop_until_sec_ = -1.0;
};

}  // namespace mpc
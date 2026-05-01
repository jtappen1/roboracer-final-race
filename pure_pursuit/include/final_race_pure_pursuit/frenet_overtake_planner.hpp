#ifndef FINAL_RACE_PURE_PURSUIT_FRENET_OVERTAKE_PLANNER_HPP
#define FINAL_RACE_PURE_PURSUIT_FRENET_OVERTAKE_PLANNER_HPP

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "overtake_msgs/msg/tracked_obstacle_array.hpp"

#include "final_race_pure_pursuit/frenet_utils.hpp"
#include "final_race_pure_pursuit/mpc_utils.hpp"

namespace mpc {

struct OvertakePlannerConfig {
  int horizon_steps = 0;
  double dt = 0.1;
  double max_speed = 0.0;
  double min_speed = 0.0;
  double track_half_width = 0.7;
  double vehicle_half_width = 0.18;
  double obstacle_radius = 0.22;
  double activation_lookahead = 7.0;
  double activation_clearance_s = 1.05;
  double activation_clearance_d = 0.45;
  double collision_clearance_s = 0.65;
  double collision_clearance_d = 0.28;
  double follow_gap = 0.90;
  double pass_front_gap = 1.15;
  double post_pass_progress = 0.70;
  double uncertainty_base_s = 0.20;
  double uncertainty_base_d = 0.08;
  double uncertainty_growth_s = 0.08;
  double uncertainty_growth_d = 0.03;
  double wall_buffer = 0.3;
  double overtaking_speed_gain = 0.25;
  double min_overtake_speed_delta = 0.05;
  double planning_time = 7.0;
  std::vector<double> lateral_targets = {0.40, 0.54, 0.62};
  std::vector<double> depart_times = {0.70, 1.00};
  std::vector<double> rejoin_times = {0.80, 1.10};
  int temporal_points = 6;
  double collision_penalty = 1.0e6;
  double clearance_weight = 4.0;
  double wall_weight = 30.0;
  double smooth_weight = 1.0;
  double efficiency_weight = 3.0;
  double rejoin_weight = 14.0;
  double temporal_weight = 8.0;
  double side_flip_penalty = 10.0;
  double incomplete_pass_penalty = 16.0;
  double incomplete_rejoin_penalty = 12.0;
};

struct ObstacleTrack {
  int id = 0;
  double x = 0.0;
  double y = 0.0;
  double s = 0.0;
  double d = 0.0;
  double yaw = 0.0;
  double vs = 0.0;
  double vd = 0.0;
  double radius = 0.0;
  double stamp = 0.0;
};

struct PlannerCandidate {
  std::string label;
  std::string mode;
  int side = 0;
  double target_d = 0.0;
  Eigen::VectorXd s_traj;
  Eigen::VectorXd d_traj;
  Eigen::MatrixXd ref_path;
  std::optional<int> pass_idx = std::nullopt;
  bool rejoin_done = false;
  double cost = std::numeric_limits<double>::infinity();
};

struct PlanInfo {
  std::string mode = "nominal";
  std::string label;
  std::string reason;
  int side = 0;
  double cost = 0.0;
};

class QuinticPolynomial {
 public:
  QuinticPolynomial(double p0, double v0, double a0, double p1, double v1,
                    double a1, double duration);

  double calcPoint(double t) const;
  double calcFirstDerivative(double t) const;
  double calcSecondDerivative(double t) const;

 private:
  double clampT(double t) const;

  double duration_ = 0.0;
  double a0_ = 0.0;
  double a1_ = 0.0;
  double a2_ = 0.0;
  double a3_ = 0.0;
  double a4_ = 0.0;
  double a5_ = 0.0;
};

class FrenetOvertakePlanner {
 public:
  FrenetOvertakePlanner(const Eigen::VectorXd &ref_x, const Eigen::VectorXd &ref_y,
                        const Eigen::VectorXd &ref_yaw,
                        const Eigen::VectorXd &ref_v, double dt,
                        int horizon_steps, double max_speed, double min_speed,
                        double track_half_width);

  void reset();
  void updateObstacles(
      const overtake_msgs::msg::TrackedObstacleArray &obstacle_array_msg,
      double stamp_sec);

  std::pair<Eigen::MatrixXd, PlanInfo> plan(
      const State &vehicle_state, const Eigen::MatrixXd &nominal_ref_path);

 private:
  struct PredictedObstacle {
    ObstacleTrack track;
    Eigen::VectorXd s;
    Eigen::VectorXd d;
    Eigen::VectorXd sigma_s;
    Eigen::VectorXd sigma_d;
    double ahead_now = 0.0;
  };

  std::vector<PredictedObstacle> predictObstacles(const Eigen::VectorXd &times,
                                                  double ego_s) const;
  std::vector<PredictedObstacle> findThreats(
      const std::vector<PredictedObstacle> &predicted_obstacles,
      const Eigen::VectorXd &nominal_s) const;

  PlannerCandidate buildFollowCandidate(const FrenetProjection &ego_proj,
                                        double ego_speed,
                                        const Eigen::VectorXd &nominal_s,
                                        const PredictedObstacle *primary,
                                        const Eigen::VectorXd &times,
                                        int short_count) const;

  PlannerCandidate buildRejoinCandidate(const FrenetProjection &ego_proj,
                                        double ego_speed,
                                        const Eigen::VectorXd &nominal_s,
                                        const Eigen::VectorXd &times,
                                        int short_count) const;

  std::optional<PlannerCandidate> buildOvertakeCandidate(
      const FrenetProjection &ego_proj, double ego_speed,
      const Eigen::VectorXd &nominal_s, const Eigen::VectorXd &nominal_v,
      const PredictedObstacle &primary, const Eigen::VectorXd &times,
      int short_count, int side, double target_abs_d, double depart_time,
      double rejoin_time) const;

  Eigen::VectorXd buildCenterRejoinProfile(double current_d,
                                           const Eigen::VectorXd &times,
                                           double duration) const;
  Eigen::VectorXd buildOvertakeLateralProfile(double current_d, double target_d,
                                              const Eigen::VectorXd &times,
                                              double depart_end,
                                              double rejoin_start,
                                              double rejoin_time) const;
  Eigen::MatrixXd sdToRefPath(const Eigen::VectorXd &s_traj,
                              const Eigen::VectorXd &d_traj) const;
  double scoreCandidate(
      const PlannerCandidate &candidate, const PredictedObstacle *primary,
      const std::vector<PredictedObstacle> &predicted_obstacles,
      const Eigen::VectorXd &nominal_s) const;

  std::pair<Eigen::VectorXd, Eigen::VectorXd> buildNominalProgress(
      double ego_s, double current_speed, const Eigen::VectorXd &times) const;

  static double quatToYaw(const geometry_msgs::msg::Quaternion &quat_msg);

  ReferenceFrenetPath reference_;
  OvertakePlannerConfig config_;
  std::vector<ObstacleTrack> obstacles_;
  std::unordered_map<int, ObstacleTrack> prev_by_id_;
  std::optional<Eigen::VectorXd> last_selected_d_;
  int last_selected_side_ = 0;
  std::string last_mode_ = "nominal";
};

Eigen::VectorXd gradient(const Eigen::VectorXd &values, double dt);

}  // namespace mpc

#endif  // FINAL_RACE_PURE_PURSUIT_FRENET_OVERTAKE_PLANNER_HPP

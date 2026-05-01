#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "overtake_msgs/msg/tracked_obstacle.hpp"
#include "overtake_msgs/msg/tracked_obstacle_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "final_race_pure_pursuit/frenet_utils.hpp"
#include "final_race_pure_pursuit/mpc_utils.hpp"

namespace mpc {

namespace {

double clipValue(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

}  // namespace

struct ObstacleSpec {
  double gap_s = 0.0;
  double speed = 0.0;
  double base_d = 0.0;
  double drift_amp = 0.0;
  double drift_freq = 0.0;
  double drift_phase = 0.0;
  double radius = 0.22;
};

class SlowDynamicObstaclePublisher : public rclcpp::Node {
 public:
  SlowDynamicObstaclePublisher() : Node("slow_dynamic_obstacle_node") {
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("waypoint_file", "");
    declare_parameter<std::string>("odom_topic", "/ego_racecar/odom");
    declare_parameter<std::string>("initialpose_topic", "/initialpose");
    declare_parameter<std::string>("obstacle_topic", "/overtake/tracked_obstacles");
    declare_parameter<std::string>("marker_topic", "/overtake/obstacle_markers");
    declare_parameter<double>("publish_rate", 15.0);
    declare_parameter<double>("track_half_width", 0.85);
    declare_parameter<std::string>("scenario", "single_drifting");
    declare_parameter<int>("num_obstacles", 2);
    declare_parameter<double>("front_spawn_min", 3.5);
    declare_parameter<double>("front_gap_step", 1.7);
    declare_parameter<double>("base_speed", 0.45);
    declare_parameter<double>("speed_step", 0.08);
    declare_parameter<double>("base_drift_amplitude", 0.10);
    declare_parameter<double>("odom_reset_distance", 1.5);
    declare_parameter<double>("odom_reset_yaw", 0.9);

    global_frame_ = get_parameter("global_frame").as_string();
    track_half_width_ = get_parameter("track_half_width").as_double();
    scenario_ = get_parameter("scenario").as_string();
    num_obstacles_ = get_parameter("num_obstacles").as_int();
    front_spawn_min_ = get_parameter("front_spawn_min").as_double();
    front_gap_step_ = get_parameter("front_gap_step").as_double();
    base_speed_ = get_parameter("base_speed").as_double();
    speed_step_ = get_parameter("speed_step").as_double();
    base_drift_amplitude_ = get_parameter("base_drift_amplitude").as_double();
    odom_reset_distance_ = get_parameter("odom_reset_distance").as_double();
    odom_reset_yaw_ = get_parameter("odom_reset_yaw").as_double();

    std::string waypoint_file = get_parameter("waypoint_file").as_string();
    if (waypoint_file.empty()) {
      waypoint_file = ament_index_cpp::get_package_share_directory(
                          "final_race_pure_pursuit") +
                      "/path/waypoints.csv";
    }

    const Eigen::MatrixXd waypoints = loadCsv(waypoint_file);
    if (waypoints.rows() == 0 || waypoints.cols() < 2) {
      throw std::runtime_error("Waypoint CSV must contain at least x,y columns.");
    }

    Eigen::VectorXd cx = waypoints.col(0);
    Eigen::VectorXd cy = waypoints.col(1);
    Eigen::VectorXd dx(cx.size());
    Eigen::VectorXd dy(cy.size());
    Eigen::VectorXd cyaw(cx.size());
    for (Eigen::Index i = 0; i < cx.size(); ++i) {
      const Eigen::Index j = (i + 1) % cx.size();
      dx(i) = cx(j) - cx(i);
      dy(i) = cy(j) - cy(i);
      cyaw(i) = std::atan2(dy(i), dx(i));
    }
    reference_ =
        std::make_unique<ReferenceFrenetPath>(cx, cy, cyaw, std::nullopt);

    obstacle_pub_ = create_publisher<overtake_msgs::msg::TrackedObstacleArray>(
        get_parameter("obstacle_topic").as_string(), 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        get_parameter("marker_topic").as_string(), 10);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        get_parameter("odom_topic").as_string(), 10,
        std::bind(&SlowDynamicObstaclePublisher::odomCallback, this,
                  std::placeholders::_1));
    initialpose_sub_ =
        create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            get_parameter("initialpose_topic").as_string(), 10,
            std::bind(&SlowDynamicObstaclePublisher::initialposeCallback, this,
                      std::placeholders::_1));

    const double publish_rate =
        std::max(get_parameter("publish_rate").as_double(), 1.0);
    timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / publish_rate),
        std::bind(&SlowDynamicObstaclePublisher::publishObstacles, this));

    RCLCPP_INFO(get_logger(),
                "Slow obstacle publisher ready | scenario=%s | topic=%s",
                scenario_.c_str(),
                get_parameter("obstacle_topic").as_string().c_str());
  }

 private:
  static std::tuple<double, double, double, double> yawToQuat(double yaw) {
    const double half = 0.5 * yaw;
    return {0.0, 0.0, std::sin(half), std::cos(half)};
  }

  void initialposeCallback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    resetScenario(msg->pose.pose, "initialpose");
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const geometry_msgs::msg::Pose &pose = msg->pose.pose;
    const double yaw = yawFromQuaternion(pose.orientation);

    if (!spawn_time_) {
      resetScenario(pose, "startup");
    } else if (last_odom_pose_ && last_odom_yaw_) {
      const double dx = pose.position.x - last_odom_pose_->position.x;
      const double dy = pose.position.y - last_odom_pose_->position.y;
      const double jump_dist = std::hypot(dx, dy);
      const double jump_yaw =
          std::abs(std::atan2(std::sin(yaw - *last_odom_yaw_),
                              std::cos(yaw - *last_odom_yaw_)));
      if (jump_dist > odom_reset_distance_ || jump_yaw > odom_reset_yaw_) {
        resetScenario(pose, "odom_jump");
      }
    }

    last_odom_pose_ = pose;
    last_odom_yaw_ = yaw;
  }

  void resetScenario(const geometry_msgs::msg::Pose &anchor_pose,
                     const std::string &reason) {
    const FrenetProjection anchor_proj =
        reference_->projectXY(anchor_pose.position.x, anchor_pose.position.y);
    spawn_anchor_s_ = anchor_proj.s;
    spawn_time_ = now();
    obstacle_specs_ = buildScenarioSpecs();

    RCLCPP_INFO(get_logger(),
                "Obstacle scenario reset from %s | count=%ld | anchor_s=%.2f",
                reason.c_str(), static_cast<long>(obstacle_specs_.size()),
                spawn_anchor_s_);
  }

  std::vector<ObstacleSpec> buildScenarioSpecs() const {
    std::vector<ObstacleSpec> specs;
    const double max_offset = track_half_width_ - 0.18;

    if (scenario_ == "single_slow_center") {
      specs.push_back({front_spawn_min_, base_speed_, 0.0, 0.0, 0.0, 0.0, 0.22});
      return specs;
    }

    if (scenario_ == "single_drifting") {
      specs.push_back({front_spawn_min_,
                       base_speed_,
                       0.05,
                       std::min(base_drift_amplitude_, max_offset),
                       0.22,
                       0.0,
                       0.22});
      return specs;
    }

    struct RawSpec {
      double gap_s;
      double speed;
      double base_d;
      double drift_amp;
      double drift_freq;
      double drift_phase;
    };

    std::vector<RawSpec> raw_specs;
    if (scenario_ == "double_file") {
      raw_specs.push_back(
          {front_spawn_min_, base_speed_, -0.12, 0.06, 0.18, 0.0});
      raw_specs.push_back({front_spawn_min_ + front_gap_step_, base_speed_ - 0.08,
                           0.18, 0.10, 0.16, 1.1});
    } else {
      for (int i = 0; i < std::max(1, num_obstacles_); ++i) {
        const double side = (i % 2 == 0) ? -1.0 : 1.0;
        raw_specs.push_back(
            {front_spawn_min_ + i * front_gap_step_,
             std::max(0.20, base_speed_ - i * speed_step_),
             side * std::min(0.18 + 0.05 * i, max_offset),
             std::min(base_drift_amplitude_ + 0.03 * i, 0.16), 0.14 + 0.03 * i,
             0.8 * i});
      }
    }

    for (int i = 0; i < std::min<int>(num_obstacles_, raw_specs.size()); ++i) {
      const RawSpec &raw = raw_specs[static_cast<std::size_t>(i)];
      ObstacleSpec spec;
      spec.gap_s = raw.gap_s;
      spec.speed = raw.speed;
      spec.base_d = clipValue(raw.base_d, -max_offset, max_offset);
      spec.drift_amp = std::min(raw.drift_amp, max_offset);
      spec.drift_freq = raw.drift_freq;
      spec.drift_phase = raw.drift_phase;
      spec.radius = 0.22;
      specs.push_back(spec);
    }
    return specs;
  }

  void publishObstacles() {
    if (!spawn_time_ || obstacle_specs_.empty()) {
      return;
    }

    const rclcpp::Time current_time = now();
    const double elapsed = (current_time - *spawn_time_).seconds();

    overtake_msgs::msg::TrackedObstacleArray obstacle_array;
    obstacle_array.header.frame_id = global_frame_;
    obstacle_array.header.stamp = current_time;

    visualization_msgs::msg::MarkerArray marker_array;

    for (std::size_t idx = 0; idx < obstacle_specs_.size(); ++idx) {
      const ObstacleSpec &spec = obstacle_specs_[idx];
      double d = spec.base_d;
      if (spec.drift_amp > 0.0 && spec.drift_freq > 0.0) {
        d += spec.drift_amp *
             std::sin(2.0 * kPi * spec.drift_freq * elapsed + spec.drift_phase);
      }

      const double d_limit = track_half_width_ - spec.radius - 0.05;
      d = clipValue(d, -d_limit, d_limit);

      const double s = spawn_anchor_s_ + spec.gap_s + spec.speed * elapsed;

      double x = 0.0;
      double y = 0.0;
      double center_yaw = 0.0;
      double kappa_unused = 0.0;
      std::tie(x, y, center_yaw, kappa_unused) =
          reference_->frenetToCartesian(s, d);

      const double ds_dt = spec.speed;
      double dd_dt = 0.0;
      if (spec.drift_amp > 0.0 && spec.drift_freq > 0.0) {
        dd_dt = spec.drift_amp * 2.0 * kPi * spec.drift_freq *
                std::cos(2.0 * kPi * spec.drift_freq * elapsed + spec.drift_phase);
      }

      const double yaw = center_yaw + std::atan2(dd_dt, std::max(ds_dt, 0.10));
      const double cos_c = std::cos(center_yaw);
      const double sin_c = std::sin(center_yaw);
      const double vx_world = ds_dt * cos_c - dd_dt * sin_c;
      const double vy_world = ds_dt * sin_c + dd_dt * cos_c;

      geometry_msgs::msg::Pose pose;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = 0.0;
      std::tie(pose.orientation.x, pose.orientation.y, pose.orientation.z,
               pose.orientation.w) = yawToQuat(yaw);

      overtake_msgs::msg::TrackedObstacle tracked;
      tracked.id = static_cast<int32_t>(idx);
      tracked.cls = -1;
      tracked.pose = pose;
      tracked.velocity.x = vx_world;
      tracked.velocity.y = vy_world;
      tracked.velocity.z = 0.0;
      tracked.radius = static_cast<float>(spec.radius);
      obstacle_array.obstacles.push_back(tracked);

      visualization_msgs::msg::Marker marker;
      marker.header = obstacle_array.header;
      marker.ns = "slow_obstacles";
      marker.id = static_cast<int32_t>(idx);
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = pose;
      marker.scale.x = spec.radius * 2.0;
      marker.scale.y = spec.radius * 2.0;
      marker.scale.z = 0.35;
      marker.color.r = 1.0;
      marker.color.g = 0.45;
      marker.color.b = 0.10;
      marker.color.a = 0.95;
      marker_array.markers.push_back(marker);
    }

    obstacle_pub_->publish(obstacle_array);
    marker_pub_->publish(marker_array);
  }

  std::string global_frame_;
  double track_half_width_ = 0.85;
  std::string scenario_;
  int num_obstacles_ = 2;
  double front_spawn_min_ = 3.5;
  double front_gap_step_ = 1.7;
  double base_speed_ = 0.45;
  double speed_step_ = 0.08;
  double base_drift_amplitude_ = 0.10;
  double odom_reset_distance_ = 1.5;
  double odom_reset_yaw_ = 0.9;

  std::unique_ptr<ReferenceFrenetPath> reference_;
  std::vector<ObstacleSpec> obstacle_specs_;
  double spawn_anchor_s_ = 0.0;
  std::optional<rclcpp::Time> spawn_time_;
  std::optional<geometry_msgs::msg::Pose> last_odom_pose_;
  std::optional<double> last_odom_yaw_;

  rclcpp::Publisher<overtake_msgs::msg::TrackedObstacleArray>::SharedPtr
      obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initialpose_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace mpc

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mpc::SlowDynamicObstaclePublisher>());
  rclcpp::shutdown();
  return 0;
}

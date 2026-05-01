#ifndef FINAL_RACE_PURE_PURSUIT_MPC_UTILS_HPP
#define FINAL_RACE_PURE_PURSUIT_MPC_UTILS_HPP

#include <string>

#include <Eigen/Core>

#include "geometry_msgs/msg/quaternion.hpp"

namespace mpc {

constexpr double kPi = 3.14159265358979323846;

struct State {
  double x = 0.0;
  double y = 0.0;
  double v = 0.0;
  double yaw = 0.0;
};

double clamp(double value, double lo, double hi);
double normalizeAngle(double angle);
double yawFromQuaternion(const geometry_msgs::msg::Quaternion &q);
Eigen::VectorXd unwrapAngles(const Eigen::VectorXd &angles);
Eigen::MatrixXd loadCsv(const std::string &path);

}  // namespace mpc

#endif  // FINAL_RACE_PURE_PURSUIT_MPC_UTILS_HPP

#include "final_race_pure_pursuit/mpc_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mpc {

double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

double normalizeAngle(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion &q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

Eigen::VectorXd unwrapAngles(const Eigen::VectorXd &angles) {
  if (angles.size() == 0) {
    return angles;
  }

  Eigen::VectorXd unwrapped = angles;
  for (Eigen::Index i = 1; i < angles.size(); ++i) {
    const double delta = normalizeAngle(unwrapped(i) - unwrapped(i - 1));
    unwrapped(i) = unwrapped(i - 1) + delta;
  }
  return unwrapped;
}

Eigen::MatrixXd loadCsv(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }

  std::vector<std::vector<double>> rows;
  std::string line;
  std::size_t cols = 0;

  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell));
    }
    if (row.empty()) {
      continue;
    }
    if (cols == 0) {
      cols = row.size();
    } else if (row.size() != cols) {
      throw std::runtime_error("Inconsistent CSV column count in " + path);
    }
    rows.push_back(row);
  }

  Eigen::MatrixXd matrix(rows.size(), static_cast<Eigen::Index>(cols));
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(rows.size()); ++i) {
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(cols); ++j) {
      matrix(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
  return matrix;
}

}  // namespace mpc

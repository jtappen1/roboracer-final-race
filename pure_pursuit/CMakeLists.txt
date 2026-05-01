cmake_minimum_required(VERSION 3.5)
project(final_race_pure_pursuit)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(nanoflann QUIET)
if(NOT nanoflann_FOUND)
  message(STATUS "nanoflann not found via cmake — assuming nanoflann.hpp is local")
endif()

# Find dependencies
foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  ament_index_cpp
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2_ros
  overtake_msgs)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Include Cpp "include" directory
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

# Create Cpp executable
add_executable(final_race_pure_pursuit_node
  src/pure_pursuit_node.cpp
  src/frenet_overtake_planner.cpp
  src/frenet_utils.cpp
  src/utils.cpp
  src/safety_supervisor.cpp)

ament_target_dependencies(final_race_pure_pursuit_node
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros overtake_msgs ament_index_cpp
)

# Optional: build the slow obstacle simulator alongside (sim-only).
add_executable(final_race_slow_obstacle_node
  src/slow_dynamic_obstacle_node.cpp
  src/frenet_utils.cpp
  src/utils.cpp)
ament_target_dependencies(final_race_slow_obstacle_node
  rclcpp geometry_msgs nav_msgs visualization_msgs overtake_msgs
)

# Install Cpp executables
install(TARGETS
  final_race_pure_pursuit_node
  final_race_slow_obstacle_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/pure_pursuit_node.py
  scripts/collect_waypoints.py
  DESTINATION lib/${PROJECT_NAME}
)

# install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY path
  DESTINATION share/${PROJECT_NAME}
)

target_include_directories(final_race_pure_pursuit_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()

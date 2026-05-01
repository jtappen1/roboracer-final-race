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
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)
endforeach()

# Include Cpp "include" directory
include_directories(include)
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

# Create Cpp executables (Fixed: separated into distinct targets)
add_executable(final_race_pure_pursuit_node src/pure_pursuit_node.cpp)
add_executable(object_detector_node src/object_detector.cpp)

# Link dependencies to their respective targets
ament_target_dependencies(final_race_pure_pursuit_node
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros
)

ament_target_dependencies(object_detector_node
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros
)

# Install Cpp executables
install(TARGETS
  final_race_pure_pursuit_node
  object_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/pure_pursuit_node.py
  scripts/collect_waypoints.py
  scripts/obstacle_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install path files
install(DIRECTORY path
  DESTINATION share/${PROJECT_NAME}
)

# Apply include directories to both nodes
target_include_directories(final_race_pure_pursuit_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(object_detector_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
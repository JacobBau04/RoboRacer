cmake_minimum_required(VERSION 3.8)
project(mppi)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  ament_index_cpp
  rclpy
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  ackermann_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
  find_package(${PACKAGE} REQUIRED)
endforeach()

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/mppi_node.py
  scripts/utils.py
  DESTINATION lib/${PROJECT_NAME}
)

# ==========================================
# Opponent LiDAR Detector
# ==========================================

add_executable(opponent_lidar_detector_node
  scripts/opponent_lidar_detector_node.cpp
)

ament_target_dependencies(opponent_lidar_detector_node
  rclcpp
  ament_index_cpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
)

# ==========================================
# Opponent Predictor
# ==========================================

add_executable(opponent_predictor_node
  scripts/opponent_predictor_node.cpp
)

ament_target_dependencies(opponent_predictor_node
  rclcpp
  ament_index_cpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install C++ executables
install(TARGETS
  opponent_lidar_detector_node
  opponent_predictor_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install waypoint files
install(DIRECTORY waypoints
  DESTINATION share/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
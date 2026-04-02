cmake_minimum_required(VERSION 3.8)
project(so101_moveit_config)

find_package(ament_cmake REQUIRED)

# Install config files and launch files into the ROS2 install space
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

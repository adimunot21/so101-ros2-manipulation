cmake_minimum_required(VERSION 3.8)
project(so101_description)

find_package(ament_cmake REQUIRED)

# Install directories into the ROS2 install space.
# After "colcon build", ROS2 can find these via $(find so101_description)/...
#
# Why each directory:
#   urdf/    — robot description (URDF/xacro) read by robot_state_publisher
#   meshes/  — 3D mesh files (.stl) referenced by the URDF
#   mjcf/    — MuJoCo model files loaded by mujoco_ros2_control
#   launch/  — launch files that start nodes
#   rviz/    — saved RViz2 layouts
#   config/  — controller configs read by ros2_control

install(DIRECTORY urdf meshes mjcf launch rviz config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

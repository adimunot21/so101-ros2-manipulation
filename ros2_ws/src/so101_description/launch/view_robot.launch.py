"""Launch file to view the SO-101 robot in RViz2 with joint sliders.

This is a "visualization only" launch — no physics, no controllers.
It loads the URDF, starts robot_state_publisher (publishes TF transforms),
joint_state_publisher_gui (interactive joint sliders), and RViz2.

Purpose: Verify the URDF loads correctly, meshes render, and joints move.

Usage:
    ros2 launch so101_description view_robot.launch.py
"""
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Locate package files ──────────────────────────────────────────
    # FindPackageShare finds the installed package directory.
    # After "colcon build", this resolves to:
    #   ~/so101_ros2_manip/ros2_ws/install/so101_description/share/so101_description
    pkg_share = FindPackageShare("so101_description")

    # Path to the xacro file (our modified URDF with ros2_control tags)
    xacro_path = PathJoinSubstitution([pkg_share, "urdf", "so101.urdf.xacro"])

    # Path to saved RViz2 layout
    rviz_config_path = PathJoinSubstitution([pkg_share, "rviz", "view_robot.rviz"])

    # ── Process xacro → URDF string ──────────────────────────────────
    # Command() runs a shell command at launch time.
    # "xacro" processes .xacro files into plain URDF XML.
    # ParameterValue(..., value_type=str) tells ROS2 "this is a string,
    # don't try to parse it as YAML." Without this wrapper, ROS2 Jazzy
    # sees the XML angle brackets and crashes trying to interpret them.
    robot_description = ParameterValue(
        Command(["xacro ", xacro_path]),
        value_type=str,
    )

    # ── Node: robot_state_publisher ───────────────────────────────────
    # Reads the URDF, subscribes to /joint_states, publishes TF transforms.
    # This is how RViz2 knows the 3D pose of every link on the robot.
    # We do NOT write this node — it comes from the robot_state_publisher package.
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    # ── Node: joint_state_publisher_gui ───────────────────────────────
    # Opens a small GUI window with sliders for each joint.
    # Publishes slider values on /joint_states.
    # In Checkpoint 1B, we replace this with MuJoCo (which publishes
    # real simulated joint states instead of slider values).
    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen",
    )

    # ── Node: RViz2 ──────────────────────────────────────────────────
    # 3D visualization tool. Subscribes to TF transforms and renders
    # the robot model. The --display-config loads our saved layout.
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_path],
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ])

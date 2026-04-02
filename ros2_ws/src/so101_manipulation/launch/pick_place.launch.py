"""Launch pick-and-place with perception.

Requires sim.launch.py and moveit.launch.py already running.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    perception_config = PathJoinSubstitution([
        FindPackageShare("so101_perception"), "config", "perception.yaml"
    ])
    manipulation_config = PathJoinSubstitution([
        FindPackageShare("so101_manipulation"), "config", "manipulation.yaml"
    ])

    return LaunchDescription([
        # Perception node (detects objects)
        Node(
            package="so101_perception",
            executable="detection_node",
            parameters=[perception_config, {"use_sim_time": True}],
            output="screen",
        ),
        # Pick-and-place state machine
        Node(
            package="so101_manipulation",
            executable="pick_place_node",
            parameters=[manipulation_config, {"use_sim_time": True}],
            output="screen",
        ),
    ])

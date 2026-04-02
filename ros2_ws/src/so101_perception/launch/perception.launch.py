"""Launch the perception pipeline."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("so101_perception"), "config", "perception.yaml"
    ])

    return LaunchDescription([
        Node(
            package="so101_perception",
            executable="detection_node",
            parameters=[config, {"use_sim_time": True}],
            output="screen",
        ),
    ])

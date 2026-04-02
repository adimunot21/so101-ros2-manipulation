"""Launch MoveIt2 move_group and RViz2 for SO-101 motion planning.

Run ALONGSIDE sim.launch.py:
  Terminal 1: ros2 launch so101_description sim.launch.py
  Terminal 2: ros2 launch so101_moveit_config moveit.launch.py
"""
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc_pkg = FindPackageShare("so101_description")
    moveit_pkg = FindPackageShare("so101_moveit_config")

    xacro_path = PathJoinSubstitution([desc_pkg, "urdf", "so101.urdf.xacro"])
    srdf_path = PathJoinSubstitution([moveit_pkg, "config", "so101.srdf"])
    joint_limits_path = PathJoinSubstitution([moveit_pkg, "config", "joint_limits.yaml"])
    ompl_planning_path = PathJoinSubstitution([moveit_pkg, "config", "ompl_planning.yaml"])
    moveit_controllers_path = PathJoinSubstitution(
        [moveit_pkg, "config", "moveit_controllers.yaml"]
    )

    robot_description = ParameterValue(
        Command(["xacro ", xacro_path]),
        value_type=str,
    )
    robot_description_semantic = ParameterValue(
        Command(["cat ", srdf_path]),
        value_type=str,
    )

    # Kinematics as inline dict — NOT a YAML file.
    # position_only_ik: SO-101 has 5 DOF, cannot solve full 6D pose.
    # Solves for XYZ position only, orientation is free.
    kinematics_config = {
        "robot_description_kinematics": {
            "arm": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.2,
                "kinematics_solver_attempts": 10,
                "position_only_ik": True,
            },
        },
    }

    shared_params = {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "use_sim_time": True,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            shared_params,
            kinematics_config,
            {
                "publish_robot_description_semantic": True,
                "planning_scene_monitor_options": {
                    "robot_description": "robot_description",
                    "joint_state_topic": "/joint_states",
                    "attached_collision_object_topic": "/attached_collision_objects",
                    "publish_planning_scene_topic": "/monitored_planning_scene",
                    "monitored_planning_scene_topic": "/monitored_planning_scene",
                    "wait_for_initial_state_timeout": 10.0,
                },
            },
            joint_limits_path,
            ompl_planning_path,
            moveit_controllers_path,
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        parameters=[
            shared_params,
            kinematics_config,
            joint_limits_path,
            ompl_planning_path,
            moveit_controllers_path,
        ],
    )

    return LaunchDescription([
        move_group_node,
        rviz_node,
    ])

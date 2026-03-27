"""Launch SO-101 in MuJoCo simulation with ros2_control.

This launch file starts:
  1. ros2_control_node — runs MuJoCo physics and the controller manager
  2. robot_state_publisher — publishes TF transforms from joint states
  3. Controller spawners — load and activate controllers
  4. RViz2 — 3D visualization

Data flow:
  MuJoCo (physics) ←→ ros2_control_node ←→ controllers
                                              ↓
                                        /joint_states
                                              ↓
                                   robot_state_publisher
                                              ↓
                                        TF transforms
                                              ↓
                                           RViz2

Usage:
    ros2 launch so101_description sim.launch.py
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Locate package files ──────────────────────────────────────────
    pkg_share = FindPackageShare("so101_description")
    xacro_path = PathJoinSubstitution([pkg_share, "urdf", "so101.urdf.xacro"])
    controllers_config = PathJoinSubstitution([pkg_share, "config", "controllers.yaml"])
    rviz_config_path = PathJoinSubstitution([pkg_share, "rviz", "view_robot.rviz"])

    # ── Process xacro → URDF ─────────────────────────────────────────
    robot_description = ParameterValue(
        Command(["xacro ", xacro_path]),
        value_type=str,
    )

    # ── Node: robot_state_publisher ───────────────────────────────────
    # Same as in view_robot.launch.py — publishes TF from joint states.
    # The key difference: joint states now come from MuJoCo (via
    # joint_state_broadcaster) instead of the slider GUI.
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": True},
        ],
        output="screen",
    )

    # ── Node: ros2_control_node ───────────────────────────────────────
    # This is the core of ros2_control. It:
    #   1. Loads the hardware plugin (mujoco_ros2_control/MujocoSystemInterface)
    #   2. Reads the URDF to know which joints exist
    #   3. Starts MuJoCo simulation internally
    #   4. Runs the controller_manager to handle controller lifecycle
    #
    # Parameters:
    #   robot_description — the URDF (tells it about joints)
    #   controllers_config — which controllers to load (from YAML)
    #   use_sim_time — use MuJoCo's simulation clock, not wall clock
    ros2_control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            controllers_config,
            {"use_sim_time": True},
        ],
        output="screen",
    )

    # ── Controller spawners ──────────────────────────────────────────
    # "Spawning" a controller means: tell controller_manager to load
    # the controller from config, configure it, and activate it.
    #
    # We use ExecuteProcess to call the `ros2 control` CLI tool.
    # The --controller-manager flag tells it which node to talk to.
    #
    # Order matters: joint_state_broadcaster MUST start first because
    # other controllers and robot_state_publisher need /joint_states.

    spawn_jsb = ExecuteProcess(
        cmd=[
            "ros2", "control", "load_controller", "--set-state", "active",
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # Spawn arm_controller AFTER joint_state_broadcaster is active.
    # We use a TimerAction as a simple delay — a more robust approach
    # would use OnProcessExit, but for sim startup a delay works fine.
    spawn_arm = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "control", "load_controller", "--set-state", "active",
                    "arm_controller",
                    "--controller-manager", "/controller_manager",
                ],
                output="screen",
            ),
        ],
    )

    spawn_gripper = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "control", "load_controller", "--set-state", "active",
                    "gripper_controller",
                    "--controller-manager", "/controller_manager",
                ],
                output="screen",
            ),
        ],
    )

    # ── Node: RViz2 ──────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_path],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([
        ros2_control_node,
        robot_state_publisher_node,
        spawn_jsb,
        spawn_arm,
        spawn_gripper,
        rviz_node,
    ])

"""Pick-and-place state machine for SO-101.

Key design insight: for a 5-DOF arm, Cartesian goals give unpredictable
orientations (position_only_ik). So we use Cartesian for the APPROACH
(to get roughly above the object), then switch to JOINT-SPACE for the
DESCEND (preserving the approach orientation by reading current joints
and adjusting only what's needed to lower the arm).

State machine:
  IDLE → MOVE_HOME → PERCEIVE → OPEN_GRIPPER → APPROACH →
  DESCEND → CLOSE_GRIPPER → LIFT → MOVE_TO_PLACE →
  LOWER_TO_PLACE → RELEASE → RETRACT → DONE

Usage:
    T1: ros2 launch so101_description sim.launch.py
    T2: ros2 launch so101_moveit_config moveit.launch.py
    T3: ros2 launch so101_manipulation pick_place.launch.py
"""
from __future__ import annotations

import enum
import math
import time
from typing import Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint


class State(enum.Enum):
    IDLE = "IDLE"
    MOVE_HOME = "MOVE_HOME"
    PERCEIVE = "PERCEIVE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    APPROACH = "APPROACH"
    RECORD_APPROACH_JOINTS = "RECORD_APPROACH_JOINTS"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    MOVE_TO_PLACE = "MOVE_TO_PLACE"
    LOWER_TO_PLACE = "LOWER_TO_PLACE"
    RELEASE = "RELEASE"
    RETRACT = "RETRACT"
    DONE = "DONE"
    ERROR = "ERROR"


class PickPlaceNode(Node):

    def __init__(self) -> None:
        super().__init__("pick_place_node")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("camera_to_base.x", 0.12)
        self.declare_parameter("camera_to_base.y", 0.10)
        self.declare_parameter("camera_to_base.z", 0.70)
        self.declare_parameter("grasp.approach_height", 0.10)
        self.declare_parameter("grasp.grasp_height", 0.03)
        self.declare_parameter("grasp.lift_height", 0.12)
        self.declare_parameter("grasp.gripper_open", 1.0)
        self.declare_parameter("grasp.gripper_closed", -0.1)
        self.declare_parameter("place.x", 0.15)
        self.declare_parameter("place.y", -0.06)
        self.declare_parameter("place.z", 0.03)
        self.declare_parameter("poses.home", [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("trajectory_time_sec", 3.0)
        self.declare_parameter("settle_time_sec", 0.5)

        # ── State ─────────────────────────────────────────────────────
        self._state = State.IDLE
        self._latched_object_pose: Optional[list] = None
        self._latest_detection: Optional[list] = None
        self._current_arm_joints: Optional[dict] = None
        self._approach_joints: Optional[list] = None  # Saved after APPROACH
        self._action_in_progress = False

        self._arm_joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll",
        ]

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, "/detected_objects", self._detection_cb, 10,
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10,
        )

        # ── Action clients ────────────────────────────────────────────
        self._arm_action = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self._gripper_action = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self._move_group_action = ActionClient(
            self, MoveGroup, "/move_action",
        )

        self._timer = self.create_timer(0.5, self._tick)
        self.get_logger().info("Pick-and-place node started — waiting in IDLE")

    # ── Callbacks ─────────────────────────────────────────────────────

    def _detection_cb(self, msg: PoseStamped) -> None:
        cam_x = self.get_parameter("camera_to_base.x").value
        cam_y = self.get_parameter("camera_to_base.y").value
        cam_z = self.get_parameter("camera_to_base.z").value
        obj = msg.pose.position
        self._latest_detection = [
            cam_x + obj.x,
            cam_y - obj.y,
            cam_z - obj.z,
        ]

    def _joint_state_cb(self, msg: JointState) -> None:
        """Store current joint positions as a dict for easy lookup."""
        self._current_arm_joints = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self._arm_joint_names:
                self._current_arm_joints[name] = pos

    def _get_current_arm_list(self) -> Optional[list]:
        """Get current arm joints as ordered list."""
        if self._current_arm_joints is None:
            return None
        try:
            return [self._current_arm_joints[j] for j in self._arm_joint_names]
        except KeyError:
            return None

    # ── State machine ─────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._action_in_progress:
            return

        state = self._state

        if state == State.IDLE:
            self.get_logger().info("=== Starting pick-and-place ===")
            self._state = State.MOVE_HOME

        elif state == State.MOVE_HOME:
            home = self.get_parameter("poses.home").value
            self.get_logger().info("[MOVE_HOME]")
            self._send_arm_joints(home, State.PERCEIVE)

        elif state == State.PERCEIVE:
            if self._latest_detection is None:
                self.get_logger().warn("Waiting for detection...", throttle_duration_sec=3.0)
                return
            self._latched_object_pose = list(self._latest_detection)
            ox, oy, oz = self._latched_object_pose
            self.get_logger().info(f"[PERCEIVE] Object at: ({ox:.3f}, {oy:.3f}, {oz:.3f})")
            dist = math.sqrt(ox * ox + oy * oy)
            if dist > 0.28 or dist < 0.05:
                self.get_logger().error(f"Object at {dist:.3f}m — unreachable")
                self._state = State.ERROR
                return
            self._state = State.OPEN_GRIPPER

        elif state == State.OPEN_GRIPPER:
            val = self.get_parameter("grasp.gripper_open").value
            self.get_logger().info(f"[OPEN_GRIPPER] {val:.2f}")
            self._send_gripper(val, State.APPROACH)

        elif state == State.APPROACH:
            ox, oy, _ = self._latched_object_pose
            h = self.get_parameter("grasp.approach_height").value
            self.get_logger().info(f"[APPROACH] ({ox:.3f}, {oy:.3f}, {h:.3f})")
            self._send_cartesian_goal(ox, oy, h, State.RECORD_APPROACH_JOINTS)

        elif state == State.RECORD_APPROACH_JOINTS:
            # Save joint positions after approach — these have a good orientation.
            # We'll adjust these for descent instead of replanning.
            joints = self._get_current_arm_list()
            if joints is None:
                self.get_logger().warn("Waiting for joint states...")
                return
            self._approach_joints = list(joints)
            self.get_logger().info(
                f"[RECORD] Approach joints: {[f'{j:.3f}' for j in joints]}"
            )
            self._state = State.DESCEND

        elif state == State.DESCEND:
            # KEY FIX: Don't call MoveGroup again — that changes orientation.
            # Instead, take the approach joints and adjust shoulder_lift
            # to lower the arm. This preserves the approach orientation
            # because we only change ONE joint by a small amount.
            #
            # shoulder_lift more negative = arm tilts further down.
            # We adjust by a small delta computed from the height difference.
            approach_h = self.get_parameter("grasp.approach_height").value
            grasp_h = self.get_parameter("grasp.grasp_height").value
            delta_h = approach_h - grasp_h  # Positive: need to go lower

            # Approximate: 1 radian of shoulder_lift ≈ arm_length change in Z
            # Upper arm is ~0.113m, so delta_joint ≈ delta_h / 0.113
            # But this is approximate — for a small delta it works fine
            delta_joint = delta_h / 0.12  # ~0.58 rad for 7cm drop

            descend_joints = list(self._approach_joints)
            descend_joints[1] -= delta_joint  # shoulder_lift index=1, more negative = lower

            self.get_logger().info(
                f"[DESCEND] Adjusting shoulder_lift by {-delta_joint:.3f} rad"
            )
            self._send_arm_joints(descend_joints, State.CLOSE_GRIPPER)

        elif state == State.CLOSE_GRIPPER:
            val = self.get_parameter("grasp.gripper_closed").value
            self.get_logger().info(f"[CLOSE_GRIPPER] {val:.2f}")
            self._send_gripper(val, State.LIFT)

        elif state == State.LIFT:
            # Lift: reverse the descent adjustment
            lift_joints = list(self._approach_joints)
            self.get_logger().info("[LIFT] Returning to approach joints")
            self._send_arm_joints(lift_joints, State.MOVE_TO_PLACE)

        elif state == State.MOVE_TO_PLACE:
            px = self.get_parameter("place.x").value
            py = self.get_parameter("place.y").value
            lift_h = self.get_parameter("grasp.lift_height").value
            self.get_logger().info(f"[MOVE_TO_PLACE] ({px:.3f}, {py:.3f}, {lift_h:.3f})")
            self._send_cartesian_goal(px, py, lift_h, State.LOWER_TO_PLACE)

        elif state == State.LOWER_TO_PLACE:
            # Same trick: read current joints, adjust shoulder_lift to lower
            joints = self._get_current_arm_list()
            if joints is None:
                self.get_logger().warn("Waiting for joint states...")
                return
            lift_h = self.get_parameter("grasp.lift_height").value
            place_z = self.get_parameter("place.z").value
            delta_h = lift_h - place_z
            delta_joint = delta_h / 0.12

            lower_joints = list(joints)
            lower_joints[1] -= delta_joint

            self.get_logger().info(f"[LOWER_TO_PLACE] Adjusting shoulder_lift by {-delta_joint:.3f}")
            self._send_arm_joints(lower_joints, State.RELEASE)

        elif state == State.RELEASE:
            val = self.get_parameter("grasp.gripper_open").value
            self.get_logger().info("[RELEASE]")
            self._send_gripper(val, State.RETRACT)

        elif state == State.RETRACT:
            home = self.get_parameter("poses.home").value
            self.get_logger().info("[RETRACT]")
            self._send_arm_joints(home, State.DONE)

        elif state == State.DONE:
            self.get_logger().info("=== Pick-and-place COMPLETE ===")
            self._timer.cancel()

        elif state == State.ERROR:
            self.get_logger().error("=== Pick-and-place FAILED ===")
            self._timer.cancel()

    # ── Motion commands ───────────────────────────────────────────────

    def _send_arm_joints(self, positions: list, next_state: State) -> None:
        traj_time = self.get_parameter("trajectory_time_sec").value
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self._arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(traj_time)
        point.time_from_start.nanosec = int((traj_time % 1) * 1e9)
        goal.trajectory.points = [point]

        if not self._arm_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Arm action server not available")
            self._state = State.ERROR
            return

        self._action_in_progress = True
        future = self._arm_action.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_accepted(f, "arm", next_state)
        )

    def _send_gripper(self, position: float, next_state: State) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["gripper"]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = 1
        goal.trajectory.points = [point]

        if not self._gripper_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server not available")
            self._state = State.ERROR
            return

        self._action_in_progress = True
        future = self._gripper_action.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_accepted(f, "gripper", next_state)
        )

    def _send_cartesian_goal(
        self, x: float, y: float, z: float, next_state: State,
    ) -> None:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "arm"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0

        ws = WorkspaceParameters()
        ws.header.frame_id = "base_link"
        ws.min_corner.x = -0.5
        ws.min_corner.y = -0.5
        ws.min_corner.z = -0.1
        ws.max_corner.x = 0.5
        ws.max_corner.y = 0.5
        ws.max_corner.z = 0.5
        req.workspace_parameters = ws

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "gripper_frame_link"
        pos_constraint.weight = 1.0

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.w = 1.0
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.02]
        pos_constraint.constraint_region.primitives.append(sphere)

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        if not self._move_group_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available")
            self._state = State.ERROR
            return

        self._action_in_progress = True
        future = self._move_group_action.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_accepted(f, "move_group", next_state)
        )

    # ── Shared action callbacks ───────────────────────────────────────

    def _on_goal_accepted(self, future, name: str, next_state: State) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{name} goal REJECTED")
            self._action_in_progress = False
            self._state = State.ERROR
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_action_done(f, name, next_state)
        )

    def _on_action_done(self, future, name: str, next_state: State) -> None:
        try:
            result = future.result()
            inner = getattr(result, "result", None)
            error_code = getattr(inner, "error_code", None)
            if error_code is not None:
                code = getattr(error_code, "val", error_code)
                if code != 1:
                    self.get_logger().warn(f"{name} error code: {code}")
        except Exception as exc:
            self.get_logger().warn(f"{name} result check: {exc}")

        settle = self.get_parameter("settle_time_sec").value
        time.sleep(settle)
        self._state = next_state
        self._action_in_progress = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

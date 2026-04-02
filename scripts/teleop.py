"""Keyboard teleoperation for SO-101.

Controls:
  1/2  — shoulder_pan   (+/-)
  3/4  — shoulder_lift   (+/-)
  5/6  — elbow_flex      (+/-)
  7/8  — wrist_flex      (+/-)
  9/0  — wrist_roll      (+/-)
  o/c  — gripper open/close
  +/-  — increase/decrease step size
  p    — PRINT current joint angles (copy these as waypoints!)
  s    — SAVE waypoint to list
  d    — DUMP all saved waypoints
  q    — quit
"""
import sys
import tty
import termios
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def get_key():
    """Read a single keypress."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

rclpy.init()
node = Node("teleop")

arm_pub = node.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
gripper_pub = node.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 10)

# Current joints from /joint_states
current_joints = {}
def js_cb(msg):
    for name, pos in zip(msg.name, msg.position):
        current_joints[name] = pos

node.create_subscription(JointState, "/joint_states", js_cb, 10)

# Wait for first joint_states
print("Waiting for joint states...")
for _ in range(50):
    rclpy.spin_once(node, timeout_sec=0.1)
    if len(current_joints) >= 6:
        break

def get_arm_positions():
    return [current_joints.get(j, 0.0) for j in ARM_JOINTS]

def send_arm(positions):
    msg = JointTrajectory()
    msg.joint_names = ARM_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in positions]
    pt.time_from_start.sec = 0
    pt.time_from_start.nanosec = 500000000  # 0.5 sec
    msg.points = [pt]
    arm_pub.publish(msg)

def send_gripper(pos):
    msg = JointTrajectory()
    msg.joint_names = ["gripper"]
    pt = JointTrajectoryPoint()
    pt.positions = [float(pos)]
    pt.time_from_start.sec = 0
    pt.time_from_start.nanosec = 500000000
    msg.points = [pt]
    gripper_pub.publish(msg)

step = 0.05  # radians per keypress
gripper_pos = 0.0
saved_waypoints = []

KEY_MAP = {
    '1': (0, +1),   '2': (0, -1),   # shoulder_pan
    '3': (1, +1),   '4': (1, -1),   # shoulder_lift
    '5': (2, +1),   '6': (2, -1),   # elbow_flex
    '7': (3, +1),   '8': (3, -1),   # wrist_flex
    '9': (4, +1),   '0': (4, -1),   # wrist_roll
}

print(f"""
╔══════════════════════════════════════════════════╗
║  SO-101 KEYBOARD TELEOP                         ║
║                                                  ║
║  1/2  shoulder_pan    (+/-)                      ║
║  3/4  shoulder_lift   (+/-)                      ║
║  5/6  elbow_flex      (+/-)                      ║
║  7/8  wrist_flex      (+/-)                      ║
║  9/0  wrist_roll      (+/-)                      ║
║  o/c  gripper open/close                         ║
║  +/-  step size (current: {step:.3f} rad)          ║
║  p    print current joints                       ║
║  s    save waypoint                              ║
║  d    dump all saved waypoints                   ║
║  q    quit                                       ║
╚══════════════════════════════════════════════════╝
""")

try:
    while True:
        rclpy.spin_once(node, timeout_sec=0.05)
        key = get_key()

        if key == 'q':
            print("\n\n=== SAVED WAYPOINTS ===")
            for i, (name, joints, grip) in enumerate(saved_waypoints):
                print(f'  {name}: {[round(j, 3) for j in joints]}, gripper={grip:.2f}')
            break

        elif key in KEY_MAP:
            joint_idx, direction = KEY_MAP[key]
            positions = get_arm_positions()
            positions[joint_idx] += direction * step
            send_arm(positions)
            time.sleep(0.3)
            rclpy.spin_once(node, timeout_sec=0.1)
            pos = get_arm_positions()
            print(f"  {ARM_JOINTS[joint_idx]}: {pos[joint_idx]:.3f}    "
                  f"all: {[f'{p:.2f}' for p in pos]}", end='\r')

        elif key == 'o':
            gripper_pos = 1.0
            send_gripper(gripper_pos)
            print(f"  gripper: OPEN ({gripper_pos:.1f})")

        elif key == 'c':
            gripper_pos = -0.1
            send_gripper(gripper_pos)
            print(f"  gripper: CLOSED ({gripper_pos:.1f})")

        elif key == '+' or key == '=':
            step = min(step + 0.01, 0.2)
            print(f"  step size: {step:.3f} rad")

        elif key == '-':
            step = max(step - 0.01, 0.01)
            print(f"  step size: {step:.3f} rad")

        elif key == 'p':
            rclpy.spin_once(node, timeout_sec=0.1)
            pos = get_arm_positions()
            grip = current_joints.get("gripper", 0.0)
            print(f"\n  JOINTS: {[round(p, 3) for p in pos]}")
            print(f"  GRIPPER: {grip:.3f}")

        elif key == 's':
            rclpy.spin_once(node, timeout_sec=0.1)
            pos = get_arm_positions()
            grip = current_joints.get("gripper", 0.0)
            name = input(f"\n  Waypoint name (e.g. approach, grasp, lift): ")
            saved_waypoints.append((name, list(pos), grip))
            print(f"  Saved '{name}': {[round(p, 3) for p in pos]}")

        elif key == 'd':
            print("\n\n=== SAVED WAYPOINTS ===")
            for i, (name, joints, grip) in enumerate(saved_waypoints):
                print(f'  {name}: {[round(j, 3) for j in joints]}, gripper={grip:.2f}')
            print()

except KeyboardInterrupt:
    pass

print("\nDone.")
node.destroy_node()
rclpy.shutdown()

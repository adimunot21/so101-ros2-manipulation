"""Find waypoints for cube at x=0.28."""
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time

rclpy.init()
node = Node("waypoint_finder")

arm_pub = node.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
gripper_pub = node.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 10)
fk_client = node.create_client(GetPositionFK, "/compute_fk")
fk_client.wait_for_service(timeout_sec=5.0)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def send_arm(positions, wait=3.0):
    msg = JointTrajectory()
    msg.joint_names = ARM_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in positions]
    pt.time_from_start.sec = 2
    msg.points = [pt]
    arm_pub.publish(msg)
    time.sleep(wait)

def send_gripper(pos, wait=1.5):
    msg = JointTrajectory()
    msg.joint_names = ["gripper"]
    pt = JointTrajectoryPoint()
    pt.positions = [float(pos)]
    pt.time_from_start.sec = 1
    msg.points = [pt]
    gripper_pub.publish(msg)
    time.sleep(wait)

def get_fk(positions):
    req = GetPositionFK.Request()
    req.header.frame_id = "base_link"
    req.fk_link_names = ["gripper_frame_link", "gripper_link"]
    req.robot_state.joint_state.name = ARM_JOINTS + ["gripper"]
    req.robot_state.joint_state.position = list(positions) + [0.0]
    future = fk_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    if future.result() and len(future.result().pose_stamped) >= 2:
        tip = future.result().pose_stamped[0].pose.position
        grip = future.result().pose_stamped[1].pose.position
        tip_pos = (tip.x, tip.y, tip.z)
        dx = tip.x - grip.x
        dy = tip.y - grip.y
        dz = tip.z - grip.z
        length = max((dx*dx + dy*dy + dz*dz)**0.5, 0.001)
        fdir = (dx/length, dy/length, dz/length)
        return tip_pos, fdir
    return None, None

# Target: cube at x=0.28
CUBE_X = 0.28
print(f"Searching for waypoints — cube at x={CUBE_X}")
print("="*60)

approach_list = []
grasp_list = []

for sl in np.arange(-1.5, 1.5, 0.1):
    for ef in np.arange(-1.5, 1.5, 0.1):
        for wf in np.arange(-0.5, 1.6, 0.1):
            joints = [0.0, sl, ef, wf, 0.0]
            tip, fdir = get_fk(joints)
            if tip is None:
                continue
            
            if abs(tip[0] - CUBE_X) > 0.03:
                continue
            if abs(tip[1]) > 0.03:
                continue
            if fdir[2] > -0.70:
                continue

            if 0.06 <= tip[2] <= 0.14:
                approach_list.append((-fdir[2], joints, tip, fdir))
            if 0.005 <= tip[2] <= 0.035:
                grasp_list.append((-fdir[2], joints, tip, fdir))

approach_list.sort(reverse=True)
grasp_list.sort(reverse=True)

print(f"Approach candidates: {len(approach_list)}")
for i, (s, j, t, f) in enumerate(approach_list[:5]):
    print(f"  #{i+1} {[f'{v:.1f}' for v in j]}  tip=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  fz={f[2]:.2f}")

print(f"\nGrasp candidates: {len(grasp_list)}")
for i, (s, j, t, f) in enumerate(grasp_list[:5]):
    print(f"  #{i+1} {[f'{v:.1f}' for v in j]}  tip=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  fz={f[2]:.2f}")

if not approach_list or not grasp_list:
    print("\nNo candidates! Dumping lowest z values at x~0.28:")
    results = []
    for sl in np.arange(-1.5, 1.5, 0.2):
        for ef in np.arange(-1.5, 1.5, 0.2):
            for wf in np.arange(-0.5, 1.6, 0.2):
                joints = [0.0, sl, ef, wf, 0.0]
                tip, fdir = get_fk(joints)
                if tip and abs(tip[0] - CUBE_X) < 0.05 and abs(tip[1]) < 0.05:
                    results.append((tip[2], joints, tip, fdir))
    results.sort()
    for z, j, t, f in results[:10]:
        print(f"  {[f'{v:.1f}' for v in j]}  tip=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  fz={f[2]:.2f}")
    node.destroy_node()
    rclpy.shutdown()
    exit()

best_approach = approach_list[0]
best_grasp = grasp_list[0]

print(f"\n{'='*60}")
print(f"BEST APPROACH: {[f'{j:.2f}' for j in best_approach[1]]}")
print(f"  tip=({best_approach[2][0]:.3f}, {best_approach[2][1]:.3f}, {best_approach[2][2]:.3f})  fz={best_approach[3][2]:.2f}")
print(f"BEST GRASP:    {[f'{j:.2f}' for j in best_grasp[1]]}")
print(f"  tip=({best_grasp[2][0]:.3f}, {best_grasp[2][1]:.3f}, {best_grasp[2][2]:.3f})  fz={best_grasp[3][2]:.2f}")
print(f"{'='*60}")

input("\nPress Enter to demo the full pick sequence...")

# Restart sim first for this to work (cube at new position)
print("\n[1/8] HOME")
send_arm([0.0, 0.0, 0.0, 0.0, 0.0])

print("[2/8] OPEN GRIPPER")
send_gripper(1.0)

print("[3/8] APPROACH")
send_arm(best_approach[1])
input("  Gripper above cube pointing down? Enter...")

print("[4/8] DESCEND")
send_arm(best_grasp[1], wait=4.0)
input("  Gripper around cube? Enter...")

print("[5/8] CLOSE GRIPPER")
send_gripper(-0.1, wait=2.0)
input("  Gripping cube? Enter...")

print("[6/8] LIFT")
send_arm(best_approach[1], wait=4.0)
input("  Cube lifted? Enter...")

print("[7/8] RELEASE")
send_gripper(1.0)

print("[8/8] HOME")
send_arm([0.0, 0.0, 0.0, 0.0, 0.0])

print("\n=== DONE ===")
node.destroy_node()
rclpy.shutdown()

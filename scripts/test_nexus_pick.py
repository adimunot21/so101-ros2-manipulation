"""Test pick sequence in SO101-Nexus environment."""
import gymnasium as gym
import so101_nexus_mujoco
import numpy as np
import cv2
import mujoco

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
obs, info = env.reset()

model = env.unwrapped.model
data = env.unwrapped.data

# Find cube body
cube_id = None
for i in range(model.nbody):
    name = model.body(i).name
    if 'pick' in name or 'cube' in name or 'slot' in name:
        print(f"  Found: body '{name}' at {data.xpos[i]}")
        cube_id = i

if cube_id is None:
    print("No cube found! Bodies:")
    for i in range(model.nbody):
        print(f"  [{i}] {model.body(i).name} at {data.xpos[i]}")
    env.close()
    exit()

cube_pos = data.xpos[cube_id].copy()
print(f"\nCube position: {cube_pos}")

pan_angle = np.arctan2(cube_pos[1], cube_pos[0])
print(f"Pan angle: {pan_angle:.3f} rad")

OPEN = 1.0
CLOSED = -0.1

home     = [0.0, 0.0, 0.0, 0.0, 0.0, OPEN]
approach = [pan_angle, 0.3, -0.3, 1.5, 0.0, OPEN]
grasp    = [pan_angle, 0.8, -0.7, 1.4, 0.0, OPEN]
close    = [pan_angle, 0.8, -0.7, 1.4, 0.0, CLOSED]
lift     = [pan_angle, 0.3, -0.3, 1.5, 0.0, CLOSED]

waypoints = [
    ("home", home, 30),
    ("approach", approach, 40),
    ("grasp_open", grasp, 40),
    ("close_gripper", close, 30),
    ("lift", lift, 40),
]

def save_frame(env, name):
    frame = env.render()
    if frame is not None:
        cv2.imwrite(f"assets/nexus_{name}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

print("\nRunning pick sequence:")
for wp_name, target, steps in waypoints:
    print(f"\n  [{wp_name}]")
    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(np.array(target, dtype=np.float32))
    
    joint_pos = data.qpos[:6]
    cube_z = data.xpos[cube_id][2]
    print(f"    joints: {[f'{j:.2f}' for j in joint_pos]}")
    print(f"    cube_z: {cube_z:.4f} ({'LIFTED!' if cube_z > 0.05 else 'on table'})")
    print(f"    reward: {reward:.4f}")
    save_frame(env, wp_name)

print(f"\nFinal cube: {data.xpos[cube_id]}")
env.close()
print("Done — check assets/nexus_*.png")

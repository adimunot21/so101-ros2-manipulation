"""Test pick with proven waypoints + slow interpolation."""
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('ros2_ws/src/so101_description/mjcf/scene.xml')
data = mujoco.MjData(model)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
viewer = mujoco.viewer.launch_passive(model, data)

def interpolate_and_run(start, end, steps=100):
    """Smooth interpolation between waypoints."""
    for t in np.linspace(0, 1, steps):
        target = start + (end - start) * t
        for i in range(6):
            data.ctrl[i] = target[i]
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.004)

def hold(steps=50):
    """Hold current position."""
    for _ in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.004)

# Waypoints from teleop
HOME     = np.array([0.0, 0.0,  0.0,  0.0, 0.0,  1.0])
APPROACH = np.array([0.0, 0.15, -0.1, 1.5, 0.05, 1.0])
GRASP    = np.array([0.0, 0.45, -0.2, 1.5, 0.05, 1.0])
CLOSE    = np.array([0.0, 0.45, -0.2, 1.5, 0.05, -0.1])
LIFT     = np.array([0.0, 0.2,  -0.4, 1.5, 0.05, -0.1])

print(f"Cube start: {data.xpos[cube_id]}")

print("[1] HOME → APPROACH")
interpolate_and_run(HOME, APPROACH, 150)
hold(50)

print("[2] APPROACH → GRASP (descend)")
interpolate_and_run(APPROACH, GRASP, 150)
hold(50)

print("[3] CLOSE GRIPPER (slow)")
interpolate_and_run(GRASP, CLOSE, 100)
hold(100)  # Extra hold to let grip settle
cube_z = data.xpos[cube_id][2]
print(f"    cube_z after close: {cube_z:.4f}")

print("[4] LIFT (slow)")
interpolate_and_run(CLOSE, LIFT, 200)  # Slow lift
hold(100)
cube_z = data.xpos[cube_id][2]
print(f"    cube_z after lift: {cube_z:.4f} ({'SUCCESS!' if cube_z > 0.05 else 'dropped'})")

input("Press Enter to close...")
viewer.close()

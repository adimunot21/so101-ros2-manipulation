"""Teleop directly in MuJoCo — no ROS2 needed.

Controls:
  1/2  shoulder_pan    3/4  shoulder_lift
  5/6  elbow_flex      7/8  wrist_flex
  9/0  wrist_roll      o/c  gripper open/close
  +/-  step size       p    print joints
  s    save waypoint   d    dump waypoints   q quit
"""
import sys, tty, termios
import mujoco
import mujoco.viewer
import numpy as np
import time

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

model = mujoco.MjModel.from_xml_path('ros2_ws/src/so101_description/mjcf/scene.xml')
data = mujoco.MjData(model)

cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
viewer = mujoco.viewer.launch_passive(model, data)

# Start from the approximate approach
ctrl = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
for i in range(6):
    data.ctrl[i] = ctrl[i]

# Run a few steps to settle
for _ in range(200):
    mujoco.mj_step(model, data)
viewer.sync()

step_size = 0.05
saved = []

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEY_MAP = {
    '1': (0, +1), '2': (0, -1),
    '3': (1, +1), '4': (1, -1),
    '5': (2, +1), '6': (2, -1),
    '7': (3, +1), '8': (3, -1),
    '9': (4, +1), '0': (4, -1),
}

print(f"""
╔════════════════════════════════════════════════╗
║  DIRECT MUJOCO TELEOP (our scene)             ║
║  1/2 pan  3/4 lift  5/6 elbow  7/8 wrist_flex ║
║  9/0 wrist_roll    o/c gripper   +/- step     ║
║  p print  s save  d dump  q quit              ║
╚════════════════════════════════════════════════╝
Cube at: {data.xpos[cube_id]}
Step: {step_size:.3f} rad
""")

try:
    while viewer.is_running():
        key = get_key()

        if key == 'q':
            print("\n\n=== SAVED WAYPOINTS ===")
            for name, joints in saved:
                print(f"  {name}: {joints}")
            break

        elif key in KEY_MAP:
            idx, direction = KEY_MAP[key]
            ctrl[idx] += direction * step_size
            for i in range(6):
                data.ctrl[i] = ctrl[i]
            for _ in range(50):
                mujoco.mj_step(model, data)
            viewer.sync()
            cube_z = data.xpos[cube_id][2]
            sys.stdout.write(f"\r  {NAMES[idx]:15s}: {ctrl[idx]:+.3f}  "
                           f"cube_z:{cube_z:.4f}  ")
            sys.stdout.flush()

        elif key == 'o':
            ctrl[5] = 1.0
            for i in range(6):
                data.ctrl[i] = ctrl[i]
            for _ in range(100):
                mujoco.mj_step(model, data)
            viewer.sync()
            print(f"\n  gripper: OPEN")

        elif key == 'c':
            ctrl[5] = -0.1
            for i in range(6):
                data.ctrl[i] = ctrl[i]
            for _ in range(150):
                mujoco.mj_step(model, data)
            viewer.sync()
            cube_z = data.xpos[cube_id][2]
            print(f"\n  gripper: CLOSED  cube_z: {cube_z:.4f}")

        elif key == '+' or key == '=':
            step_size = min(step_size + 0.01, 0.2)
            print(f"\n  step: {step_size:.3f}")

        elif key == '-':
            step_size = max(step_size - 0.01, 0.01)
            print(f"\n  step: {step_size:.3f}")

        elif key == 'p':
            cube_pos = data.xpos[cube_id]
            print(f"\n  CTRL:  {[round(c, 3) for c in ctrl]}")
            print(f"  CUBE:  {[round(float(c), 4) for c in cube_pos]}")

        elif key == 's':
            sys.stdout.write("\n  Name: ")
            sys.stdout.flush()
            name_chars = []
            while True:
                ch = get_key()
                if ch == '\r' or ch == '\n':
                    break
                name_chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
            name = ''.join(name_chars)
            saved.append((name, [round(c, 3) for c in ctrl]))
            print(f"\n  Saved '{name}': {saved[-1][1]}")

        elif key == 'd':
            print("\n\n=== WAYPOINTS ===")
            for name, joints in saved:
                print(f"  {name}: {joints}")
            print()

except KeyboardInterrupt:
    pass

viewer.close()
print("\nDone.")

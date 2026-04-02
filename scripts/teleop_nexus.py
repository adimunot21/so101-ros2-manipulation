"""Keyboard teleop in SO101-Nexus with MuJoCo native viewer.

Controls:
  1/2  shoulder_pan    3/4  shoulder_lift
  5/6  elbow_flex      7/8  wrist_flex
  9/0  wrist_roll      o/c  gripper open/close
  +/-  step size       p    print joints
  s    save waypoint   d    dump waypoints
  r    reset episode   q    quit
"""
import sys, tty, termios
import gymnasium as gym
import so101_nexus_mujoco
import numpy as np
import mujoco
import mujoco.viewer

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

env = gym.make("MuJoCoPickLift-v1", render_mode=None)
obs, info = env.reset()
model = env.unwrapped.model
data = env.unwrapped.data

# Find cube
cube_id = None
for i in range(model.nbody):
    if 'pick' in model.body(i).name or 'slot' in model.body(i).name:
        cube_id = i
        break

# Launch viewer
viewer = mujoco.viewer.launch_passive(model, data)

target = data.qpos[:6].copy().astype(float)
step_size = 0.05
saved = []

KEY_MAP = {
    '1': (0, +1), '2': (0, -1),
    '3': (1, +1), '4': (1, -1),
    '5': (2, +1), '6': (2, -1),
    '7': (3, +1), '8': (3, -1),
    '9': (4, +1), '0': (4, -1),
}
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

print(f"""
╔════════════════════════════════════════════════╗
║  SO101-Nexus TELEOP (MuJoCo viewer open)      ║
║  1/2 pan  3/4 lift  5/6 elbow  7/8 wrist_flex ║
║  9/0 wrist_roll    o/c gripper   +/- step     ║
║  p print  s save  d dump  r reset  q quit     ║
╚════════════════════════════════════════════════╝
""")
if cube_id:
    print(f"Cube at: {data.xpos[cube_id]}")
print(f"Step size: {step_size:.3f} rad\n")

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
            target[idx] += direction * step_size
            for _ in range(5):
                env.step(target.astype(np.float32))
            viewer.sync()
            joints = data.qpos[:6]
            cube_z = data.xpos[cube_id][2] if cube_id else 0
            sys.stdout.write(f"\r  {NAMES[idx]:15s}: {joints[idx]:+.3f}  "
                           f"cube_z:{cube_z:.4f}  ")
            sys.stdout.flush()

        elif key == 'o':
            target[5] = 1.0
            for _ in range(10):
                env.step(target.astype(np.float32))
            viewer.sync()
            print(f"\n  gripper: OPEN")

        elif key == 'c':
            target[5] = -0.1
            for _ in range(20):
                env.step(target.astype(np.float32))
            viewer.sync()
            cube_z = data.xpos[cube_id][2] if cube_id else 0
            print(f"\n  gripper: CLOSED  cube_z: {cube_z:.4f}")

        elif key == '+' or key == '=':
            step_size = min(step_size + 0.01, 0.2)
            print(f"\n  step: {step_size:.3f}")

        elif key == '-':
            step_size = max(step_size - 0.01, 0.01)
            print(f"\n  step: {step_size:.3f}")

        elif key == 'p':
            joints = data.qpos[:6]
            cube_pos = data.xpos[cube_id] if cube_id else [0,0,0]
            print(f"\n  JOINTS: {[round(float(j), 3) for j in joints]}")
            print(f"  CUBE:   {[round(float(c), 4) for c in cube_pos]}")

        elif key == 's':
            joints = list(data.qpos[:6])
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
            saved.append((name, [round(float(j), 3) for j in joints]))
            print(f"\n  Saved '{name}': {saved[-1][1]}")

        elif key == 'd':
            print("\n\n=== WAYPOINTS ===")
            for name, joints in saved:
                print(f"  {name}: {joints}")
            print()

        elif key == 'r':
            obs, info = env.reset()
            target = data.qpos[:6].copy().astype(float)
            viewer.sync()
            if cube_id:
                print(f"\n  RESET — cube at: {data.xpos[cube_id]}")

except KeyboardInterrupt:
    pass

viewer.close()
env.close()
print("\nDone.")

"""Collect pick-and-lift demos. Saves each episode to disk immediately."""
import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HOME     = np.array([0.0, 0.0,   0.0,  0.0, 0.0,  1.0])
APPROACH = np.array([0.0, 0.15, -0.1,  1.5, 0.05, 1.0])
GRASP    = np.array([0.0, 0.45, -0.2,  1.5, 0.05, 1.0])
CLOSE    = np.array([0.0, 0.45, -0.2,  1.5, 0.05, -0.1])
LIFT     = np.array([0.0, 0.2,  -0.4,  1.5, 0.05, -0.1])

SEQUENCE = [
    (HOME,     APPROACH, 150),
    (APPROACH, GRASP,    150),
    (GRASP,    CLOSE,    100),
    (CLOSE,    LIFT,     200),
]

SCENE_PATH = "ros2_ws/src/so101_description/mjcf/scene.xml"


def run_and_save_episode(ep_idx, output_dir, noise_scale=0.005):
    """Run one episode and save directly to disk. Returns success bool."""
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    overhead_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    mujoco.mj_forward(model, data)
    cube_start = data.xpos[cube_id].copy()

    renderer = mujoco.Renderer(model, height=480, width=640)

    ep_dir = Path(output_dir) / f"episode_{ep_idx:04d}"
    oh_dir = ep_dir / "overhead"
    wr_dir = ep_dir / "wrist"
    oh_dir.mkdir(parents=True, exist_ok=True)
    wr_dir.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    frame_idx = 0

    for start_wp, end_wp, steps in SEQUENCE:
        for t in np.linspace(0, 1, steps):
            target = start_wp + (end_wp - start_wp) * t
            if noise_scale > 0:
                noise = np.random.randn(5) * noise_scale
                target[:5] = target[:5] + noise

            # Record state
            states.append(data.qpos[:6].copy())
            actions.append(target.copy())

            # Save overhead image
            renderer.update_scene(data, camera=overhead_cam)
            img = renderer.render()
            cv2.imwrite(str(oh_dir / f"{frame_idx:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Save wrist image
            renderer.update_scene(data, camera=wrist_cam)
            img = renderer.render()
            cv2.imwrite(str(wr_dir / f"{frame_idx:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Step sim
            for i in range(6):
                data.ctrl[i] = target[i]
            mujoco.mj_step(model, data)

            frame_idx += 1

    # Hold to settle
    for _ in range(50):
        mujoco.mj_step(model, data)

    final_z = float(data.xpos[cube_id][2])
    success = final_z > 0.05

    # Save states and actions
    np.save(ep_dir / "states.npy", np.array(states, dtype=np.float32))
    np.save(ep_dir / "actions.npy", np.array(actions, dtype=np.float32))

    renderer.close()

    return {
        "success": bool(success),
        "cube_final_z": final_z,
        "cube_start": cube_start.tolist(),
        "length": frame_idx,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data/pick_lift_demos")
    parser.add_argument("--noise", type=float, default=0.005)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.num_episodes} episodes")
    print(f"Noise: {args.noise}")
    print(f"Output: {args.output_dir}\n")

    results = []
    for i in range(args.num_episodes):
        info = run_and_save_episode(i, args.output_dir, args.noise)
        results.append(info)
        status = "OK" if info["success"] else "FAIL"
        print(f"  Episode {i:3d}: {status}  cube_z={info['cube_final_z']:.3f}")

    # Save metadata
    num_ok = sum(1 for r in results if r["success"])
    metadata = {
        "num_episodes": len(results),
        "num_successful": num_ok,
        "success_rate": num_ok / max(len(results), 1),
        "state_dim": 6,
        "action_dim": 6,
        "image_shape": [480, 640, 3],
        "fps": 50,
        "joint_names": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
        "episodes": results,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done: {num_ok}/{len(results)} successful ({num_ok/len(results)*100:.0f}%)")
    print(f"Saved to: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

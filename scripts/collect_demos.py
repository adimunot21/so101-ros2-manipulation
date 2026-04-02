"""Collect pick-and-lift demonstrations using SO101-Nexus.

Uses fixed waypoints from manual teleop with very tight cube spawn
to ensure reliable grasping. Noise is added to actions for diversity.

Usage:
    python3 scripts/collect_demos.py --num-episodes 1 --visualize
    python3 scripts/collect_demos.py --num-episodes 50
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import so101_nexus_mujoco
from so101_nexus_core.config import PickConfig


# ── Exact waypoints from manual teleop (proven to work) ──────
HOME     = np.array([0.0,   0.0,    0.0,    0.0,  0.0,   1.0])
APPROACH = np.array([0.149, 0.394, -0.299,  1.61, 0.003, 1.0])
GRASP    = np.array([0.149, 0.69,  -0.499,  1.61, 0.003, 1.0])
CLOSE    = np.array([0.149, 0.69,  -0.499,  1.61, 0.003, -0.1])
LIFT     = np.array([0.149, -0.002, -0.498, 1.61, 0.003, -0.1])
RELEASE  = np.array([0.149, -0.002, -0.498, 1.61, 0.003, 1.0])

SEQUENCE = [
    ("home",     HOME,     APPROACH, 30),
    ("approach", APPROACH, GRASP,    25),
    ("grasp",    GRASP,    CLOSE,    15),
    ("lift",     CLOSE,    LIFT,     30),
]


def make_env():
    """Create env with cube spawning at the position our waypoints target.
    
    Teleop was done with cube at ~(0.276, 0.004). Spawn center is (0.15, 0).
    Radius from center: sqrt((0.276-0.15)^2 + 0.004^2) ≈ 0.126m.
    Angle: ~1.8°. We use tiny ranges around these values.
    """
    config = PickConfig(
        spawn_min_radius=0.124,
        spawn_max_radius=0.128,
        spawn_angle_half_range_deg=3.0,
        robot_init_qpos_noise=0.0,
    )
    env = gym.make("MuJoCoPickLift-v1", render_mode=None, config=config)
    return env


def interpolate(start, end, steps):
    """Linear interpolation between two joint configs."""
    return [start + (end - start) * t for t in np.linspace(0, 1, steps)]


def add_noise(target, noise_scale=0.01):
    """Add small Gaussian noise to action for demo diversity."""
    noise = np.random.randn(len(target)) * noise_scale
    noise[5] = 0.0  # Don't add noise to gripper
    return target + noise


def run_episode(env, visualize=False, noise_scale=0.01):
    """Run one pick-lift episode, return recorded data."""
    obs, info = env.reset()
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Find cube
    cube_id = None
    for i in range(model.nbody):
        if 'pick' in model.body(i).name or 'slot' in model.body(i).name:
            cube_id = i
            break
    if cube_id is None:
        return None

    cube_pos = data.xpos[cube_id].copy()

    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    states = []
    actions = []
    images = []

    for seg_name, start_wp, end_wp, steps in SEQUENCE:
        targets = interpolate(start_wp, end_wp, steps)
        for target in targets:
            # Record state before action
            state = data.qpos[:6].copy()
            states.append(state)

            # Add noise for diversity
            noisy_target = add_noise(target, noise_scale)
            actions.append(noisy_target.copy())

            # Render image
            renderer.update_scene(data)
            img = renderer.render()
            images.append(img.copy())

            # Step
            env.step(noisy_target.astype(np.float32))

            if viewer:
                viewer.sync()
                time.sleep(0.02)

    renderer.close()

    final_cube_z = data.xpos[cube_id][2]
    success = final_cube_z > 0.04

    if viewer:
        try:
            viewer.close()
        except:
            pass

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "images": np.array(images),
        "success": success,
        "cube_final_z": float(final_cube_z),
        "cube_start_pos": cube_pos.tolist(),
    }


def save_dataset(episodes, output_dir):
    """Save collected episodes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [e for e in episodes if e["success"]]

    metadata = {
        "num_episodes": len(episodes),
        "num_successful": len(successful),
        "success_rate": len(successful) / max(len(episodes), 1),
        "state_dim": 6,
        "action_dim": 6,
        "image_shape": [480, 640, 3],
        "fps": 50,
        "joint_names": [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ],
    }

    for i, ep in enumerate(episodes):
        ep_dir = output_dir / f"episode_{i:04d}"
        ep_dir.mkdir(exist_ok=True)

        np.save(ep_dir / "states.npy", ep["states"])
        np.save(ep_dir / "actions.npy", ep["actions"])

        img_dir = ep_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for j, img in enumerate(ep["images"]):
            cv2.imwrite(
                str(img_dir / f"frame_{j:04d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

        metadata[f"episode_{i:04d}"] = {
            "length": len(ep["states"]),
            "success": ep["success"],
            "cube_final_z": ep["cube_final_z"],
            "cube_start_pos": ep["cube_start_pos"],
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Total episodes: {metadata['num_episodes']}")
    print(f"  Successful: {metadata['num_successful']}")
    print(f"  Success rate: {metadata['success_rate'] * 100:.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data/pick_lift_demos")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Action noise scale for diversity")
    args = parser.parse_args()

    env = make_env()
    episodes = []

    print(f"Collecting {args.num_episodes} episodes...")
    print(f"Noise scale: {args.noise}")
    print(f"Output: {args.output_dir}\n")

    for i in range(args.num_episodes):
        ep = run_episode(env, visualize=args.visualize, noise_scale=args.noise)
        if ep is None:
            print(f"  Episode {i:3d}: ERROR")
            continue

        episodes.append(ep)
        status = "OK" if ep["success"] else "FAIL"
        print(f"  Episode {i:3d}: {status}  cube_z={ep['cube_final_z']:.4f}  "
              f"start=({ep['cube_start_pos'][0]:.3f}, {ep['cube_start_pos'][1]:.3f})")

    env.close()
    save_dataset(episodes, args.output_dir)


if __name__ == "__main__":
    main()

"""Convert our raw demo data to LeRobot dataset format.

Reads from data/pick_lift_demos/ (numpy + PNG files)
Creates a LeRobot dataset and pushes to HuggingFace Hub.

Usage:
    python3 scripts/convert_to_lerobot.py --repo-id adimunot21/so101-pick-lift
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def convert(input_dir: str, repo_id: str, push: bool = True):
    input_dir = Path(input_dir)
    metadata = json.load(open(input_dir / "metadata.json"))

    num_episodes = metadata["num_episodes"]
    fps = metadata["fps"]
    print(f"Converting {num_episodes} episodes at {fps} fps")
    print(f"Repo ID: {repo_id}")

    # Define features — what each frame contains
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [JOINT_NAMES],
        },
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="so101",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    for ep_idx in range(num_episodes):
        ep_dir = input_dir / f"episode_{ep_idx:04d}"
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")

        oh_dir = ep_dir / "overhead"
        wr_dir = ep_dir / "wrist"
        num_frames = len(states)

        print(f"  Episode {ep_idx:3d}: {num_frames} frames", end="")

        for frame_idx in range(num_frames):
            # Load images (BGR from cv2 → RGB)
            oh_img = cv2.imread(str(oh_dir / f"{frame_idx:04d}.png"))
            oh_img = cv2.cvtColor(oh_img, cv2.COLOR_BGR2RGB)

            wr_img = cv2.imread(str(wr_dir / f"{frame_idx:04d}.png"))
            wr_img = cv2.cvtColor(wr_img, cv2.COLOR_BGR2RGB)

            frame = {
                "observation.state": torch.from_numpy(states[frame_idx]),
                "action": torch.from_numpy(actions[frame_idx]),
                "observation.images.overhead": torch.from_numpy(oh_img),
                "observation.images.wrist": torch.from_numpy(wr_img),
            }
            frame["task"] = "pick up the red cube"
            dataset.add_frame(frame)

        dataset.save_episode()
        print(" ✓")

    print("\nFinalizing dataset...")
    dataset.finalize()

    if push:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=False)
        print(f"Done! Dataset at: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"Dataset saved locally (not pushed)")
        print(f"To push later: dataset.push_to_hub()")

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pick_lift_demos")
    parser.add_argument("--repo-id", required=True,
                        help="HuggingFace repo ID, e.g. adimunot21/so101-pick-lift")
    parser.add_argument("--no-push", action="store_true",
                        help="Save locally only, don't push to Hub")
    args = parser.parse_args()

    convert(args.input_dir, args.repo_id, push=not args.no_push)


if __name__ == "__main__":
    main()

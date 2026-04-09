"""Evaluate ACT policy with proper preprocessing and postprocessing.

The v1 eval script was broken because it bypassed LeRobot's normalization
pipeline. The model expects MEAN_STD normalized inputs and outputs normalized
actions. LeRobot ships preprocessor/postprocessor pipelines with the model
that handle this correctly.

Usage (in lerobotROS conda env):
    cd ~/so101_ros2_manip
    python3 scripts/eval_policy.py
"""
import mujoco
import numpy as np
import torch
import cv2
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

# ── Configuration ─────────────────────────────────────────────────────
REPO_ID = "adimunot/act-so101-pick-lift"
SCENE_XML = "ros2_ws/src/so101_description/mjcf/scene.xml"
NUM_TRIALS = 3
NUM_STEPS = 600
JOINT_LIMITS_LOW = np.array([-1.92, -1.75, -1.69, -1.66, -2.74, -0.17])
JOINT_LIMITS_HIGH = np.array([1.92, 1.75, 1.69, 1.66, 2.84, 1.75])

# ── Load policy + processors ─────────────────────────────────────────
print(f"Loading policy from {REPO_ID}...")
policy = ACTPolicy.from_pretrained(REPO_ID)
policy.eval()
policy.to("cuda")

preprocessor = DataProcessorPipeline.from_pretrained(REPO_ID, "policy_preprocessor.json")
postprocessor = DataProcessorPipeline.from_pretrained(REPO_ID, "policy_postprocessor.json")
print("Policy, preprocessor, and postprocessor loaded.")

# ── Setup MuJoCo ──────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
overhead_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
wrist_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
renderer = mujoco.Renderer(model, height=480, width=640)

# ── Run evaluation trials ─────────────────────────────────────────────
results = []

for trial in range(NUM_TRIALS):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    policy.reset()

    frames = []
    print(f"\nTrial {trial}: cube at {data.xpos[cube_id]}")

    for step in range(NUM_STEPS):
        state = data.qpos[:6].copy().astype(np.float32)

        renderer.update_scene(data, camera=overhead_cam)
        oh_img = renderer.render().copy()

        renderer.update_scene(data, camera=wrist_cam)
        wr_img = renderer.render().copy()

        obs = {
            "observation.state": torch.from_numpy(state),
            "observation.images.overhead": torch.from_numpy(oh_img).float().permute(2, 0, 1) / 255.0,
            "observation.images.wrist": torch.from_numpy(wr_img).float().permute(2, 0, 1) / 255.0,
        }

        obs_processed = preprocessor.process_observation(obs)

        with torch.no_grad():
            raw_action = policy.select_action(obs_processed)

        action = postprocessor.process_action(raw_action)
        action = action.squeeze().cpu().numpy()

        action = np.clip(action, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        for i in range(6):
            data.ctrl[i] = float(action[i])
        mujoco.mj_step(model, data)

        if step % 50 == 0:
            cube_z = data.xpos[cube_id][2]
            ctrl_str = ", ".join(f"{data.ctrl[i]:+.3f}" for i in range(6))
            print(f"  step {step:3d}: ctrl=[{ctrl_str}]  cube_z={cube_z:.4f}")

        if step % 3 == 0:
            renderer.update_scene(data, camera=overhead_cam)
            frames.append(renderer.render().copy())

    cube_z = data.xpos[cube_id][2]
    success = cube_z > 0.05
    results.append(success)
    print(f"  RESULT: {'SUCCESS' if success else 'FAIL'} cube_z={cube_z:.4f}")

    if frames:
        out_path = f"assets/eval_trial_{trial}.mp4"
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video: {out_path}")

renderer.close()

successes = sum(results)
print(f"\n{'='*40}")
print(f"Results: {successes}/{NUM_TRIALS} successful ({100*successes/NUM_TRIALS:.0f}%)")
print(f"{'='*40}")

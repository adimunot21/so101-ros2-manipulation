"""Train ACT with action deltas disabled."""
import sys

# Monkey-patch BEFORE any training imports
from lerobot.policies.act import configuration_act
configuration_act.ACTConfig.action_delta_indices = property(lambda self: None)

from lerobot.policies.act.configuration_act import ACTConfig
cfg = ACTConfig()
assert cfg.action_delta_indices is None, "Patch failed!"
print(f"action_delta_indices: {cfg.action_delta_indices}")
print("Patch applied: action deltas DISABLED\n")

# Use the main entry point directly
from lerobot.scripts.lerobot_train import main as train_main

sys.argv = [
    "lerobot-train",
    "--dataset.repo_id=adimunot/so101-pick-lift",
    "--policy.type=act",
    "--policy.repo_id=adimunot/act-so101-pick-lift-v2",
    "--output_dir=outputs/train/act_so101_v2",
    "--batch_size=2",
    "--steps=30000",
    "--save_checkpoint=true",
    "--save_freq=10000",
    "--policy.device=cuda",
    "--wandb.enable=false",
    "--seed=42",
]

train_main()

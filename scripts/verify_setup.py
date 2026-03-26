#!/usr/bin/env python3
"""Verify Phase 0 environment setup for SO-101 ROS2 Manipulation Stack.

Validates that all dependencies are installed and accessible in the correct
environment. Run with --ros2 from the ROS2 venv, or --lerobot from the
conda lerobot env.

Usage:
    # ROS2 environment (from venv with ROS2 sourced):
    ros2ml
    python scripts/verify_setup.py --ros2

    # LeRobot environment (from conda):
    conda activate lerobot
    python scripts/verify_setup.py --lerobot

    # With MJCF model check:
    python scripts/verify_setup.py --ros2 --mjcf-path upstream/SO-ARM100/...

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# ── Result container ──────────────────────────────────────────────────

class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str

# ── Individual checks ────────────────────────────────────────────────

def check_python_version(min_major: int = 3, min_minor: int = 12) -> CheckResult:
    """Verify Python meets minimum version requirement."""
    v = sys.version_info
    passed = (v.major, v.minor) >= (min_major, min_minor)
    detail = f"{v.major}.{v.minor}.{v.micro}"
    return CheckResult(f"Python >= {min_major}.{min_minor}", passed, detail)


def check_import(module: str, display: str | None = None) -> CheckResult:
    """Import a module and report its version."""
    label = display or module
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "imported OK"))
        return CheckResult(label, True, f"v{version}")
    except ImportError as exc:
        return CheckResult(label, False, str(exc))


def check_ros2_env() -> CheckResult:
    """Verify ROS_DISTRO is set to jazzy."""
    distro = os.environ.get("ROS_DISTRO", "")
    if distro == "jazzy":
        return CheckResult("ROS_DISTRO=jazzy", True, distro)
    if distro:
        return CheckResult("ROS_DISTRO=jazzy", False, f"got '{distro}'")
    return CheckResult("ROS_DISTRO=jazzy", False, "not set — source setup.zsh first")


def check_rmw() -> CheckResult:
    """Verify CycloneDDS middleware is configured."""
    rmw = os.environ.get("RMW_IMPLEMENTATION", "")
    expected = "rmw_cyclonedds_cpp"
    if rmw == expected:
        return CheckResult("RMW=CycloneDDS", True, rmw)
    return CheckResult("RMW=CycloneDDS", False, f"got '{rmw}' — set RMW_IMPLEMENTATION")


def check_ros2_pkg(pkg: str) -> CheckResult:
    """Check that a ROS2 package is installed and locatable."""
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", pkg],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return CheckResult(f"ros2 pkg: {pkg}", True, result.stdout.strip())
        return CheckResult(f"ros2 pkg: {pkg}", False, "not found")
    except FileNotFoundError:
        return CheckResult(f"ros2 pkg: {pkg}", False, "'ros2' command not found")
    except subprocess.TimeoutExpired:
        return CheckResult(f"ros2 pkg: {pkg}", False, "timed out")


def check_command(cmd: list[str], label: str) -> CheckResult:
    """Run a shell command and report success/failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            first_line = result.stdout.strip().split("\n")[0][:100]
            return CheckResult(label, True, first_line)
        err = result.stderr.strip().split("\n")[0][:100]
        return CheckResult(label, False, err)
    except FileNotFoundError:
        return CheckResult(label, False, "command not found")
    except subprocess.TimeoutExpired:
        return CheckResult(label, False, "timed out")


def check_torch_cuda() -> CheckResult:
    """Verify PyTorch sees CUDA and report GPU info."""
    try:
        import torch
    except ImportError:
        return CheckResult("PyTorch CUDA", False, "torch not installed")

    if not torch.cuda.is_available():
        return CheckResult(
            "PyTorch CUDA", False,
            f"torch {torch.__version__} — cuda not available",
        )

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    # Handle attribute name differences across PyTorch versions
    vram_bytes = getattr(
        props, "total_mem",
        getattr(props, "total_memory", getattr(props, "total_global_mem", 0)),
    )
    vram_gb = vram_bytes / (1024 ** 3)
    return CheckResult(
        "PyTorch CUDA", True,
        f"torch {torch.__version__} | {name} | {vram_gb:.1f} GB",
    )


def check_mujoco_model(mjcf_path: str) -> CheckResult:
    """Load an MJCF file in MuJoCo and report model stats."""
    try:
        import mujoco
    except ImportError:
        return CheckResult("SO-101 MJCF load", False, "mujoco not installed")

    path = Path(mjcf_path).expanduser().resolve()
    if not path.exists():
        return CheckResult("SO-101 MJCF load", False, f"not found: {path}")

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        return CheckResult(
            "SO-101 MJCF load", True,
            f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}",
        )
    except Exception as exc:
        return CheckResult("SO-101 MJCF load", False, str(exc)[:120])


def check_venv_active() -> CheckResult:
    """Check if running inside the expected venv."""
    venv = os.environ.get("VIRTUAL_ENV", "")
    if "so101" in venv:
        return CheckResult("venv active (so101)", True, venv)
    if venv:
        return CheckResult("venv active (so101)", False, f"wrong venv: {venv}")
    return CheckResult("venv active (so101)", False, "no venv — activate ros2_venvs/so101")


def check_conda_env() -> CheckResult:
    """Check if the lerobotROS conda env is active."""
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env == "lerobotROS":
        return CheckResult("conda env: lerobotROS", True, env)
    if env:
        return CheckResult("conda env: lerobotROS", False, f"wrong env: {env}")
    return CheckResult("conda env: lerobotROS", False, "no conda env — run: conda activate lerobot")


def check_numpy_version() -> CheckResult:
    """Verify NumPy < 2 (PyTorch compatibility)."""
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major < 2:
            return CheckResult("NumPy < 2.0", True, f"v{np.__version__}")
        return CheckResult("NumPy < 2.0", False, f"v{np.__version__} — install numpy<2")
    except ImportError:
        return CheckResult("NumPy < 2.0", False, "not installed")


# ── Runner ────────────────────────────────────────────────────────────

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def run_section(title: str, checks: list[CheckResult]) -> bool:
    """Print a section of check results. Returns True if all passed."""
    print(f"\n{ANSI_BOLD}=== {title} ==={ANSI_RESET}")
    col_width = max(len(c.name) for c in checks) + 2
    all_ok = True

    for chk in checks:
        if chk.passed:
            tag = f"{ANSI_GREEN}✓ PASS{ANSI_RESET}"
        else:
            tag = f"{ANSI_RED}✗ FAIL{ANSI_RESET}"
            all_ok = False
        print(f"  {tag}  {chk.name:<{col_width}} {chk.detail}")

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Phase 0 setup for SO-101 ROS2 Manipulation Stack",
    )
    parser.add_argument(
        "--ros2", action="store_true",
        help="Check ROS2 venv environment (perception, planning, policy inference)",
    )
    parser.add_argument(
        "--lerobot", action="store_true",
        help="Check LeRobot conda environment (training, demo collection)",
    )
    parser.add_argument(
        "--mjcf-path", type=str, default="",
        help="Path to SO-101 MJCF XML file to test model loading",
    )
    args = parser.parse_args()

    if not args.ros2 and not args.lerobot:
        parser.print_help()
        print(f"\n{ANSI_RED}Specify --ros2 or --lerobot (or both).{ANSI_RESET}")
        return 1

    all_passed = True

    # ── System checks (always) ────────────────────────────────────────
    system_checks = [
        check_python_version(),
        check_command(["git", "--version"], "git"),
        check_command(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            "NVIDIA GPU",
        ),
    ]
    all_passed &= run_section("System", system_checks)

    # ── ROS2 environment ──────────────────────────────────────────────
    if args.ros2:
        env_checks = [
            check_venv_active(),
            check_ros2_env(),
            check_rmw(),
        ]
        all_passed &= run_section("ROS2 Environment", env_checks)

        pkg_checks = [
            check_ros2_pkg("controller_manager"),
            check_ros2_pkg("joint_trajectory_controller"),
            check_ros2_pkg("moveit"),
            check_ros2_pkg("cv_bridge"),
            check_ros2_pkg("robot_state_publisher"),
            check_ros2_pkg("xacro"),
        ]
        all_passed &= run_section("ROS2 Packages", pkg_checks)

        python_checks = [
            check_import("rclpy"),
            check_import("sensor_msgs", "sensor_msgs"),
            check_import("geometry_msgs", "geometry_msgs"),
            check_import("mujoco"),
            check_import("torch", "PyTorch"),
            check_torch_cuda(),
            check_numpy_version(),
            check_import("ultralytics", "YOLOv8 (ultralytics)"),
            check_import("cv2", "OpenCV"),
            check_import("gymnasium"),
            check_import("yaml", "PyYAML"),
            check_import("scipy"),
        ]
        all_passed &= run_section("ROS2 Python Libraries", python_checks)

    # ── LeRobot environment ───────────────────────────────────────────
    if args.lerobot:
        env_checks = [check_conda_env()]
        all_passed &= run_section("LeRobot Environment", env_checks)

        lib_checks = [
            check_import("lerobot"),
            check_import("torch", "PyTorch"),
            check_torch_cuda(),
            check_numpy_version(),
            check_import("mujoco"),
            check_import("gymnasium"),
            check_import("wandb"),
            check_import("matplotlib"),
            check_import("scipy"),
        ]
        all_passed &= run_section("LeRobot Python Libraries", lib_checks)

    # ── MJCF model ────────────────────────────────────────────────────
    if args.mjcf_path:
        model_checks = [check_mujoco_model(args.mjcf_path)]
        all_passed &= run_section("Model Verification", model_checks)

    # ── Summary ───────────────────────────────────────────────────────
    print()
    if all_passed:
        print(f"{ANSI_GREEN}{ANSI_BOLD}✓ All checks passed — ready for Phase 1.{ANSI_RESET}")
        return 0

    print(f"{ANSI_RED}{ANSI_BOLD}✗ Some checks failed — fix issues above before proceeding.{ANSI_RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

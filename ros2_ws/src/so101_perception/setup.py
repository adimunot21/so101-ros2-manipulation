"""Setup for so101_perception package.

In ROS2, Python packages use setup.py (not CMakeLists.txt) to:
  1. Declare which Python modules to install
  2. Register console_scripts as ROS2 node entry points
  3. Install data files (configs, launch files)

When you run "colcon build", it calls this setup.py which installs
the Python code so "ros2 run so101_perception detection_node" works.
"""
from setuptools import find_packages, setup
import os
from glob import glob

package_name = "so101_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        # Register package with ament index (required for ros2 pkg find)
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Install launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        # Install config files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            # This registers "detection_node" as a ROS2 executable.
            # "ros2 run so101_perception detection_node" calls
            # so101_perception.detection_node:main
            "detection_node = so101_perception.detection_node:main",
        ],
    },
)

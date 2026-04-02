from setuptools import find_packages, setup
import os
from glob import glob

package_name = "so101_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "pick_place_node = so101_manipulation.pick_place_node:main",
        ],
    },
)

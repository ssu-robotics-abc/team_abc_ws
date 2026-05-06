import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    abc_bringup_dir = get_package_share_directory("abc_bringup")

    realsense_launch = os.path.join(
        abc_bringup_dir,
        "launch",
        "realsense.launch.py",
    )

    doosan_moveit_launch = os.path.join(
        abc_bringup_dir,
        "launch",
        "doosan_rviz.launch.py",
        # "doosan_moveit.launch.py",
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(doosan_moveit_launch),
            launch_arguments={
                "mode": "real",
                "model": "m0609",
                "host": "192.168.1.100",
                "port": "12345",
            }.items(),
        ),

    ])
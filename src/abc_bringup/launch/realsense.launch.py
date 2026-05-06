import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    realsense_launch_file = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch",
        "rs_launch.py",
    )

    depth_profile = LaunchConfiguration("depth_profile")
    color_profile = LaunchConfiguration("color_profile")
    initial_reset = LaunchConfiguration("initial_reset")
    align_depth = LaunchConfiguration("align_depth")

    return LaunchDescription([
        DeclareLaunchArgument(
            "depth_profile",
            default_value="640x480x30",
        ),
        DeclareLaunchArgument(
            "color_profile",
            default_value="640x480x30",
        ),
        DeclareLaunchArgument(
            "initial_reset",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "align_depth",
            default_value="true",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments={
                "depth_module.depth_profile": depth_profile,
                "rgb_camera.color_profile": color_profile,
                "initial_reset": initial_reset,
                "align_depth.enable": align_depth,
            }.items(),
        ),
    ])
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    abc_bringup_dir = get_package_share_directory("abc_bringup")
    abc_speech_dir = get_package_share_directory("abc_speech")
    abc_order_dir = get_package_share_directory("abc_order")
    vlm_dir = get_package_share_directory("vlm_select")

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

    speech_launch = os.path.join(
        abc_speech_dir,
        "launch",
        "speech.launch.py",
    )
    
    order_launch = os.path.join(
        abc_order_dir,
        "launch",
        "order.launch.py",
    )

    vlm_launch = os.path.join(
        vlm_dir,
        "launch",
        "vlm_select.launch.py",
    )

    return LaunchDescription([
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(realsense_launch),
        # ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(doosan_moveit_launch),
        #     launch_arguments={
        #         "mode": "real",
        #         "model": "m0609",
        #         "host": "192.168.1.100",
        #         "port": "12345",
        #     }.items(),
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(speech_launch),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(order_launch),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vlm_launch),
        ),
    ])
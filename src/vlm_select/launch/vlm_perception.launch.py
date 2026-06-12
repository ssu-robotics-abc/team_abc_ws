from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py"
            )
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
        }.items()
    )

    vlm_perception_node = Node(
        package="vlm_select",
        executable="vlm_perception_node",
        name="vlm_perception_node",
        output="screen",
    )

    return LaunchDescription([
        realsense_launch,
        vlm_perception_node,
    ])
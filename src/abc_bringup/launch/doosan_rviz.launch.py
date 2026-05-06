import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    dsr_rviz_launch_file = os.path.join(
        get_package_share_directory("dsr_bringup2"),
        "launch",
        "dsr_bringup2_rviz.launch.py",
    )

    mode = LaunchConfiguration("mode")
    model = LaunchConfiguration("model")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    color = LaunchConfiguration("color")

    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="real",
            description="Doosan robot mode: real or virtual",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="m0609",
            description="Doosan robot model",
        ),
        DeclareLaunchArgument(
            "host",
            default_value="192.168.1.100",
            description="Doosan robot controller IP",
        ),
        DeclareLaunchArgument(
            "port",
            default_value="12345",
            description="Doosan robot controller port",
        ),
        DeclareLaunchArgument(
            "color",
            default_value="white",
            description="Robot color: white or blue",
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(dsr_rviz_launch_file),
            launch_arguments={
                "mode": mode,
                "model": model,
                "host": host,
                "port": port,
                "color": color,
            }.items(),
        ),
    ])
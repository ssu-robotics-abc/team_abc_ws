import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package launch paths
    abc_bringup_dir = get_package_share_directory("abc_bringup")
    abc_manipulation_dir = get_package_share_directory("abc_manipulation")

    realsense_launch = os.path.join(
        abc_bringup_dir,
        "launch",
        "realsense.launch.py",
    )

    manipulation_launch = os.path.join(
        abc_manipulation_dir,
        "launch",
        "manipulation.launch.py",
    )

    # Launch arguments
    use_realsense = LaunchConfiguration("use_realsense")
    use_fastapi = LaunchConfiguration("use_fastapi")
    fastapi_dir = LaunchConfiguration("fastapi_dir")
    publish_scan_done_delay = LaunchConfiguration("publish_scan_done_delay")

    # RealSense camera launch
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        condition=IfCondition(use_realsense),
    )

    # Manipulation/control launch
    manipulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(manipulation_launch),
    )

    # Vision processing node: ros2 run abc_perception yolo_node
    yolo_node = Node(
        package="abc_perception",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
    )

    # VLM node: ros2 run vlm_select vlm_node
    vlm_node = Node(
        package="vlm_select",
        executable="vlm_node",
        name="vlm_node",
        output="screen",
    )

    # STT node: ros2 run abc_speech stt_node
    stt_node = Node(
        package="abc_speech",
        executable="stt_node",
        name="stt_node",
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_realsense",
            default_value="true",
            description="Start RealSense camera launch before vision processing.",
        ),
        DeclareLaunchArgument(
            "use_fastapi",
            default_value="true",
            description="Start FastAPI server with ./run.sh. If false, publish /scan_done once.",
        ),
        DeclareLaunchArgument(
            "fastapi_dir",
            default_value=os.environ.get(
                "ABC_FASTAPI_DIR",
                os.path.expanduser("~/00_T/01/r/FastAPI"),
            ),
            description="Directory containing FastAPI run.sh. Can also be set with ABC_FASTAPI_DIR.",
        ),

        realsense,
        manipulation,
        yolo_node,
        vlm_node,
        stt_node,
    ])
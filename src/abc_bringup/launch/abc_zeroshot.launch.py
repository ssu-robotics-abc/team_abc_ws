import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    abc_bringup_dir = get_package_share_directory("abc_bringup")
    abc_manipulation_dir = get_package_share_directory("abc_manipulation")

    # 1. ros2 launch abc_bringup demo.launch.py
    #    -> RealSense 카메라 + Doosan MoveIt(move_group) 구동
    demo_launch = os.path.join(abc_bringup_dir, "launch", "demo.launch.py")

    # 2. ros2 launch abc_manipulation manipulation.launch.py
    #    -> MoveIt 기반 pick/scan/place/task_planner 노드 (move_group 필요)
    manipulation_launch = os.path.join(
        abc_manipulation_dir, "launch", "manipulation.launch.py"
    )

    # demo: 카메라 + move_group 먼저 띄운다 (t=0)
    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(demo_launch),
    )

    # manipulation: move_group 이 올라온 뒤 실행 (t=8s)
    manipulation = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(manipulation_launch),
            ),
        ],
    )

    # 3. ros2 run vlm_select vlm_perception_node -> 카메라 필요 (zero-shot, t=12s)
    vlm_perception_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="vlm_select",
                executable="vlm_perception_node",
                name="vlm_perception_node",
                output="screen",
            ),
        ],
    )

    # 4. ros2 run abc_speech stt_node -> 마이크 입력 (독립 실행, t=14s)
    stt_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="abc_speech",
                executable="stt_node",
                name="stt_node",
                output="screen",
            ),
        ],
    )

    return LaunchDescription([
        demo,
        manipulation,
        vlm_perception_node,
        stt_node,
    ])

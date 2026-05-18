from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    stt_node = Node(
        package="abc_speech",
        executable="stt_node",
        name="stt_node",
        output="screen",
        parameters=[
            {"language": "ko-KR"},
            {"device_index": -1},
            {"energy_threshold": 50.0},
            {"pause_threshold": 0.8},
            {"phrase_time_limit": 5.0},
            {"dynamic_energy": True},
            {"ambient_duration": 2.0},
        ],
    )

    tts_node = Node(
        package="abc_speech",
        executable="tts_node",
        name="tts_node",
        output="screen",
    )

    return LaunchDescription([
        stt_node,
        tts_node,
    ])
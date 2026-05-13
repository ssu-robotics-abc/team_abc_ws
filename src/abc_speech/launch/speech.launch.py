from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    stt_node = Node(
        package="dsr_practice",
        executable="stt_node",
        output="screen",
        parameters=[
            {"language": "ko-KR"},
            {"device_index": 1},
            {"energy_threshold": 300.0},
            {"pause_threshold": 0.8},
            {"phrase_time_limit": 5.0},
            {"dynamic_energy": True},
            {"ambient_duration": 1.0},
        ],
    )


    return LaunchDescription([stt_node])
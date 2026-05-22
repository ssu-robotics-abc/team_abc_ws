from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    vlm_node = Node(
        package="vlm_select",
        executable="vlm_node",
        name="vlm_node",
        output="screen",
    )

    parse_node = Node(
        package="vlm_select",
        executable="parse_order_server",
        name="parse_order_server",
        output="screen",
    )

    return LaunchDescription([
        vlm_node, parse_node,
    ])
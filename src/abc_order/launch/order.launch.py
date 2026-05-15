from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    order_manager_node = Node(
        package="abc_order",
        executable="order_manager",
        name="order_manager",
        output="screen",
    )

    return LaunchDescription([
        order_manager_node,
    ])
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Pick Item 서버 노드
    pick_node = Node(
        package='abc_manipulation',
        executable='pick_item',
        name='pick_item_server',
        output='screen',
    )
        
    # 2. Scan Barcode 서버 노드
    scan_node = Node(
        package='abc_manipulation',
        executable='scan_barcode',
        name='scan_barcode_server',
        output='screen',
    )
        
    # 3. Place Item 서버 노드
    place_node = Node(
        package='abc_manipulation',
        executable='place_item',
        name='place_item_server',
        output='screen',
    )
        
    # 4. Task Planner 노드
    planner_node = Node(
        package='abc_manipulation',
        executable='task_planner',
        name='task_planner_node',
        output='screen',
    )
    
    return LaunchDescription([
        pick_node, scan_node, place_node, planner_node
    ])
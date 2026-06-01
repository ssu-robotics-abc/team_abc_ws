from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 💡 robot_name과 package_name을 각각 명시하여 _moveit_config가 중복 결합되는 현상을 방지합니다.
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="m0609",
            package_name="dsr_moveit_config_m0609"
        )
        .robot_description(file_path="config/m0609.urdf.xacro")
        .robot_description_semantic(file_path="config/dsr.srdf")       # 진짜 srdf 파일 지정
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor()
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .to_moveit_configs()
    )

    moveit_py_params = PathJoinSubstitution(
        [FindPackageShare("abc_manipulation"), "config", "moveit_py.yaml"]
    )

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
        parameters=[
            moveit_config.to_dict(), # 수정: 와일드카드로 래핑된 로봇 설정 데이터 주입
            moveit_py_params         # 이전에 검토한 커스텀 YAML
        ],
    )
        
    # 3. Place Item 서버 노드
    place_node = Node(
        package='abc_manipulation',
        executable='place_item',
        name='place_item_server',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            moveit_py_params
        ],
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

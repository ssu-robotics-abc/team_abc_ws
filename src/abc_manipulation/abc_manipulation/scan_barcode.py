import time
import math
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

print("[디버그] 1. 라이브러리 임포트 중...")
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState
from abc_manipulation.onrobot import RG
from abc_interfaces.action import ScanBarcode

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY_SCALE = 0.6    
ACCELERATION_SCALE = 0.6 

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

class ScanBarcodeServer(Node):
    def __init__(self):
        # 💡 충돌 방지를 위해 파이썬 서브 노드 이름을 다르게 임시 지정
        super().__init__('scan_barcode_action_helper')
        print("[디버그] 3. ScanBarcodeServer 클래스 내부 진입 완료.")
        cb_group = ReentrantCallbackGroup()

        # ---------------------------------------------------------------------
        # 함정 ① 해결: 그리퍼 하드웨어 연결 체크 (테스트 시 블로킹 방지)
        # ---------------------------------------------------------------------
        print("[디버그] 4. OnRobot 그리퍼 연결 시도 중... (여기서 멈추면 네트워크/하드웨어 문제입니다)")
        try:
            # 💡 하드웨어 미연결 상태에서 테스트하려면 아래 한 줄을 주석 처리하세요.
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            print("[디버그] => 그리퍼 연결 성공!")
        except Exception as e:
            print(f"[디버그] => 그리퍼 연결 예외 발생 (무시하고 진행): {e}")

        # ---------------------------------------------------------------------
        # 함정 ③ 해결: MoveItPy 엔진 초기화
        # ---------------------------------------------------------------------
        print("[디버그] 5. MoveItPy 엔진 초기화 시작... (여기서 멈추면 런치 파일 파라미터 문제입니다)")
        try:
            # 런치 파일의 파라미터를 정상 분할해 오기 위해 전용 이름 지정
            self.robot = MoveItPy(node_name="scan_barcode_server")
            self.arm = self.robot.get_planning_component("manipulator")
            print("[디버그] => MoveItPy 엔진 로드 완료!")
        except Exception as e:
            print(f"❌ MoveItPy 치명적 초기화 에러: {e}")
            sys.exit(1)

        # 관절 공간 데이터 정의
        self.joints_home_horiz = {
            "joint_1": math.radians(1.0), "joint_2": math.radians(38.89), "joint_3": math.radians(123.21),
            "joint_4": math.radians(-0.08), "joint_5": math.radians(-71.09), "joint_6": math.radians(89.94)
        }
        self.joints_scanner = {
            "joint_1": math.radians(-27.84), "joint_2": math.radians(29.23), "joint_3": math.radians(51.69),
            "joint_4": math.radians(-2.09), "joint_5": math.radians(99.83), "joint_6": math.radians(54.72)
        }

        print("[디버그] 6. ROS 2 통신인터페이스(Action/Topic) 생성 중...")
        self._action_server = ActionServer(
            self, ScanBarcode, 'scan_barcode', self.execute_callback, callback_group=cb_group
        )
        self.subscription = self.create_subscription(
            String, 'scan_done', self.scan_callback, 10, callback_group=cb_group
        )
        self.is_scanned = False
        self.received_product_id = None
        print("[디버그] 7. ScanBarcodeServer 모든 초기화 완료.")

    def scan_callback(self, msg):
        self.received_product_id = msg.data
        self.is_scanned = True

    def plan_and_execute_joints(self, joint_goal, v_scale=None, a_scale=None):
        """MoveIt 2를 이용해 목표 Joint 상태로 궤적을 계획하고 실행하는 공용 헬퍼 함수"""
        self.arm.set_start_state_to_current_state()
        
        # ---------------------------------------------------------------------
        # 🎯 [명세 교정] RobotState 객체 생성 및 순서 보장 리스트 주입
        # ---------------------------------------------------------------------
        # 1. 로봇 모델 정보를 기반으로 깨끗한 RobotState 객체를 새로 생성합니다.
        goal_state = RobotState(self.robot.get_robot_model())
        
        # 2. 딕셔너리의 키 값을 URDF 관절 순서(1번~6번)에 맞게 정렬된 리스트로 펼쳐줍니다.
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        joint_values = [joint_goal[name] for name in joint_names]
        
        # 3. 플래닝 그룹("manipulator") 명의 근육 계층에 정렬된 각도 리스트를 주입합니다.
        goal_state.set_joint_group_positions("manipulator", joint_values)
        
        # 4. 완벽하게 포장된 RobotState를 set_goal_state 인자에 대입합니다.
        self.arm.set_goal_state(robot_state=goal_state)
        # ---------------------------------------------------------------------

        # 속도 및 가속도 파라미터 제어 프로필 적용 (기존 유지)
        req_params = PlanRequestParameters(self.robot)
        if v_scale is not None:
            req_params.max_velocity_scaling_factor = v_scale
        if a_scale is not None:
            req_params.max_acceleration_scaling_factor = a_scale
            
        # 계획 및 실행
        plan_result = self.arm.plan(parameters=req_params)
        
        if not plan_result:
            log.error("Planning failed")
            return False
        log.info("Executing plan")
        self.robot.execute(
            group_name="manipulator",
            robot_trajectory=plan_result.trajectory,
            blocking=True,
        )
        
        if plan_result:
            return self.robot.execute(plan_result.trajectory)
        return False

    def execute_callback(self, goal_handle):
        # 기존 로직과 동일
        self.is_scanned = False
        self.received_product_id = None
        product_id_str = goal_handle.request.product_id
        feedback_msg = ScanBarcode.Feedback()
        
        self.get_logger().info("제품 스캔 시퀀스 시작 (MoveIt 2).")
        feedback_msg.state = "스캐너 앞으로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        if not self.plan_and_execute_joints(self.joints_scanner):
            self.get_logger().error("⚠️ 스캐너 위치 이동 계획 실패!")
            goal_handle.abort()
            return ScanBarcode.Result(success=False, is_corrected=False)
        time.sleep(0.5)

        feedback_msg.state = "바코드 스캔 대기 및 툴 회전..."
        goal_handle.publish_feedback(feedback_msg)

        start_wait_time = time.time()
        timeout_duration = 30.0
        success_scan = False

        start_j6 = self.joints_scanner["joint_6"]
        step_rotation = math.radians(20.0)   
        steps = int(math.radians(360.0) / step_rotation)
        scan_goal_joints = self.joints_scanner.copy()
        
        for i in range(steps):
            if self.is_scanned:
                success_scan = True
                break
            if (time.time() - start_wait_time) > timeout_duration:
                break
            scan_goal_joints["joint_6"] = start_j6 + (i + 1) * step_rotation
            self.plan_and_execute_joints(scan_goal_joints, v_scale=0.15, a_scale=0.2)
            time.sleep(0.02)

        result = ScanBarcode.Result()
        if success_scan and (self.received_product_id == product_id_str):
            result.is_corrected = True
            result.success = True
        else:
            result.is_corrected = False
            result.success = False

        self.plan_and_execute_joints(self.joints_home_horiz)
        goal_handle.succeed()
        return result

def main(args=None):
    print("[디버그] 2. main 함수 진입 및 rclpy.init 호출 중...")
    rclpy.init(args=args)
    
    scan_server = ScanBarcodeServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(scan_server)

    print("[디버gu] 8. MultiThreadedExecutor spin 시작!")
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[Scan_Barcode] 사용자에 의해 종료되었습니다.")
    finally:
        scan_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
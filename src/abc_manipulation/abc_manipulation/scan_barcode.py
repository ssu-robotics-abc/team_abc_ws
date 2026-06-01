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
from moveit_msgs.msg import Constraints, JointConstraint
from abc_manipulation.onrobot import RG
from abc_interfaces.action import ScanBarcode

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY_SCALE = 0.6    
ACCELERATION_SCALE = 0.6 
GROUP_NAME = "manipulator"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
JOINT_6_MIN = -3.14
JOINT_6_MAX = 3.14

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
            self.arm = self.robot.get_planning_component(GROUP_NAME)
            print("[디버그] => MoveItPy 엔진 로드 완료!")
        except Exception as e:
            print(f"❌ MoveItPy 치명적 초기화 에러: {e}")
            sys.exit(1)

        # 관절 공간 데이터 정의
        self.joints_home_horiz = {
            "joint_1": math.radians(0.0), "joint_2": math.radians(0), "joint_3": math.radians(90),
            "joint_4": math.radians(0), "joint_5": math.radians(90), "joint_6": math.radians(0)
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

    def build_joint_constraints(self, joint_goal):
        constraints = Constraints()
        for joint_name in JOINT_NAMES:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = joint_goal[joint_name]
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return [constraints]

    def plan_and_execute_joints(self, joint_goal, v_scale=None, a_scale=None):
        """MoveIt 2를 이용해 목표 Joint 상태로 궤적을 계획하고 실행하는 공용 헬퍼 함수"""
        if not (JOINT_6_MIN <= joint_goal["joint_6"] <= JOINT_6_MAX):
            self.get_logger().error(
                f"joint_6 목표가 MoveIt 제한 범위를 벗어났습니다: {joint_goal['joint_6']:.3f} rad"
            )
            return False

        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=self.build_joint_constraints(joint_goal))

            req_params = PlanRequestParameters(self.robot)
            if v_scale is not None:
                req_params.max_velocity_scaling_factor = v_scale
            if a_scale is not None:
                req_params.max_acceleration_scaling_factor = a_scale

            plan_result = self.arm.plan(parameters=req_params)
            if not plan_result:
                self.get_logger().error("Planning failed")
                return False

            self.get_logger().info("Executing plan")
            self.robot.execute(
                group_name=GROUP_NAME,
                robot_trajectory=plan_result.trajectory,
                blocking=True,
            )
            return True
        except Exception as e:
            self.get_logger().error(f"MoveIt 계획/실행 중 예외 발생: {e}")
            return False

    def sweep_joint_6_for_scan(self, start_wait_time, timeout_duration):
        start_j6 = self.joints_scanner["joint_6"]
        step_rotation = math.radians(20.0)
        scan_goal_joints = self.joints_scanner.copy()

        target_positions = self._make_joint_6_scan_positions(start_j6, step_rotation)

        last_target_j6 = None
        for target_j6 in target_positions:
            if self.is_scanned:
                return True
            if (time.time() - start_wait_time) > timeout_duration:
                self.get_logger().warning("바코드 스캔 대기 시간이 초과되었습니다.")
                return False

            if last_target_j6 is not None and abs(target_j6 - last_target_j6) < 1e-6:
                continue

            scan_goal_joints["joint_6"] = target_j6

            if not self.plan_and_execute_joints(scan_goal_joints, v_scale=0.15, a_scale=0.2):
                self.get_logger().error(f"스캔 회전 이동 실패: joint_6={target_j6:.3f} rad")
                return False
            last_target_j6 = target_j6
            time.sleep(0.02)

        return self.is_scanned

    def _make_joint_6_scan_positions(self, start_j6, step):
        positions = []

        current = start_j6 + step
        while current <= JOINT_6_MAX:
            positions.append(current)
            current += step
        if not positions or abs(positions[-1] - JOINT_6_MAX) > 1e-6:
            positions.append(JOINT_6_MAX)

        positions.append(start_j6)

        current = start_j6 - step
        while current >= JOINT_6_MIN:
            positions.append(current)
            current -= step
        if abs(positions[-1] - JOINT_6_MIN) > 1e-6:
            positions.append(JOINT_6_MIN)

        return positions

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
        success_scan = self.sweep_joint_6_for_scan(start_wait_time, timeout_duration)

        result = ScanBarcode.Result()
        if success_scan:
            result.success = True
            result.is_corrected = self.received_product_id == product_id_str
            if result.is_corrected:
                self.get_logger().info("바코드 검증 성공: 상품 ID가 일치합니다.")
            else:
                self.get_logger().warning(
                    f"바코드 검증 불일치: 요청={product_id_str}, 수신={self.received_product_id}"
                )
        else:
            result.success = False
            result.is_corrected = False
            self.get_logger().error("바코드 스캔 실패: 바코드를 읽지 못했거나 로봇 이동이 실패했습니다.")

        if not self.plan_and_execute_joints(self.joints_home_horiz):
            self.get_logger().error("홈 위치 복귀 실패")

        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
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

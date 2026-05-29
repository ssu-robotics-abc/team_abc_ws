import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

# 두산 로봇 및 그리퍼 관련 라이브러리
import DR_init
from abc_manipulation.onrobot import RG
from abc_interfaces.action import ScanBarcode

# ========================================================================
# [사람이 개입해야 하는 부분 1] 물리적 환경 및 물품 치수 정의 (단위: mm)
# ========================================================================
# 💡 실제 환경에 맞게 이 두 값을 자로 재서 정확히 입력하셔야 공정이 성공합니다.
REAL_TABLE_Z = 5.0    # 티칭 펜던트로 측정한 실제 진열대 바닥의 Z 좌표
REAL_ITEM_HEIGHT = 133.0 # 빼빼로 상자나 음료수캔의 실제 총 높이 (세로 길이)

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
Z_SAFE = 400.0          # 장애물을 치지 않는 안전 상공 높이
VELOCITY, ACC = 60, 60  # 일반 이동 속도/가속도

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

# ========================================================================
# 두산 라이브러리 초기화 및 임포트
# ========================================================================
if not rclpy.ok():
    rclpy.init()

dsr_node = rclpy.create_node("scan_barcode_server_node", namespace=ROBOT_ID)
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "m0609"
DR_init.__dsr__node = dsr_node

try:
    # 💡 하이브리드 전략을 위해 task_compliance_ctrl(유연제어) 함수를 가져옵니다.
    from DSR_ROBOT2 import (get_current_posx, movej, movel, wait, amovel, 
                            get_tool_force, task_compliance_ctrl, release_compliance_ctrl)
    from DR_common2 import posx, posj
    
    try:
        from DR_common2 import DR_TOOL, DR_BASE
    except ImportError:
        DR_BASE = 0
        DR_TOOL = 1
except ImportError as e:
    print(f"❌ 두산 로봇 라이브러리 로드 실패: {e}")
    exit()


class ScanBarcodeServer(Node):
    def __init__(self):
        super().__init__('scan_barcode_server')
        cb_group = ReentrantCallbackGroup()

        # 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Scan_Barcode] 그리퍼 연결 성공")
        except Exception as e:
            self.get_logger().error(f"[Scan_Barcode] 그리퍼 연결 실패: {e}")

        # [사람이 개입해야 하는 부분 2] 관절 공간 티칭 데이터
        self.pos_home_horiz = posj([1, 38.89, 123.21, -0.08, -71.09, 89.94])  # 수평 시작 자세             
        self.pos_home_vert  = posj([0, -20, 120, 0, 15, 90])                  # 수직 전환 자세             
        self.pos_scanner    = posj([-27.84, 29.23, 51.69, -2.09, 99.83, 54.72]) # 스캐너 위치

        self._action_server = ActionServer(
            self, ScanBarcode, 'scan_barcode', self.execute_callback, callback_group=cb_group
        )
        self.subscription = self.create_subscription(
            String, 'scan_done', self.scan_callback, 10, callback_group=cb_group
        )
        self.is_scanned = False
        self.received_product_id = None

    def scan_callback(self, msg):
        self.received_product_id = msg.data
        self.is_scanned = True

    def execute_callback(self, goal_handle):
        self.is_scanned = False
        self.received_product_id = None
        product_id_str = goal_handle.request.product_id
        feedback_msg = ScanBarcode.Feedback()
        
        self.get_logger().info(f"▶️ [{product_id_str}] 하이브리드 제어 시퀀스 시작.")


        
        movej(self.pos_home_vert, VELOCITY, ACC)
        wait(0.5)
        
    

        # ---------------------------------------------------------------------
        # 로직 5) 잡은상태로 스캐너위치로 이동한다
        # ---------------------------------------------------------------------
        feedback_msg.state = "스캐너 앞으로 고속 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_scanner, VELOCITY, ACC) # 움찔거림 없는 관절 공간 고속 주행
        wait(0.5)

        # ---------------------------------------------------------------------
        # 로직 6) 스캐너위치에 도달하면 회전하면서 바코드가 스캔될때까지 기다린다
        # ---------------------------------------------------------------------
        feedback_msg.state = "바코드 스캔 대기 및 툴 회전..."
        goal_handle.publish_feedback(feedback_msg)

        start_wait_time = time.time()
        timeout_duration = 30.0
        success_scan = False

        self.get_logger().info("🔄 물품을 툴 좌표계 기준으로 한 바퀴 회전합니다.")
        scanner_pose = get_current_posx()[0]
        scan_target_pose = posx([scanner_pose[0], scanner_pose[1], scanner_pose[2], 
                                 scanner_pose[3], scanner_pose[4], scanner_pose[5] + 360.0])
        
        amovel(scan_target_pose, vel=20, acc=20, ref=DR_TOOL) # 360도 부드러운 회전 시작
        
        while rclpy.ok():
            if self.is_scanned:
                success_scan = True
                movel(get_current_posx()[0], vel=5, acc=150) # 인식 즉시 정지
                break
            if (time.time() - start_wait_time) > timeout_duration:
                self.get_logger().warn("⚠️ 스캔 타임아웃 발생.")
                movel(get_current_posx()[0], vel=5, acc=150)
                break
            wait(0.05)

        # 결과 반환 및 수평 복귀 마무리
        result = ScanBarcode.Result()
        if success_scan and (self.received_product_id == product_id_str):
            self.get_logger().info(f"✅ 바코드 일치 확인 완료 ({product_id_str})")
            result.is_corrected = True
            result.success = True
        else:
            result.is_corrected = False
            result.success = False

        movej(self.pos_home_horiz, VELOCITY, ACC)
        goal_handle.succeed()
        return result

def main(args=None):
    scan_server = ScanBarcodeServer()
    
    # 멀티스레드 실행기 (액션 서버와 두산 로봇 제어가 동시에 돌아가기 위함)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(scan_server)
    executor.add_node(dsr_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n[Scan_Barcode] 사용자에 의해 종료되었습니다.")
    finally:
        scan_server.destroy_node()
        dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
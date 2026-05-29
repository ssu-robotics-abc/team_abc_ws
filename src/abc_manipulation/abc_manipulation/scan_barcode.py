import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

# 두산 로봇 및 그리퍼 관련
import DR_init
from abc_manipulation.onrobot import RG

from abc_interfaces.action import ScanBarcode

# ======================
# 로봇 및 그리퍼 설정
# ======================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

class ScanBarcodeServer(Node):
    def __init__(self):
        super().__init__('scan_barcode_server')
        
        # 멀티스레드 처리를 위한 콜백 그룹
        cb_group = ReentrantCallbackGroup()

        # 1. 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Scan_Barcode] 그리퍼 연결 성공")
        except Exception as e:
            self.get_logger().error(f"[Scan_Barcode] 그리퍼 연결 실패: {e}")

        # 2. 주요 위치 정의 (전역 변수로 선언된 posj, posx 활용)
        self.pos_home = posj([0, 0, 135, 0, -45, 90])
        self.pos_scanner    = posj([-27.84, 29.23, 51.69, -2.09, 99.83, 54.72]) # 스캐너 위치
        self.safe_z_offset = 100.0

        # 3. Scan_Barcode 액션 서버 생성
        self._action_server = ActionServer(
            self,
            ScanBarcode,
            'scan_barcode',
            self.execute_callback,
            callback_group=cb_group
        )

        self.subscription = self.create_subscription(
            String, 'scan_done', self.scan_callback, 10, callback_group=cb_group
        )
        self.is_scanned = False
        self.received_product_id = None

        self.get_logger().info("🚀 [Scan_Barcode] Scan Barcode 서버가 시작되었습니다.")

    def scan_callback(self, msg):
        self.get_logger().info(f"앱[스캐너]으로 부터 신호 수신: {msg.data}")
        self.received_product_id = msg.data
        self.is_scanned = True

    def execute_callback(self, goal_handle):
        self.is_scanned = False
        self.received_product_id = None
        product_id = goal_handle.request.product_id

        self.get_logger().info(f"\n[Scan_Barcode 시작] 상품({product_id}) 스캐너로 이동합니다.")

        feedback_msg = ScanBarcode.Feedback()

        # 1. 홈으로 이동
        feedback_msg.state = "홈 위치 대기 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home, VELOCITY, ACC)
        wait(0.5)

        # 2. 스캐너 위치로 이동
        feedback_msg.state = "스캐너 앞으로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)

        movej(self.pos_scanner, VELOCITY, ACC)

        # 3. 스캐너에 물건 바코드가 읽힐때까지 기다림.
        feedback_msg.state = "스캔 대기 중..."
        goal_handle.publish_feedback(feedback_msg)

        start_wait_time = time.time()
        timeout_duration = 30.0

        success_scan = False
        while rclpy.ok():
            if self.is_scanned:
                success_scan = True
                break

            if(time.time() - start_wait_time) > timeout_duration:
                self.get_logger().warn("⚠️ [경고] 스캔 타임아웃 발생!")
                break
            
            wait(0.1)

        # 최종 성공 반환
        result = ScanBarcode.Result()
        if success_scan:
            # 요청한 ID와 실제 스캔된 ID 비교
            if self.received_product_id == product_id:
                self.get_logger().info(f"✅ 일치하는 상품 확인: {product_id}")
                result.is_corrected = True
            else:
                self.get_logger().error(f"❌ 상품 불일치! (요청:{product_id} / 인식:{self.received_product_id})")
                result.is_corrected = False
            result.success = True
        else:
            result.success = False
            result.is_corrected = False


        # 홈으로 복귀
        feedback_msg.state = "홈으로 복귀 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home, VELOCITY, ACC)
        
        goal_handle.succeed()
        return result

    

def main(args=None):
    rclpy.init(args=args)
    
    # 두산 로봇 API 초기화
    dsr_node = rclpy.create_node("scan_barcode_server_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # 클래스 외부에서도 Doosan 함수를 쓸 수 있도록 전역(global) 선언
    global get_current_posx, movej, movel, wait, posx, posj
    try:
        from DSR_ROBOT2 import get_current_posx, movej, movel, wait
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"DSR_ROBOT2 Import 실패: {e}")
        return

    # 서버 노드 생성 및 멀티스레드 실행
    scan_server = ScanBarcodeServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(scan_server)
    executor.add_node(dsr_node) # 두산 API 전용 노드도 같이 스핀

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        scan_server.destroy_node()
        dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
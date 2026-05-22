import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

# 두산 로봇 및 그리퍼 관련 기초 라이브러리
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

# ======================
# 두산 라이브러리 초기화 및 안전 임포트
# ======================
if not rclpy.ok():
    rclpy.init()

dsr_node = rclpy.create_node("scan_barcode_server_node", namespace=ROBOT_ID)
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    # 💡 스크립트를 죽이는 정지 함수를 빼고, 순수 제어 함수들만 깔끔하게 임포트합니다.
    from DSR_ROBOT2 import get_current_posx, movej, movel, wait, amovel, check_force_condition
    from DR_common2 import posx, posj
    
    try:
        from DR_common2 import DR_TOOL, DR_BASE
    except ImportError:
        DR_BASE = 0
        DR_TOOL = 1
except ImportError as e:
    print(f"❌ 두산 로봇 라이브러리 로드 실패: {e}")
    exit()

# ======================
# 안전을 위한 Z축 가이드 파라미터
# ======================
Z_SAFE = 400.0          # 자세를 바꿀 때 물체를 치지 않도록 올라갈 안전 Z 높이 (mm)
Z_SAFETY_LIMIT = 100.0  # [외력감지 마지노선] 센서 미작동 시 충돌을 막기 위한 강제 최하단 제한선 (mm)

class ScanBarcodeServer(Node):
    def __init__(self):
        super().__init__('scan_barcode_server')
        cb_group = ReentrantCallbackGroup()

        # 1. 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Scan_Barcode] 그리퍼 연결 성공")
        except Exception as e:
            self.get_logger().error(f"[Scan_Barcode] 그리퍼 연결 실패: {e}")

        # 2. 주요 위치 정의
        self.pos_home_horiz = posj([0, 0, 135, 0, -45, 90])              
        self.pos_home_vert  = posj([0, -20, 120, 0, 15, 90])             
        self.pos_scanner    = posx([408.0, -153.0, 342.0, 133.0, -172.0, -120.0])
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
        self.get_logger().info(f"앱[스캐너]으로부터 신호 수신: {msg.data}")
        self.received_product_id = msg.data
        self.is_scanned = True

    def execute_callback(self, goal_handle):
        self.is_scanned = False
        self.received_product_id = None
        
        product_id_str = goal_handle.request.product_id
        self.get_logger().info(f"\n[Scan_Barcode 시작] 요청 상품 바코드:({product_id_str}) 스캔 작업을 시작합니다.")

        feedback_msg = ScanBarcode.Feedback()
        self.get_logger().info(f"🔄 모든 상품에 대해 수평 홈 이동 및 수직 재그립 시퀀스를 구동합니다.")
        
        # 1) 원래 잡고 있던 수평 상태로 홈 위치 이동
        feedback_msg.state = "수평 홈 위치로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home_horiz, VELOCITY, ACC)
        wait(0.5)

        # 2) [외력 감지 하강] 바닥 접촉을 느끼며 안전하게 수직 하강
        feedback_msg.state = "바닥 접촉 감지하며 수직 하강 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        cur_p = get_current_posx()[0] 
        down_pose = posx([cur_p[0], cur_p[1], Z_SAFETY_LIMIT, cur_p[3], cur_p[4], cur_p[5]])
        
        # 안전 감속 하강 시작
        amovel(down_pose, vel=15, acc=20)
        
        # 🔥 [트랩 방어 1] 출발할 때 튀는 순간 관성 토크 노이즈를 무시하기 위해 0.5초간 센싱을 대기합니다.
        wait(0.5) 
        
        start_time = time.time()
        while rclpy.ok():
            # 안정성을 위해 임계값을 20N(약 2kg 저항)으로 세팅
            if check_force_condition(2, 20.0, ref=DR_BASE) == 1:
                # 🔥 [트랩 방어 2] drl_script_stop 대신, 현재 위치로 즉시 무빙 명령을 내려 부드럽게 브레이크를 겁니다.
                stop_p = get_current_posx()[0]
                movel(stop_p, vel=5, acc=10)
                self.get_logger().info("💥 [외력 감지] 선반 바닥 접촉 확인! 안전하게 하강을 중단합니다.")
                break
            
            if (time.time() - start_time) > 5.0:
                stop_p = get_current_posx()[0]
                movel(stop_p, vel=5, acc=10)
                self.get_logger().warn("⚠️ [경고] 외력 감지 타임아웃으로 하강을 강제 중단합니다.")
                break
            wait(0.01) 
        
        wait(0.5) 
        self.gripper.open_gripper()
        time.sleep(0.5)

        # 3) 그리퍼를 연 채로 안전 높이로 상승
        post_drop_p = get_current_posx()[0]
        up_pose = posx([post_drop_p[0], post_drop_p[1], Z_SAFE, post_drop_p[3], post_drop_p[4], post_drop_p[5]])
        movel(up_pose, VELOCITY, ACC)
        wait(0.5)

        # 4) 공중에서 수직 자세(JReady)로 그리퍼 방향 전환
        feedback_msg.state = "그리퍼 수직 자세 전환 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home_vert, VELOCITY, ACC)
        wait(0.5)

        v_pose = get_current_posx()[0]
        v_rx, v_ry, v_rz = v_pose[3:]

        # 정렬 교정: 아까 물품을 내려놓았던 정확한 X, Y 정위치 상공으로 수평 복귀
        feedback_msg.state = "물품 상공 안전 위치로 조치 정렬 중..."
        goal_handle.publish_feedback(feedback_msg)
        target_above_can = posx([cur_p[0], cur_p[1], Z_SAFE, v_rx, v_ry, v_rz])
        movel(target_above_can, VELOCITY, ACC)
        wait(0.5)

        # 5) [외력 감지 재그립] 수직 상태로 물품 상단에 터치할 때까지 정밀 수직 하강
        feedback_msg.state = "외력 감지하며 물품 접근 하강 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        pick_pose_limit = posx([cur_p[0], cur_p[1], Z_SAFETY_LIMIT, v_rx, v_ry, v_rz])
        amovel(pick_pose_limit, vel=15, acc=20) 
        
        # 🔥 [트랩 방어 1] 수직 출발 관성 노이즈 패스 대기
        wait(0.5) 

        start_time = time.time()
        while rclpy.ok():
            if check_force_condition(2, 20.0, ref=DR_BASE) == 1:
                # 🔥 [트랩 방어 2] 현재 위치 강제 멈춤 제어 적용
                stop_p = get_current_posx()[0]
                movel(stop_p, vel=5, acc=10)
                self.get_logger().info("💥 [외력 감지] 물품 상단 접촉 확인! 파지를 시작합니다.")
                break
            if (time.time() - start_time) > 5.0:
                stop_p = get_current_posx()[0]
                movel(stop_p, vel=5, acc=10)
                self.get_logger().warn("⚠️ [경고] 재그립 외력 감지 타임아웃 발생.")
                break
            wait(0.01)

        wait(0.5)
        # 수직 방향에서 물품 몸통 정밀 파지
        self.gripper.close_gripper_detail(width=600)
        time.sleep(0.5)

        # 6) 수직으로 꽉 잡은 상태로 안전 높이로 상승
        up_v_pose = posx([cur_p[0], cur_p[1], Z_SAFE, v_rx, v_ry, v_rz])
        movel(up_v_pose, VELOCITY, ACC)
        wait(0.5)

        # ----------------------------------------
        # 공통 로직: 스캐너 앞으로 이동 및 바코드 인식 단계
        # ----------------------------------------
        feedback_msg.state = "스캐너 앞으로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        movel(self.pos_scanner, VELOCITY, ACC)
        wait(0.5)

        feedback_msg.state = "스캔 대기 및 회전 중..."
        goal_handle.publish_feedback(feedback_msg)

        start_wait_time = time.time()
        timeout_duration = 30.0
        success_scan = False

        self.get_logger().info("🔄 모든 물품 제자리 360도 회전 스캔을 시작합니다.")
        total_rotated_angle = 0.0
        angle_step = 30.0
        
        while rclpy.ok():
            if self.is_scanned:
                success_scan = True
                break
            if (time.time() - start_wait_time) > timeout_duration:
                self.get_logger().warn("⚠️ [경고] 스캔 타임아웃 발생!")
                break

            if total_rotated_angle < 360.0:
                movel(posx([0, 0, 0, 0, 0, angle_step]), vel=20, acc=20, ref=DR_TOOL)
                total_rotated_angle += angle_step
                wait(0.4)
            else:
                time.sleep(0.1)

        # 최종 검증 결과 반환
        result = ScanBarcode.Result()
        if success_scan:
            if self.received_product_id == product_id_str:
                self.get_logger().info(f"✅ 일치하는 상품 확인 완료 (바코드: {product_id_str})")
                result.is_corrected = True
            else:
                self.get_logger().error(f"❌ 상품 불일치! (요청 바코드:{product_id_str} / 인식 바코드:{self.received_product_id})")
                result.is_corrected = False
            result.success = True
        else:
            result.success = False
            result.is_corrected = False

        # 기본 수평 홈 위치로 복귀하며 마무리
        feedback_msg.state = "홈으로 복귀 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home_horiz, VELOCITY, ACC)
        
        goal_handle.succeed()
        return result

def main(args=None):
    scan_server = ScanBarcodeServer()
    
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
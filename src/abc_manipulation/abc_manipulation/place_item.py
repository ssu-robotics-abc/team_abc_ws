import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# 두산 로봇 및 그리퍼 관련
import DR_init
from abc_manipulation.onrobot import RG

# 커스텀 액션 인터페이스
from abc_interfaces.action import PlaceItem

# ======================
# 로봇 및 그리퍼 설정
# ======================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

class PlaceItemServer(Node):
    def __init__(self):
        super().__init__('place_item_server')
        
        # 멀티스레드 처리를 위한 콜백 그룹
        cb_group = ReentrantCallbackGroup()

        # 1. 그리퍼 초기화
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Place_Item] 그리퍼 연결 성공")
        except Exception as e:
            self.get_logger().error(f"[Place_Item] 그리퍼 연결 실패: {e}")

        # 2. 주요 위치 정의 (전역 변수로 선언된 posj, posx 활용)
        self.pos_home = posj([0, -20, 120, 0, 15, 90])
        self.pos_checkout = posx([408.0, 153.0, 342.0, 33.0, 180.0, 100.0])  # 판매대
        self.pos_return = posx([408.0, -153.0, 342.0, 133.0, -172.0, -120.0]) # 반품대
        self.safe_z_offset = 100.0

        # 3. Place_Item 액션 서버 생성
        self._action_server = ActionServer(
            self,
            PlaceItem,
            'place_item',
            self.execute_callback,
            callback_group=cb_group
        )

        self.get_logger().info("🚀 [Place_Item] Place Item 서버가 시작되었습니다.")

    def execute_callback(self, goal_handle):
        """Place_Item 메인 로직: 분류 이송 및 물체 놓기"""
        is_match = goal_handle.request.is_corrected
        dest_name = "판매대" if is_match else "반품대"
        
        self.get_logger().info(f"\n[Place_Item 시작] 검증 결과({is_match})에 따라 {dest_name}로 이송합니다.")

        feedback_msg = PlaceItem.Feedback()

        # Step 1: 시작 전 홈 위치 이동 (파지 상태 유지 중)
        feedback_msg.state = "홈 위치 대기 중..."
        goal_handle.publish_feedback(feedback_msg)
        movej(self.pos_home, VELOCITY, ACC)
        wait(0.5)

        # Step 2: 목적지 결정
        target_pose = self.pos_checkout if is_match else self.pos_return

        # Step 3: 목적지 상단 안전 높이로 이동
        feedback_msg.state = f"{dest_name} 상단으로 접근 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        approach_target = posx(target_pose.copy())
        approach_target[2] += self.safe_z_offset
        movel(approach_target, VELOCITY, ACC)
        wait(0.2)
        
        # Step 4: 하강 및 물체 놓기
        feedback_msg.state = "물체 내려놓는 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        movel(target_pose, VELOCITY, ACC)
        wait(0.5)
        self.gripper.open_gripper()
        time.sleep(1.0)
        
        # Step 5: 다시 상승 및 홈 복귀
        feedback_msg.state = "작업 완료 후 홈 복귀 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        movel(approach_target, VELOCITY, ACC)
        movej(self.pos_home, VELOCITY, ACC)

        # 최종 성공 반환
        goal_handle.succeed()
        result = PlaceItem.Result()
        result.success = True
        self.get_logger().info(f"[Place_Item 완료] {dest_name}에 배치 성공")
        
        return result

def main(args=None):
    rclpy.init(args=args)
    
    # 두산 로봇 API 초기화
    dsr_node = rclpy.create_node("place_item_server_node", namespace=ROBOT_ID)
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
    place_server = PlaceItemServer()
    executor = MultiThreadedExecutor()
    executor.add_node(place_server)
    executor.add_node(dsr_node) # 두산 API 전용 노드도 같이 스핀

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        place_server.destroy_node()
        dsr_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import time

# 커스텀 Action 인터페이스 임포트 (패키지명에 맞게 수정 필요)
from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class DummyTaskServers(Node):
    def __init__(self):
        super().__init__('test_server')

        # 3개의 가짜 액션 서버 생성
        self._PickItem_server = ActionServer(self, PickItem, 'PickItem_pick_item', self.execute_PickItem)
        self._ScanBarcode_server = ActionServer(self, ScanBarcode, 'ScanBarcode_scan_barcode', self.execute_ScanBarcode)
        self._PlaceItem_server = ActionServer(self, PlaceItem, 'PlaceItem_place_item', self.execute_PlaceItem)

        self.get_logger().info("✅ 테스트용 더미 Task 1, 2, 3 서버가 모두 켜졌습니다. 명령을 대기합니다.")

    # ---------------------------------------------------------
    # [Task 1] 가짜 서버 로직 (파지 및 홈 복귀)
    # ---------------------------------------------------------
    def execute_PickItem(self, goal_handle):
        product_id = goal_handle.request.product_id
        self.get_logger().info(f"[Task 1 수신] 물품 ID {product_id} 파지 명령 받음")

        # 피드백 전송 및 딜레이 (로봇이 움직이는 척)
        feedback_msg = PickItem.Feedback()
        
        feedback_msg.state = "목표 물체 탐색 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.5)

        feedback_msg.state = "그리퍼로 물체 파지 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.5)

        # 결과 반환
        goal_handle.succeed()
        result = PickItem.Result()
        result.success = True
        self.get_logger().info("[Task 1 완료] 파지 성공")
        return result

    # ---------------------------------------------------------
    # [Task 2] 가짜 서버 로직 (바코드 스캔 및 일치 여부 확인)
    # ---------------------------------------------------------
    def execute_ScanBarcode(self, goal_handle):
        product_id = goal_handle.request.product_id
        self.get_logger().info(f"[Task 2 수신] 물품 ID {product_id} 바코드 스캔 명령 받음")

        feedback_msg = ScanBarcode.Feedback()
        
        feedback_msg.state = "바코드 스캐너 앞으로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.5)

        feedback_msg.state = "바코드 스캔 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.5)

        goal_handle.succeed()
        result = ScanBarcode.Result()
        result.success = True
        
        # 더미 데이터 생성: product_id가 1234면 일치(True), 아니면 불일치(False)로 흉내냄
        if product_id == 1234:
            result.is_corrected = True
            self.get_logger().info("[Task 2 완료] 검증 결과: 바코드 일치 (True)")
        else:
            result.is_corrected = False
            self.get_logger().info("[Task 2 완료] 검증 결과: 바코드 불일치 (False)")
            
        return result

    # ---------------------------------------------------------
    # [Task 3] 가짜 서버 로직 (분류 이송 및 복귀)
    # ---------------------------------------------------------
    def execute_PlaceItem(self, goal_handle):
        is_corrected = goal_handle.request.is_corrected
        destination = "판매대" if is_corrected else "반품대"
        self.get_logger().info(f"[Task 3 수신] 일치 여부({is_corrected})에 따라 [{destination}]로 이송 명령 받음")

        feedback_msg = PlaceItem.Feedback()
        
        feedback_msg.state = f"{destination}로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.5)

        feedback_msg.state = "물체 내려놓는 중 (그리퍼 개방)..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.0)

        goal_handle.succeed()
        result = PlaceItem.Result()
        result.success = True
        self.get_logger().info(f"[Task 3 완료] {destination}에 배치 완료\n")
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DummyTaskServers()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
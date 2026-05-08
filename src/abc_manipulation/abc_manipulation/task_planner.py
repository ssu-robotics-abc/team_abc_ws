import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# 커스텀 메시지 및 액션 임포트 (패키지 이름에 맞춰 수정 필요)
from abc_interfaces.msg import PurchaseItem
from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # 3개의 하위 노드와 통신할 Action Client 생성
        self.pick_client = ActionClient(self, PickItem, 'pick_item')
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode')
        self.place_client = ActionClient(self, PlaceItem, 'place_item')
        self.get_logger().info("Task Planner Node (Main) Started. Waiting for actions...")

    async def execute_orders(self, items):
        """
        주문 리스트(PurchaseItem[])를 받아 반복 작업을 수행하는 메인 루프
        """
        for item in items:
            product_id = item.product_id
            quantity = item.quantity

            self.get_logger().info(f"==== [주문 처리 시작] 물품 ID: {product_id}, 수량: {quantity} ====")

            # 수량만큼 반복 수행
            for i in range(quantity):
                self.get_logger().info(f"--- 진행 상황: {i+1} / {quantity} 개 ---")
                
                # ---------------------------------------------------------
                # [Task 1] 물품 파지 및 홈 이동
                # ---------------------------------------------------------
                self.get_logger().info(">> [Task 1] 호출: 물품 파지")
                task1_goal = Task1.Goal()
                task1_goal.product_id = product_id
                
                task1_result = await self.call_action(self.task1_client, task1_goal)
                if not task1_result or not task1_result.success:
                    self.get_logger().error("Task 1 실패! 다음 작업을 중단합니다.")
                    break  # 실패 시 에러 처리 (해당 아이템 스킵 또는 전체 중단)

                # ---------------------------------------------------------
                # [Task 2] 바코드 스캔 및 검증
                # ---------------------------------------------------------
                self.get_logger().info(">> [Task 2] 호출: 바코드 스캔")
                task2_goal = Task2.Goal()
                task2_goal.product_id = product_id
                
                task2_result = await self.call_action(self.task2_client, task2_goal)
                if not task2_result or not task2_result.success:
                    self.get_logger().error("Task 2 실패! 바코드 스캔 중 문제 발생.")
                    break

                # Task 2의 결과로부터 일치 여부 추출
                is_corrected = task2_result.is_corrected
                match_str = "일치" if is_corrected else "불일치"
                self.get_logger().info(f"Task 2 완료. 검증 결과: [{match_str}]")

                # ---------------------------------------------------------
                # [Task 3] 검증 결과에 따른 분류 이송 (판매대/반품대)
                # ---------------------------------------------------------
                self.get_logger().info(">> [Task 3] 호출: 최종 이송 및 놓기")
                task3_goal = Task3.Goal()
                task3_goal.is_corrected = is_corrected
                
                task3_result = await self.call_action(self.task3_client, task3_goal)
                if not task3_result or not task3_result.success:
                    self.get_logger().error("Task 3 실패! 분류 이송 중 문제 발생.")
                    break
                
                self.get_logger().info(f"--- 1개 처리 완료 ---\n")
                
        self.get_logger().info("==== 모든 주문 처리가 완료되었습니다 ====")

    async def call_action(self, action_client, goal_msg):
        """
        Action Client를 호출하고 결과를 반환받는 비동기 헬퍼 함수
        """
        # 서버가 켜질 때까지 대기
        action_client.wait_for_server()
        
        # 목표 전송
        send_goal_future = action_client.send_goal_async(goal_msg)
        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.get_logger().warn("Action 목표가 서버로부터 거절당했습니다.")
            return None

        # 결과 대기
        result_future = goal_handle.get_result_async()
        action_result = await result_future
        
        return action_result.result


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()

    # 테스트용 가상 주문 데이터 생성
    # 실제로는 Subscriber를 통해 abc_speech 등에서 데이터를 받아 트리거되어야 합니다.
    mock_item1 = PurchaseItem()
    mock_item1.product_id = 1234
    mock_item1.quantity = 2
    
    mock_item2 = PurchaseItem()
    mock_item2.product_id = 5678
    mock_item2.quantity = 1

    orders = [mock_item1, mock_item2]

    # 비동기 루프를 사용하여 순차 로직 실행
    # (ROS 2 Python에서 Node 클래스 내부 비동기 함수 실행 방법)
    import asyncio
    
    # rclpy.spin을 비동기와 함께 처리하기 위해 별도 스레드 대신 이벤트 루프 활용
    future = asyncio.ensure_future(node.execute_orders(orders))
    
    try:
        rclpy.spin_until_future_complete(node, future)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
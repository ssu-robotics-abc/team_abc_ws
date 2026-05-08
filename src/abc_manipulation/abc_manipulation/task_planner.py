import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import asyncio

from abc_interfaces.msg import PurchaseItem
from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.cb_group = ReentrantCallbackGroup()

        # 클라이언트 설정
        self.pick_client = ActionClient(self, PickItem, 'pick_item', callback_group=self.cb_group)
        # Task 2, 3는 연결 확인만 하거나 주석 처리 가능
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode', callback_group=self.cb_group)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=self.cb_group)

        self.get_logger().info("Task Planner Node Started.")
        self.create_timer(1.0, self.run_sequence, callback_group=self.cb_group)
        self.is_running = False

    async def run_sequence(self):
        if self.is_running: return
        self.is_running = True

        # 테스트 데이터: 1234 (초코파이) 한 개 주문
        items = [PurchaseItem(product_id=1234, quantity=1)]

        for item in items:
            self.get_logger().info(f"\n==== [처리 시작] ID: {item.product_id} ====")
            
            # 1. PickItem 실행
            self.get_logger().info(">> (Task 1) Pick 서버 연결 대기...")
            if not self.pick_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("Pick 서버가 오프라인입니다.")
                break
            
            self.get_logger().info(">> [Task 1] 호출: 물체 인식 및 파지 명령")
            goal_msg = PickItem.Goal(product_id=item.product_id)
            
            # 결과 대기 (인식 및 동작이 포함되어 시간이 걸림)
            res1 = await self.call_action(self.pick_client, goal_msg)
            
            if res1 and res1.success:
                self.get_logger().info("✅ Task 1 성공! (물체를 잡았습니다)")
                
                # 테스트를 위해 Task 2, 3 생략 혹은 더미 처리
                self.get_logger().info("테스트를 위해 다음 단계를 종료합니다.")
            else:
                self.get_logger().error("❌ Task 1 실패")
                break
            
        self.get_logger().info("==== 모든 테스트 종료 ====")

    async def call_action(self, client, goal):
        handle = await client.send_goal_async(goal, feedback_callback=self.fb_cb)
        if not handle.accepted: return None
        result = await handle.get_result_async()
        return result.result

    def fb_cb(self, msg):
        self.get_logger().info(f"   [실시간 상태] {msg.feedback.state}")

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
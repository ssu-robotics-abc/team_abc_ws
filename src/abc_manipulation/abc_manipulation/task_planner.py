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
        cb_group = ReentrantCallbackGroup()

        # 액션 클라이언트 생성 (서버와 이름을 정확히 맞춰야 함)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=cb_group)

        self.get_logger().info("Task Planner Node (Main) Started.")
        self.timer = self.create_timer(1.0, self.run_sequence, callback_group=cb_group)
        self.is_running = False

    async def run_sequence(self):
        if self.is_running: return
        self.is_running = True
        self.timer.cancel()

        items = [PurchaseItem(product_id=1234, quantity=1)]

        for item in items:
            self.get_logger().info(f"\n==== [처리 시작] ID: {item.product_id} ====")
            
            # ★ 임시 테스트를 위해 Task 1, 2 생략하고 결과값 강제 설정
            is_match = True 
            
            # 3. PlaceItem
            self.get_logger().info(">> (Task 3) Place 서버 연결 대기...")
            
            # 서버가 켜질 때까지 무한 대기하지 않도록 timeout을 주는 방식 권장
            if not self.place_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error("❌ Place 서버를 찾을 수 없습니다! 이름을 확인하세요.")
                return

            self.get_logger().info(">> [Task 3] 호출 시도")
            goal_msg = PlaceItem.Goal()
            goal_msg.is_corrected = is_match
            
            # 비동기 호출
            result = await self.call_action(self.place_client, goal_msg)
            
            if result and result.success:
                self.get_logger().info("✅ Task 3 작업 성공!")
            else:
                self.get_logger().error("❌ Task 3 작업 실패")
            
        self.get_logger().info("==== 모든 작업 종료 ====")

    async def call_action(self, client, goal):
        handle = await client.send_goal_async(goal, feedback_callback=self.fb_cb)
        if not handle.accepted: 
            self.get_logger().error("Goal 거절됨")
            return None
        result = await handle.get_result_async()
        return result.result

    def fb_cb(self, msg):
        self.get_logger().info(f"   [진행상태] {msg.feedback.state}")

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
        if rclpy.ok():
            rclpy.shutdown()
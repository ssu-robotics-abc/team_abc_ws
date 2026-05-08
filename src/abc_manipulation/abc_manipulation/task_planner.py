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

        # ✅ 서버 코드와 똑같은 이름을 사용합니다.
        self.pick_client = ActionClient(self, PickItem, 'pick_item', callback_group=cb_group)
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode', callback_group=cb_group)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=cb_group)

        self.get_logger().info("Task Planner Node (Main) Started.")
        self.timer = self.create_timer(1.0, self.run_sequence, callback_group=cb_group)
        self.is_running = False

    async def run_sequence(self):
        if self.is_running: return
        self.is_running = True
        self.timer.cancel()

        # 가상 데이터
        items = [PurchaseItem(product_id=1234, quantity=1)]

        for item in items:
            self.get_logger().info(f"\n==== [처리 시작] ID: {item.product_id} ====")
            
            # 1. PickItem
            self.get_logger().info(">> (Task 1) Pick 서버 연결 대기...")
            # 비동기 루프 안에서는 wait_for_server()가 길을 막으므로 아래 루프가 안전합니다.
            while not self.pick_client.server_is_ready():
                await asyncio.sleep(0.1)
            
            self.get_logger().info(">> [Task 1] 호출")
            res1 = await self.call_action(self.pick_client, PickItem.Goal(product_id=item.product_id))
            if not res1 or not res1.success: break

            # 2. ScanBarcode
            self.get_logger().info(">> (Task 2) Scan 서버 연결 대기...")
            while not self.scan_client.server_is_ready():
                await asyncio.sleep(0.1)
            
            self.get_logger().info(">> [Task 2] 호출")
            res2 = await self.call_action(self.scan_client, ScanBarcode.Goal(product_id=item.product_id))
            if not res2 or not res2.success: break
            
            is_match = res2.is_corrected
            self.get_logger().info(f"결과: {'일치' if is_match else '불일치'}")

            # 3. PlaceItem
            self.get_logger().info(">> (Task 3) Place 서버 연결 대기...")
            while not self.place_client.server_is_ready():
                await asyncio.sleep(0.1)
            
            self.get_logger().info(">> [Task 3] 호출")
            await self.call_action(self.place_client, PlaceItem.Goal(is_corrected=is_match))
            
        self.get_logger().info("==== 모든 작업 종료 ====")

    async def call_action(self, client, goal):
        handle = await client.send_goal_async(goal, feedback_callback=self.fb_cb)
        if not handle.accepted: return None
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
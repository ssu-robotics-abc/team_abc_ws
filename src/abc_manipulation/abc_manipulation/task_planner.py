import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from abc_interfaces.msg import PurchaseItem
from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.cb_group = ReentrantCallbackGroup()

        # 클라이언트 설정 (돌려볼 로직 주석처리 하면서 테스트)
        self.pick_client = ActionClient(self, PickItem, 'pick_item', callback_group=self.cb_group)
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode', callback_group=self.cb_group)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=self.cb_group)

        self.get_logger().info("Task Planner Node Started.")
        self.create_timer(1.0, self.run_sequence, callback_group=self.cb_group)
        self.is_running = False

    async def run_sequence(self):
        if self.is_running: return
        self.is_running = True

        # 테스트 데이터: 2345 (포카리스웨트) 한 개 주문
        items = [PurchaseItem(product_id=2345, quantity=1)]

        for item in items:
            self.get_logger().info(f"\n==== [처리 시작] ID: {item.product_id} ====")

            # 1. PickItem 노드 통신
            self.get_logger().info(">> Pick 서버 연결 대기...")
            if not self.pick_client.wait_for_server(timeout_sec=20):
                self.get_logger().error("Pick 서버가 오프라인입니다.")
                break
            
            self.get_logger().info(">> pick_item 노드 호출: 물체 인식 및 파지 명령")
            goal_msg = PickItem.Goal(product_id=item.product_id)
            
            # 결과 응답 대기
            res1 = await self.call_action(self.pick_client, goal_msg)
            
            if res1 and res1.success:
                self.get_logger().info("Pick_item 성공! 물체를 잡았습니다.")
                
                # =========================================================
                # 3. PlaceItem 노드 통신 (현재 테스트를 위해 주석 처리됨)
                # =========================================================
                # self.get_logger().info(">> Place 서버 연결 대기...")
                # if not self.place_client.wait_for_server(timeout_sec=20):
                #     self.get_logger().error("Place 서버가 오프라인입니다.")
                #     break
                # 
                # self.get_logger().info(">> place_item 노드 호출: 분류 이송 및 물체 놓기 명령")
                # 
                # # Scan 결과가 맞다고 가정하고(is_corrected=True) 판매대로 놓기 테스트
                # goal_msg_place = PlaceItem.Goal(is_corrected=True)
                # 
                # # 결과 응답 대기
                # res3 = await self.call_action(self.place_client, goal_msg_place)
                # 
                # if res3 and res3.success:
                #     self.get_logger().info("Place_item 성공! 물체를 지정된 장소에 놓았습니다.")
                # else:
                #     self.get_logger().error("Place_item 실패")
                #     break
                # =========================================================

            else:
                self.get_logger().error("Pick_item 실패")
            break

        self.get_logger().info("==== 모든 테스트 종료 ====")

    async def call_action(self, client, goal):
        handle = await client.send_goal_async(goal, feedback_callback=self.fb_cb)
        if not handle.accepted: return None
        result = await handle.get_result_async()
        return result.result

    def fb_cb(self, msg):
        self.get_logger().info(f" [feedback 내용] {msg.feedback.state}")

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
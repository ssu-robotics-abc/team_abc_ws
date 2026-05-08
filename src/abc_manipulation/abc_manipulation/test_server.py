import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time

from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class DummyTaskServers(Node):
    def __init__(self):
        super().__init__('test_server')
        cb_group = ReentrantCallbackGroup()

        self._pick_server = ActionServer(self, PickItem, 'pick_item', self.execute_PickItem, callback_group=cb_group)
        self._scan_server = ActionServer(self, ScanBarcode, 'scan_barcode', self.execute_ScanBarcode, callback_group=cb_group)
        self._place_server = ActionServer(self, PlaceItem, 'place_item', self.execute_PlaceItem, callback_group=cb_group)

        self.get_logger().info("테스트용 서버 [pick_item, scan_barcode, place_item] 가동...")

    def execute_PickItem(self, goal_handle):
        product_id = goal_handle.request.product_id
        self.get_logger().info(f"[PickItem] 요청 수신 (ID: {product_id})")
        
        feedback = PickItem.Feedback()
        feedback.state = "물체 집는 중..."
        goal_handle.publish_feedback(feedback)
        time.sleep(2.0) # 로봇 동작 시뮬레이션

        goal_handle.succeed()
        return PickItem.Result(success=True)

    def execute_ScanBarcode(self, goal_handle):
        product_id = goal_handle.request.product_id
        self.get_logger().info(f"[ScanBarcode] 요청 수신 (ID: {product_id})")
        
        feedback = ScanBarcode.Feedback()
        feedback.state = "바코드 스캔 중..."
        goal_handle.publish_feedback(feedback)
        time.sleep(2.0)

        goal_handle.succeed()
        # 1234일 때만 일치하는 것 테스트
        is_match = (product_id == 1234)
        return ScanBarcode.Result(success=True, is_corrected=is_match)

    def execute_PlaceItem(self, goal_handle):
        is_corrected = goal_handle.request.is_corrected
        self.get_logger().info(f"[PlaceItem] 요청 수신 (일치: {is_corrected})")
        
        feedback = PlaceItem.Feedback()
        feedback.state = "물체 내려놓는 중..."
        goal_handle.publish_feedback(feedback)
        time.sleep(2.0)

        goal_handle.succeed()
        return PlaceItem.Result(success=True)

def main(args=None):
    rclpy.init(args=args)
    node = DummyTaskServers()
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
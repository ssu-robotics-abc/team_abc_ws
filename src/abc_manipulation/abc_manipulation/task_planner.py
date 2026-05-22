import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from abc_interfaces.msg import DetectionArray
from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.cb_group = ReentrantCallbackGroup()

        # YOLO 처리 후 받아올 subscriber 설정
        self.sub_detections = self.create_subscription(
            DetectionArray,
            '/detections',
            self.detection_callback,
            10, #메시지 큐 size는 임의로 10으로 설정해놓음.
            callback_group=self.cb_group
        )

        # 클라이언트 설정
        self.pick_client = ActionClient(self, PickItem, 'pick_item', callback_group=self.cb_group)
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode', callback_group=self.cb_group)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=self.cb_group)

        self.get_logger().info("Task Planner Node Started.")
        self.is_running = False
        self.detections_arr = []
        self.create_timer(1.0, self.run_sequence, callback_group=self.cb_group)

    #YOLO에서 받아온 탐지 결과 대기열 처리
    def detection_callback(self, msg):
        if not self.is_running and not self.detections_arr:
            if len(msg.detections) > 0:
                self.detection_arr = msg.detections
                self.get_logger().info(f"새로운 주문 상품 수신: {len(msg.detections)}개 물품을 받아왔습니다.")

    async def run_sequence(self):
        if self.is_running or not self.detections_arr:
            return
        self.is_running = True

        items = self.detections_arr
        self.detections_arr = []

        for item in items:
            
            barcode = str(item.class_name).strip()
            cx = float(item.center_x)
            cy = float(item.center_y)
            w = float(item.width)
            h = float(item.height)

            self.get_logger().info(f"\n==== [처리 시작] ID: {barcode} ==== ")

            # 1. PickItem 노드 통신
            self.get_logger().info(">> Pick 서버 연결 대기...")
            if not self.pick_client.wait_for_server(timeout_sec=20):
                self.get_logger().error("Pick 서버가 오프라인입니다.")
                break
            
            self.get_logger().info(">> pick_item 노드 호출: 물체 인식 및 파지 명령")
            goal_msg = PickItem.Goal(
                center_x=cx,
                center_y=cy,
                width=w,
                height=h
            )
            
            # 결과 응답 대기
            res1 = await self.call_action(self.pick_client, goal_msg)
            
            if res1 and res1.success:
                self.get_logger().info("Pick_item 성공! 물체를 잡았습니다.")
                
                self.get_logger().info(" Scan 서버 연결 대기...")
                if not self.scan_client.wait_for_server(timeout_sec=20):
                    self.get_logger().error("Scan 서버가 오프라인입니다.")
                    break

                self.get_logger().info(">> scan_barcode 노드 호출: 바코드 스캔 및 검증")
                goal_msg_scan = ScanBarcode.Goal(product_id=barcode)

                res2 = await self.call_action(self.scan_client, goal_msg_scan)

                if res2 and res2.success:
                    # 스캔 일치 여부 확인
                    is_match = res2.is_corrected
                    match_result = "일치: 판매대" if is_match else "불일치: 반품대"

                    # 3. PlaceItem 노드 통신 
                    self.get_logger().info(">> Place 서버 연결 대기...")
                    if not self.place_client.wait_for_server(timeout_sec=20):
                        self.get_logger().error("Place 서버가 오프라인입니다.")
                        break
                    
                    self.get_logger().info(">> place_item 노드 호출: 분류 이송 및 물체 놓기 명령")
                     
                    # Scan 결과에 따른 물품 이동
                    goal_msg_place = PlaceItem.Goal(is_corrected=is_match)
                     
                    # 결과 응답 대기
                    res3 = await self.call_action(self.place_client, goal_msg_place)
                     
                    if res3 and res3.success:
                        self.get_logger().info("Place_item 성공! 물체를 지정된 장소에 놓았습니다.")
                    else:
                        self.get_logger().error("Place_item 실패")
                        break
                else:
                    self.get_logger().error("Scan_barcode 실패")
                    break
            else:
                self.get_logger().error("Pick_item 실패")
                break

        self.get_logger().info("==== 모든 테스트 종료 ====")

        self.is_running = False

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
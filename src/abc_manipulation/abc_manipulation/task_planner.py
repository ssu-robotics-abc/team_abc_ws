import rclpy
import requests
import os
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from abc_interfaces.action import PickItem, ScanBarcode, PlaceItem
from abc_interfaces.srv import UserRequest, RequestItem, ResponseItem

from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        package_share_directory = get_package_share_directory('abc_manipulation')

        env_path = os.path.join(package_share_directory, 'config', '.env')

        if os.path.exists(env_path):
            load_dotenv(env_path)
            self.get_logger().info(".env 파일 로드 성공")
        else:
            self.get_logger().error(f".env 파일 로드 실패: {env_path}")

        self.web_server_url = os.getenv('WEB_SERVER_URL', 'https://ssu-abc-store-kiosk.ssammwu.info')
        self.get_logger().info(f"설정된 웹 서버 URL: {self.web_server_url}")

        self.cb_group = ReentrantCallbackGroup()

        self.vlm_srv = self.create_service(
            UserRequest,
            '/vlm_request',
            self.vlm_request_callback,
            callback_group=self.cb_group
        )

        self.yolo_request_client = self.create_client(
            RequestItem,
            '/request_item',
            callback_group=self.cb_group
        )

        self.yolo_response_srv = self.create_service(
            ResponseItem,
            '/response_item',
            self.yolo_response_callback,
            callback_group=self.cb_group
        )

        # 클라이언트 설정
        self.pick_client = ActionClient(self, PickItem, 'pick_item', callback_group=self.cb_group)
        self.scan_client = ActionClient(self, ScanBarcode, 'scan_barcode', callback_group=self.cb_group)
        self.place_client = ActionClient(self, PlaceItem, 'place_item', callback_group=self.cb_group)

        self.get_logger().info("Task Planner Node Started.")
        self.is_running = False
        self.request_queue = []
        self.detection_queue = []
        self.pending_item_name = None
        self.waiting_yolo_response = False
        self.create_timer(1.0, self.run_sequence, callback_group=self.cb_group)

    def vlm_request_callback(self, request, response):
        if len(request.class_name) == 0:
            response.success = False
            response.message = "요청된 상품이 없습니다."
            self.get_logger().error(response.message)
            return response

        if len(request.class_name) != len(request.iteration):
            response.success = False
            response.message = (
                f"class_name 개수({len(request.class_name)})와 "
                f"iteration 개수({len(request.iteration)})가 다릅니다."
            )
            self.get_logger().error(response.message)
            return response

        added_items = []
        post_payload = []

        for class_name, count in zip(request.class_name, request.iteration):
            item_name = str(class_name).strip()
            item_count = int(count)

            if not item_name:
                continue
            if item_count <= 0:
                response.success = False
                response.message = f"{item_name} 요청 개수는 1 이상이어야 합니다."
                self.get_logger().error(response.message)
                return response

            post_payload.append({
                "id": item_name,
                "amount": item_count
            })

            for _ in range(item_count):
                self.request_queue.append(item_name)
                added_items.append(item_name)

        if not added_items:
            response.success = False
            response.message = "유효한 상품 요청이 없습니다."
            self.get_logger().error(response.message)
            return response

        try:
            self.get_logger().info(f"상품 정보 서버 POST 준비: {post_payload}")

            res = requests.post(self.web_server_url+"/api/v1/cart", json=post_payload, timeout=5.0)

            if res.status_code in [200, 201]:
                self.get_logger().info(f"POST 서버 전송 성공")
            else:
                self.get_logger().error("POST 서버 전송 실패")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"서버 POST 통신 에러 {e}")


        response.success = True
        response.message = f"상품 요청 {len(added_items)}개 접수"
        self.get_logger().info(
            f"{response.message}: {added_items}, "
            f"대기열 {len(self.request_queue)}개"
        )
        return response

    def yolo_response_callback(self, request, response):
        if not self.waiting_yolo_response:
            response.success = False
            response.message = "YOLO 위치 응답을 기다리는 요청이 없습니다."
            self.get_logger().warn(response.message)
            return response

        if self.pending_item_name and request.class_name != self.pending_item_name:
            response.success = False
            response.message = (
                f"요청 상품({self.pending_item_name})과 "
                f"응답 상품({request.class_name})이 다릅니다."
            )
            self.get_logger().warn(response.message)
            return response

        self.detection_queue.append(request)
        self.pending_item_name = None
        self.waiting_yolo_response = False

        response.success = True
        response.message = f"{request.class_name} 위치 수신 완료"
        self.get_logger().info(response.message)
        return response

    async def run_sequence(self):
        if self.is_running:
            return

        if not self.detection_queue:
            self.request_next_item_location()
            return

        self.is_running = True

        item = self.detection_queue.pop(0)

        try:
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
                return
            
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
                
                self.get_logger().info("Scan 서버 연결 대기...")
                if not self.scan_client.wait_for_server(timeout_sec=20):
                    self.get_logger().error("Scan 서버가 오프라인입니다.")
                    return

                self.get_logger().info(">> scan_barcode 노드 호출: 바코드 스캔 및 검증")
                goal_msg_scan = ScanBarcode.Goal(
                    product_id=barcode,
                    height=h
                    )

                res2 = await self.call_action(self.scan_client, goal_msg_scan)

                if res2 and res2.success:
                    # 스캔 일치 여부 확인
                    is_match = res2.is_corrected

                    # 3. PlaceItem 노드 통신 
                    self.get_logger().info(">> Place 서버 연결 대기...")
                    if not self.place_client.wait_for_server(timeout_sec=20):
                        self.get_logger().error("Place 서버가 오프라인입니다.")
                        return
                    
                    self.get_logger().info(">> place_item 노드 호출: 분류 이송 및 물체 놓기 명령")
                     
                    # Scan 결과에 따른 물품 이동
                    goal_msg_place = PlaceItem.Goal(is_corrected=is_match)
                     
                    # 결과 응답 대기
                    res3 = await self.call_action(self.place_client, goal_msg_place)
                     
                    if res3 and res3.success:
                        self.get_logger().info("Place_item 성공! 물체를 지정된 장소에 놓았습니다.")
                    else:
                        self.get_logger().error("Place_item 실패")
                        return
                else:
                    self.get_logger().error("Scan_barcode 실패")
                    return
            else:
                self.get_logger().error("Pick_item 실패")
                return

            self.get_logger().info("==== 상품 처리 완료 ====")
        finally:
            self.is_running = False

    def request_next_item_location(self):
        if self.waiting_yolo_response or not self.request_queue:
            return

        item_name = self.request_queue.pop(0)
        self.get_logger().info(f">> YOLO 위치 요청: {item_name}")

        if not self.yolo_request_client.wait_for_service(timeout_sec=5):
            self.get_logger().error("YOLO 요청 서비스(/request_item)가 준비되지 않았습니다.")
            self.request_queue.insert(0, item_name)
            return

        self.pending_item_name = item_name
        self.waiting_yolo_response = True

        req = RequestItem.Request()
        req.class_name = item_name

        future = self.yolo_request_client.call_async(req)
        future.add_done_callback(self.yolo_request_done_callback)

    def yolo_request_done_callback(self, future):
        try:
            yolo_result = future.result()
            if not yolo_result.success:
                self.get_logger().error(f"YOLO 요청 실패: {yolo_result.message}")
                self.pending_item_name = None
                self.waiting_yolo_response = False
                return

            self.get_logger().info(f"YOLO 요청 응답: {yolo_result.message}")
        except Exception as e:
            self.get_logger().error(f"YOLO 위치 요청 중 오류: {e}")
            self.pending_item_name = None
            self.waiting_yolo_response = False

    async def call_action(self, client, goal):
        handle = await client.send_goal_async(goal, feedback_callback=self.fb_cb)
        if not handle.accepted:
            return None
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

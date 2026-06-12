import rclpy
import requests
import os
import uuid
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

        self.web_server_url = os.getenv('WEB_SERVER_URL')
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
        self.pending_item_uuid = None
        self.waiting_yolo_response = False
        self.create_timer(1.0, self.run_sequence, callback_group=self.cb_group)

    def send_web_request(self, endpoint: str, payload: dict, method: str = 'POST'):
        url = f"{self.web_server_url}{endpoint}"
        try:
            self.get_logger().info(f"웹 서버 {method} 요청: {url} | Payload: {payload}")
            
            if method.upper() == 'PATCH':
                res = requests.patch(url, json=payload, timeout=5.0)
            else:
                res = requests.post(url, json=payload, timeout=5.0)
                
            # 성공 응답 코드 처리 (200 OK, 201 Created, 204 No Content)
            if res.status_code in [200, 201, 204]:
                self.get_logger().info(f"웹 서버 {method} 전송 성공")
                return True
            else:
                self.get_logger().error(f"웹 서버 {method} 전송 실패 (상태 코드: {res.status_code})")
                return False
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"서버 {method} 통신 에러 {e}")
            return False

    #주문 후 상품 리스트 전송
    def send_cart_list(self, items_data: list):
        payload = []
        for item_name, item_uuid in items_data:
            payload.append({
                "uuid": item_uuid,
                "id": item_name
            })
        self.send_web_request("/api/v1/cart", payload, method='POST')

    def send_item_status(self, item_uuid: str, is_match: bool):
        payload = {
            "uuid": item_uuid,
            "correct": is_match
        }
        self.send_web_request("/api/v1/cart", payload, method='PATCH')

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
        items_to_send = []

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

            for _ in range(item_count):
                item_uuid = str(uuid.uuid4())

                self.request_queue.append((item_name, item_uuid))

                items_to_send.append((item_name, item_uuid))
                
                added_items.append(item_name)

        if not added_items:
            response.success = False
            response.message = "유효한 상품 요청이 없습니다."
            self.get_logger().error(response.message)
            return response

        self.send_cart_list(items_to_send)

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

        self.detection_queue.append((request, self.pending_item_uuid))
        self.pending_item_name = None
        self.pending_item_uuid = None
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

        # 상품 목록에서 pop
        item_data = self.detection_queue.pop(0)
        item = item_data[0]
        item_uuid = item_data[1]
        barcode = str(item.class_name).strip()
        is_match = False

        try:
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
                
                if not self.scan_client.wait_for_server(timeout_sec=20):
                    self.get_logger().error("Scan 서버가 오프라인입니다.")
                    return

                self.get_logger().info(">> scan_barcode 노드 호출: 바코드 스캔 및 검증")
                goal_msg_scan = ScanBarcode.Goal(product_id=barcode)

                res2 = await self.call_action(self.scan_client, goal_msg_scan)

                if res2 and res2.success:
                    is_match = res2.is_corrected
                else:
                    self.get_logger().warning("Scan_barcode 실패: 미검증 상품으로 처리하여 반품대로 이동합니다.")
                    is_match = False

                # 3. PlaceItem 노드 통신
                self.get_logger().info(">> Place 서버 연결 대기...")
                if not self.place_client.wait_for_server(timeout_sec=20):
                    self.get_logger().error("Place 서버가 오프라인입니다.")
                    return

                self.get_logger().info(">> place_item 노드 호출: 분류 이송 및 물체 놓기 명령")

                # Scan 결과에 따른 물품 이동: True는 판매대, False는 반품대
                goal_msg_place = PlaceItem.Goal(is_corrected=is_match)

                # 결과 응답 대기
                res3 = await self.call_action(self.place_client, goal_msg_place)

                if res3 and res3.success:
                    self.get_logger().info("Place_item 성공! 물체를 지정된 장소에 놓았습니다.")
                else:
                    self.get_logger().error("Place_item 실패")
                    return
            else:
                self.get_logger().error("Pick_item 실패")
                return

            self.get_logger().info("==== 상품 처리 완료 ====")
        finally:
            self.send_item_status(item_uuid, is_match)
            self.is_running = False

    def request_next_item_location(self):
        if self.waiting_yolo_response or not self.request_queue:
            return

        item_data = self.request_queue.pop(0)
        item_name = item_data[0]
        item_uuid = item_data[1]

        self.get_logger().info(f">> YOLO 위치 요청: {item_name}")

        if not self.yolo_request_client.wait_for_service(timeout_sec=5):
            self.get_logger().error("YOLO 요청 서비스(/request_item)가 준비되지 않았습니다.")
            self.request_queue.insert(0, item_data)
            return

        self.pending_item_name = item_name
        self.pending_item_uuid = item_uuid
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
                self.pending_item_uuid = None
                self.waiting_yolo_response = False
                return

            self.get_logger().info(f"YOLO 요청 응답: {yolo_result.message}")
        except Exception as e:
            self.get_logger().error(f"YOLO 위치 요청 중 오류: {e}")
            self.pending_item_name = None
            self.pending_item_uuid = None
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

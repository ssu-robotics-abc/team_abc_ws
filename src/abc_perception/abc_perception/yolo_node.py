import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from geometry_msgs.msg import Vector3
from abc_interfaces.srv import RequestItem, ResponseItem

import cv2
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        self.bridge = CvBridge()

        self.latest_frame = None
        self.latest_header = None

        # GUI 환경에서만 True 권장
        self.debug_view = True

        package_share_dir = get_package_share_directory("abc_perception")
        model_path = os.path.join(package_share_dir, "models", "best.pt")
        self.model = YOLO(model_path)
        self.class_name_map = {
            0: "8801062518210", #Kancho
            1: "8801117536411", #chocopie
            2: "8801062012725", #pepero_almond
            3: "unknown",       #pepero_original
            4: "8801056248703", #pepsi
            5: "8801097150010", #pocari_sweat
            6: "8801121768440", #soy_milk
        }
        self.class_name= {"Kancho", "chocopie", "pepero_almond", "pepero_original", "pepsi", "pocari_sweat", "soy_milk"}

        self.request_service_name = (
            self.declare_parameter("request_service_name", "/request_item")
            .get_parameter_value()
            .string_value
        )
        self.response_service_name = (
            self.declare_parameter("response_service_name", "/response_item")
            .get_parameter_value()
            .string_value
        )

        self.request_srv = self.create_service(
            RequestItem,
            self.request_service_name,
            self.handle_item_request
        )

        self.response_cli = self.create_client(
            ResponseItem,
            self.response_service_name
        )

        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.tracking_pub = self.create_publisher(Vector3, "/yolo_tracking", 10)
        self.tracking_class = None

        self.get_logger().info(
            "YOLO Node 구동 중 "
            f"(request: {self.request_service_name}, "
            f"response: {self.response_service_name})"
        )

    def get_class_name(self, cls_id):
        if cls_id in self.class_name_map:
            return self.class_name_map[cls_id]

        return self.model.names.get(cls_id, str(cls_id))

    def handle_item_request(self, request, response):
        class_name = request.class_name.strip()
        known_classes = set(self.class_name_map.values())

        if not class_name:
            response.success = False
            response.message = "에러: 요청된 클래스가 없습니다."
            self.get_logger().error(response.message)
            return response

        if class_name not in known_classes:
            response.success = False
            response.message = (
                f"에러: '{class_name}'은(는) 모델에 없는 클래스입니다. "
                f"사용 가능 클래스: {sorted(known_classes)}"
            )
            self.get_logger().error(response.message)
            return response

        if self.latest_frame is None:
            response.success = False
            response.message = "에러: 아직 카메라 이미지가 수신되지 않았습니다."
            self.get_logger().error(response.message)
            return response

        detection = self.detect_requested_item(class_name)
        if detection is None:
            response.success = False
            response.message = f"'{class_name}' 탐지 실패"
            self.get_logger().warn(response.message)
            return response

        if not self.send_item_response(detection):
            response.success = False
            response.message = (
                f"'{class_name}' 위치는 탐지했지만 "
                f"{self.response_service_name} 서비스 전송에 실패했습니다."
            )
            return response

        self.tracking_class = class_name
        response.success = True
        response.message = f"'{class_name}' 위치 응답 전송 완료"
        self.get_logger().info(response.message)
        return response

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        self.latest_frame = frame
        self.latest_header = msg.header

        if self.tracking_class is not None:
            det = self.detect_requested_item(self.tracking_class)
            if det is not None:
                tracking_msg = Vector3()
                tracking_msg.x = det["center_x"]
                tracking_msg.y = det["center_y"]
                tracking_msg.z = det["confidence"]
                self.tracking_pub.publish(tracking_msg)

        if self.debug_view:
            results = self.model(frame, conf=0.5, verbose=False)
            result = results[0]
            debug_frame = self.draw_debug_frame(frame, result)
            cv2.imshow("YOLO Real-time Debug", debug_frame)
            cv2.waitKey(1)

    def detect_requested_item(self, requested_class_name):
        results = self.model(self.latest_frame, conf=0.5, verbose=False)
        result = results[0]

        best_detection = None
        best_confidence = -1.0

        if result.boxes is None:
            return None

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = str(self.get_class_name(cls_id))

            if class_name != requested_class_name:
                continue

            class_name = self.get_class_name(cls_id)
            
            confidence = float(box.conf[0])
            if confidence <= best_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best_confidence = confidence
            best_detection = {
                "class_name": class_name,
                "confidence": confidence,
                "center_x": float((x1 + x2) / 2.0),
                "center_y": float((y1 + y2) / 2.0),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }

        return best_detection

    def send_item_response(self, detection):
        if not self.response_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"서비스 서버({self.response_service_name})가 준비되지 않았습니다."
            )
            return False

        req = ResponseItem.Request()
        req.class_name = detection["class_name"]
        req.confidence = detection["confidence"]
        req.center_x = detection["center_x"]
        req.center_y = detection["center_y"]
        req.width = detection["width"]
        req.height = detection["height"]

        future = self.response_cli.call_async(req)
        future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(
                    f"{self.response_service_name} 응답 성공: {response.message}"
                )
            else:
                self.get_logger().warn(
                    f"{self.response_service_name} 응답 실패: {response.message}"
                )
        except Exception as e:
            self.get_logger().error(f"{self.response_service_name} 호출 실패: {e}")

    def draw_debug_frame(self, frame, result):
        debug_frame = frame.copy()

        if result.boxes is None:
            return debug_frame

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = str(self.get_class_name(cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = f"{class_name} {confidence:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            text_w, text_h = text_size
            text_y = max(y1, text_h + baseline + 4)

            cv2.rectangle(
                debug_frame,
                (x1, text_y - text_h - baseline - 4),
                (x1 + text_w + 4, text_y),
                color,
                -1
            )
            cv2.putText(
                debug_frame,
                label,
                (x1 + 2, text_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return debug_frame


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
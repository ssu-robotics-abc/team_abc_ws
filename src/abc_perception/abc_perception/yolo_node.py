import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from abc_interfaces.msg import DetectedObject, DetectionArray
from abc_interfaces.srv import UserRequest

import cv2
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        self.bridge = CvBridge()

        self.target_class_names = []
        self.target_counts = []
        self.is_active = True

        # True면 한 번 detection publish 후 멈춤
        # 계속 탐지하고 싶으면 False로 바꾸면 됨
        self.publish_once = False

        # GUI 환경에서만 True 권장
        self.debug_view = True

        package_share_dir = get_package_share_directory("abc_perception")
        model_path = os.path.join(package_share_dir, "models", "best.pt")
        self.model = YOLO(model_path)

        self.srv = self.create_service(
            UserRequest,
            "/vlm_request",
            self.handle_vlm_request
        )

        self.pub = self.create_publisher(
            DetectionArray,
            "/detections",
            10
        )

        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info("YOLO Node 구동 중")

    def handle_vlm_request(self, request, response):
        known_classes = list(self.model.names.values())

        if len(request.class_name) == 0:
            self.is_active = False
            response.success = False
            response.message = "에러: 요청된 클래스가 없습니다."
            self.get_logger().error(response.message)
            return response

        if len(request.class_name) != len(request.iteration):
            self.is_active = False
            response.success = False
            response.message = (
                f"에러: class_name 개수({len(request.class_name)})와 "
                f"iteration 개수({len(request.iteration)})가 다릅니다."
            )
            self.get_logger().error(response.message)
            return response

        for class_name, count in zip(request.class_name, request.iteration):
            if count <= 0:
                self.is_active = False
                response.success = False
                response.message = f"에러: {class_name}의 iteration은 1 이상이어야 합니다."
                self.get_logger().error(response.message)
                return response

            if class_name not in known_classes:
                self.is_active = False
                response.success = False
                response.message = (
                    f"에러: '{class_name}'은(는) 모델에 없는 클래스입니다. "
                    f"사용 가능 클래스: {known_classes}"
                )
                self.get_logger().error(response.message)
                return response

        self.target_class_names = list(request.class_name)
        self.target_counts = list(request.iteration)
        self.is_active = True

        response.success = True
        response.message = (
            f"탐지 시작: {self.target_class_names}, "
            f"요청 개수: {self.target_counts}"
        )
        self.get_logger().info(response.message)

        return response

    def image_callback(self, msg):
        if not self.is_active:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")
            return

        results = self.model(frame, conf=0.25, verbose=False)
        result = results[0]

        det_array = DetectionArray()
        det_array.header = msg.header

        found_counts = {name: 0 for name in self.target_class_names}
        target_limits = dict(zip(self.target_class_names, self.target_counts))

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = str(self.model.names[cls_id])

                if class_name not in target_limits:
                    continue

                if found_counts[class_name] >= target_limits[class_name]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det = DetectedObject()
                det.class_name = class_name
                det.confidence = float(box.conf[0])
                det.center_x = float((x1 + x2) / 2.0)
                det.center_y = float((y1 + y2) / 2.0)
                det.width = float(x2 - x1)
                det.height = float(y2 - y1)

                det_array.detections.append(det)
                found_counts[class_name] += 1

        if det_array.detections:
            self.pub.publish(det_array)

            self.get_logger().info(
                f"[{self.target_class_names}] 탐지 결과 "
                f"{len(det_array.detections)}개 송신"
            )

            if self.publish_once:
                self.is_active = False
                self.get_logger().info("1회 탐지 완료. YOLO 탐지 비활성화.")

        if self.debug_view:
            debug_frame = result.plot()
            cv2.imshow("YOLO Real-time Debug", debug_frame)
            cv2.waitKey(1)


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

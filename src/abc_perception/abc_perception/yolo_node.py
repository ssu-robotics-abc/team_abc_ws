import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory # 필수
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from abc_interfaces.msg import DetectedObject, DetectionArray
import cv2
from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        self.bridge = CvBridge()

        # [수정] 절대 경로를 지우고, 패키지 설치 경로에서 모델을 찾습니다.
        package_share_dir = get_package_share_directory('abc_perception')
        model_path = os.path.join(package_share_dir, 'models', 'best.pt')

        self.get_logger().info(f"Model path: {model_path}")
        self.model = YOLO(model_path)

        # [수정] pick_item.py와 토픽 이름을 일치시킵니다.
        self.pub = self.create_publisher(DetectionArray, "/detections", 10)

        self.sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model(frame, conf=0.25, verbose=False)
        result = results[0]

        det_array = DetectionArray()
        det_array.header = msg.header

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det = DetectedObject()
                det.class_name = str(self.model.names[cls_id])
                det.confidence = conf
                det.center_x = (x1 + x2) / 2.0
                det.center_y = (y1 + y2) / 2.0
                det.width = float(x2 - x1)
                det.height = float(y2 - y1)

                # msg 정의에 따라 'detections' 혹은 'objects' 필드명을 확인하세요.
                det_array.detections.append(det) 

        self.pub.publish(det_array)
        
        # 시각화 (디버깅용)
        annotated = result.plot()
        cv2.imshow("YOLO RealSense", annotated)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
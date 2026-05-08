import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from abc_interfaces.msg import DetectedObject, DetectionArray

import cv2
from ultralytics import YOLO
from pathlib import Path


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        self.bridge = CvBridge()

        model_path = Path("/home/ssu/team_abc_ws/src/abc_perception/models/best.pt")
        self.get_logger().info(f"model path: {model_path}")
        self.get_logger().info(f"model exists: {model_path.exists()}")

        self.model = YOLO(str(model_path))

        self.pub = self.create_publisher(
            DetectionArray,
            "/perception/detections_2d",
            10
        )

        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info("YOLO node started")
        self.get_logger().info("Subscribing: /camera/camera/color/image_raw")
        self.get_logger().info("Publishing: /perception/detections_2d")

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
                class_name = self.model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                det = DetectedObject()
                det.class_name = str(class_name)
                det.confidence = float(conf)
                det.center_x = float(cx)
                det.center_y = float(cy)
                det.width = float(w)
                det.height = float(h)

                det_array.detections.append(det)

        self.pub.publish(det_array)

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
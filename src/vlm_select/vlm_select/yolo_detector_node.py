#!/usr/bin/env python3

import cv2
import rclpy
import json
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")
        self.bridge = CvBridge()

        # 모델 로드
        model_path = Path(get_package_share_directory("dsr_practice")) / "best.pt"
        self.get_logger().info(f"YOLO 모델 로드 중: {model_path}")
        try:
            self.yolo = YOLO(str(model_path))
        except Exception as e:
            self.get_logger().error(f"YOLO 모델 로드 실패: {e}")
            raise RuntimeError("YOLO 모델 로드 실패") from e

        # Subscriber (카메라 원본 영상)
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 10)

        # Publishers
        self.img_pub = self.create_publisher(Image, "/yolo/annotated_image", 10)
        self.det_pub = self.create_publisher(String, "/yolo/detections", 10)
        
        self.get_logger().info("YOLO Detector 노드 가동 완료.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 오류: {e}")
            return

        # YOLO 추론
        results = self.yolo(frame, verbose=False)
        boxes = results[0].boxes
        annotated_frame = frame.copy()
        
        detection_list = []

        # 바운딩 박스 그리기 및 데이터 추출
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            cls_name = self.yolo.names[cls_idx]
            conf = float(box.conf[0])

            # 리스트에 추가 (VLM 노드 전달용)
            detection_list.append({
                "class": cls_name,
                "conf": conf,
                "box": [x1, y1, x2, y2]
            })

            # 시각화 박스 그리기
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + max(150, len(label)*12), y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 결과 발행 1: 주석 처리된 이미지 (rqt_image_view용)
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.img_pub.publish(img_msg)

        # 결과 발행 2: 객체 데이터 (JSON String 포맷)
        det_msg = String()
        det_msg.data = json.dumps(detection_list)
        self.det_pub.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
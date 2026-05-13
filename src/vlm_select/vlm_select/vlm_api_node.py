#!/usr/bin/env python3

import base64
import requests
import json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import google.generativeai as genai

# 설정 파라미터
TARGET_CLASSES = [
    "Kancho", "pepero_original", "pepsi", "pocarisweat",
    "soy_milk", "chocopie", "pepero_almond"
]

KEYWORD_MAP = {
    "칸초": "Kancho", "칸쵸": "Kancho",
    "오리지널 빼빼로": "pepero_original", "빼빼로": "pepero_original",
    "펩시": "pepsi", "콜라": "pepsi",
    "포카리스웨트": "pocarisweat", "포카리": "pocarisweat",
    "소이밀크": "soy_milk", "두유": "soy_milk",
    "초코파이": "chocopie",
    "아몬드 빼빼로": "pepero_almond", "아몬드": "pepero_almond"
}


def get_target_from_gemini(cv_image, user_command):
    # API 키 설정
    api_key = "AIzaSyC3CZIsFgJu3oBbYYj8EViLgxE46GNhQmg"
    genai.configure(api_key=api_key)
    
    # 모델 로드
    model = genai.GenerativeModel('gemini-3-flash-preview')

    # OpenCV 이미지를 JPEG 바이트로 변환
    ok, buf = cv2.imencode('.jpg', cv_image)
    if not ok:
        print("[VLM 오류] 이미지 인코딩 실패")
        return None
    image_bytes = buf.tobytes()

    prompt = (
        "너는 로봇의 시각 판단 모듈이다.\n"
        "사용자의 명령에 알맞은 물체를 카메라 원본 이미지에서 찾고,\n"
        f"다음 리스트에서 단 하나만 정확히 골라 출력해라.\n"
        "반드시 영문 클래스명만 출력할 것 (마침표 등 제외).\n"
        f"리스트: {TARGET_CLASSES}\n"
        f"사용자 명령: {user_command}"
    )

    try:
        # SDK를 이용한 콘텐츠 생성 호출
        response = model.generate_content(
            contents=[
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            ),
            # 네트워크 타임아웃 15초 설정
            request_options={"timeout": 15.0} 
        )
        
        # SDK가 제공하는 텍스트 추출 속성 사용
        if not response.parts:
            print("[VLM 오류] 텍스트가 생성되지 않았습니다.")
            return None
            
        text = response.text.strip()
        
        if text not in TARGET_CLASSES:
            print(f"[VLM 경고] 리스트에 없는 값이 출력되었습니다: '{text}'")
            
        return text

    except Exception as e:
        print(f"\n[VLM SDK 통신 실패] {e}\n")
        return None


class VlmLogicNode(Node):
    def __init__(self):
        super().__init__("vlm_logic_node")
        self.bridge = CvBridge()
        
        # 상태 저장을 원본 영상과 YOLO 영상으로 분리
        self.latest_raw_image = None
        self.latest_yolo_image = None
        self.latest_detections = []

        # Subscribers
        # 1. VLM에 넘길 원본 영상 구독 추가
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.raw_image_callback, 10)
        # 2. 최종 화면에 보여줄 YOLO 주석 영상
        self.create_subscription(Image, "/yolo/annotated_image", self.yolo_image_callback, 10)
        
        self.create_subscription(String, "/yolo/detections", self.detections_callback, 10)
        # self.create_subscription(String, "/vlm_user_command", self.command_callback, 10)
        self.create_subscription(String, "/stt_result", self.command_callback, 10)
        
        # Publisher (Pick을 위해 확정된 타겟 시각화 이미지)
        self.result_pub = self.create_publisher(Image, "/vlm/target_image", 10)

        self.get_logger().info("VLM Logic 노드 가동 완료. 원본 이미지 연동 됨. 명령 대기 중...")

    # 원본 이미지 콜백
    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    # YOLO 이미지 콜백
    def yolo_image_callback(self, msg):
        try:
            self.latest_yolo_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def detections_callback(self, msg):
        try:
            self.latest_detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("JSON 파싱 에러")

    def command_callback(self, msg):
        user_command = msg.data.strip()
        self.get_logger().info(f"\n[{user_command}] 명령 수신됨. 분석 시작...")

        # 원본 영상과 YOLO 영상이 모두 준비되었는지 확인
        if self.latest_raw_image is None or self.latest_yolo_image is None or not self.latest_detections:
            self.get_logger().error("아직 카메라 원본 영상이나 YOLO 탐지 결과가 들어오지 않았습니다.")
            return

        # 1단계: Fast-path (키워드 매칭)
        target_word = None
        for kr_word in sorted(KEYWORD_MAP, key=len, reverse=True):
            if kr_word in user_command:
                target_word = KEYWORD_MAP[kr_word]
                self.get_logger().info(f"▶ 키워드 맵핑 성공: '{target_word}'")
                break

        # 2단계: VLM Fallback
        if target_word is None:
            self.get_logger().info("▶ 키워드 없음. Gemini API로 문맥 분석 요청 중...")
            # YOLO 이미지가 아닌 순수 카메라 원본 이미지를 VLM에 전달합니다.
            target_word = get_target_from_gemini(self.latest_raw_image, user_command)
            self.get_logger().info(f"▶ Gemini 응답: '{target_word}'")

        if target_word not in TARGET_CLASSES:
            self.get_logger().error("타겟 확정 실패.")
            return

        # 3단계: 화면에서 타겟 좌표 찾기 및 시각화
        found = False
        # 결과 화면은 보기 편하게 YOLO 박스가 그려진 이미지를 베이스로 사용합니다.
        result_img = self.latest_yolo_image.copy()

        for det in self.latest_detections:
            if det['class'] == target_word:
                x1, y1, x2, y2 = det['box']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                self.get_logger().info(f"★ 최종 타겟 픽업 위치: {target_word} | 중심 좌표: ({cx}, {cy})")

                # 타겟 강조 (빨간색)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.circle(result_img, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"PICK TARGET: {target_word}", (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # 결과 이미지 토픽 발행
                result_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
                self.result_pub.publish(result_msg)
                found = True
                break

        if not found:
            self.get_logger().error(f"'{target_word}'이(가) 화면에서 탐지되지 않았습니다.")


def main(args=None):
    rclpy.init(args=args)
    node = VlmLogicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
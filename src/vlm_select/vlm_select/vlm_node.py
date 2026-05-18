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
from dotenv import load_dotenv
import os
import google.generativeai as genai

from abc_interfaces.srv import UserRequest

# 설정 파라미터
TARGET_CLASSES = [
    "Kancho", "pepero_original", "pepsi", "pocarisweat",
    "soy_milk", "chocopie", "pepero_almond"
]

def get_target_from_gemini(cv_image, user_command):
    # API 키 설정
    api_key = os.getenv("GEMINI_API_KEY")
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
        f"다음 리스트에서 단 하나만 정확히 골라 사용자가 요구하는 개수와 함께 출력해라.\n"
        "만약 요구하는 개수가 명시되어 있지 않다면 0 취급해라.\n"
        "출력 형태는 다음과 같다:\n"
        "클래스명 요구개수\n"
        "반드시 출력 형태를 맞추고 영문 클래스명만 출력할 것 (마침표 등 제외).\n"
        f"리스트: {TARGET_CLASSES}\n"
        f"사용자 명령: {user_command}"
    )

    try:
        response = model.generate_content(
            contents=[
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            ),
            request_options={"timeout": 15.0} 
        )
        
        if not response.parts:
            print("[VLM 오류] 텍스트가 생성되지 않았습니다.")
            return None
            
        text = response.text.strip()
        
        # '클래스명 개수' 형태로 오므로, 첫 번째 단어(클래스명)만 추출해서 리스트에 있는지 검증
        parsed_class = text.split()[0] if text else ""
        if parsed_class not in TARGET_CLASSES:
            print(f"[VLM 경고] 리스트에 없는 값이 출력되었습니다: '{parsed_class}'")
            
        return text

    except Exception as e:
        print(f"\n[VLM SDK 통신 실패] {e}\n")
        return None


class VlmLogicNode(Node):
    def __init__(self):
        super().__init__("vlm_logic_node")
        self.bridge = CvBridge()
        self.latest_raw_image = None

        # Subscribers
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.raw_image_callback, 10)
        self.create_subscription(String, "/vlm_user_command", self.command_callback, 10)

        self.cli = self.create_client(UserRequest, "/vlm_request")

        self.get_logger().info("VLM Logic 노드 가동 완료. 원본 이미지 연동 됨. 명령 대기 중...")

    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def command_callback(self, msg):
        user_command = msg.data.strip()
        self.get_logger().info(f"\n[{user_command}] 명령 수신됨. 분석 시작...")

        if self.latest_raw_image is None:
            self.get_logger().error("아직 카메라 원본 영상이 들어오지 않았습니다.")
            return

        self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        target_word = get_target_from_gemini(self.latest_raw_image, user_command)
        self.get_logger().info(f"▶ Gemini 응답: '{target_word}'")
        
        if not target_word:
            self.get_logger().error("VLM 응답이 올바르지 않습니다.")
            return

        # "클래스명 요구개수" 문자열 파싱 (예: "snack 100")
        parts = target_word.split()
        if len(parts) >= 1:
            class_name = parts[0]
            # 두 번째 인자가 존재하고 숫자형태면 int로 변환, 아니면 기본값 0
            iteration = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        else:
            self.get_logger().error("출력 포맷이 맞지 않습니다.")
            return

        if class_name not in TARGET_CLASSES:
            self.get_logger().error(f"타겟 확정 실패: {class_name}은(는) 리스트에 없습니다.")
            return

        # 서비스 서버가 켜져 있는지 확인
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_request)가 준비되지 않았습니다.")
            return

        # 서비스 요청 데이터 세팅 및 전송
        self.get_logger().info(f"▶ 타겟 확정 완료: {class_name}, {iteration}개. 서비스 요청 전송 중...")
        req = UserRequest.Request()
        req.class_name = class_name
        req.iteration = iteration

        # 비동기 방식으로 서비스 요청
        future = self.cli.call_async(req)
        future.add_done_callback(self.response_callback)

    # 서비스 응답을 처리하는 콜백 함수
    def response_callback(self, future):
        try:
            response = future.result()
            # 서비스 응답에 따라 로그 출력 (response.success 등 서비스 정의에 맞게 수정 가능)
            self.get_logger().info(f"✅ 서비스 요청 완료. 서버 응답 수신됨.")
        except Exception as e:
            self.get_logger().error(f"❌ 서비스 호출 실패: {e}")


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

#!/usr/bin/env python3

import os
import json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import google.generativeai as genai
from dotenv import load_dotenv

# abc_interfaces 패키지의 UserRequest 서비스 임포트
from abc_interfaces.srv import UserRequest

# 설정 파라미터
TARGET_CLASSES = [
    "Kancho", "pepero_original", "pepsi", "pocarisweat",
    "soy_milk", "chocopie", "pepero_almond"
]

# .env 파일 로드 (환경 변수 적용)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

def get_target_from_gemini(cv_image, user_command):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        print("[오류] .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다!")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    ok, buf = cv2.imencode('.jpg', cv_image)
    if not ok:
        print("[VLM 오류] 이미지 인코딩 실패")
        return None
    image_bytes = buf.tobytes()

    # ==========================================
    # 1. 다중 물품 처리를 위한 프롬프트 수정
    # ==========================================
    prompt = (
        "너는 로봇의 시각 판단 모듈이다.\n"
        "사용자의 명령을 분석하여, 카메라 원본 이미지 내에서 조건에 맞는 물품들을 모두 찾아라.\n"
        f"반드시 다음 제공된 리스트 내의 영문 클래스명만 선택해야 하며, 리스트에 없는 물품은 철저히 무시해라.\n"
        f"리스트: {TARGET_CLASSES}\n\n"
        "개수가 명시되지 않은 경우 0으로 취급하라.\n"
        "출력은 반드시 아래 구조를 가진 JSON 배열(Array) 형식으로만 반환해야 한다.\n"
        "예시:\n"
        "[\n"
        "  {\"class_name\": \"pepero_original\", \"iteration\": 2},\n"
        "  {\"class_name\": \"pepsi\", \"iteration\": 1}\n"
        "]\n\n"
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
                # 2. JSON 포맷으로 강제 출력 설정
                response_mime_type="application/json", 
            ),
            request_options={"timeout": 15.0} 
        )
        
        if not response.parts:
            print("[VLM 오류] 텍스트가 생성되지 않았습니다.")
            return []
            
        text = response.text.strip()
        
        # 3. JSON 문자열을 파이썬 리스트(List of Dicts)로 파싱
        parsed_data = json.loads(text)
        
        # 클래스명이 리스트에 있는지 한 번 더 검증 (안전 장치)
        valid_targets = []
        for item in parsed_data:
            c_name = item.get("class_name", "")
            if c_name in TARGET_CLASSES:
                valid_targets.append(item)
            else:
                print(f"[VLM 경고] 리스트에 없는 값이 필터링 되었습니다: '{c_name}'")
                
        return valid_targets

    except json.JSONDecodeError as e:
        print(f"\n[VLM 파싱 실패] JSON 형식이 올바르지 않습니다: {e}\n응답 데이터: {text}\n")
        return []
    except Exception as e:
        print(f"\n[VLM SDK 통신 실패] {e}\n")
        return []


class VlmLogicNode(Node):
    def __init__(self):
        super().__init__("vlm_logic_node")
        self.bridge = CvBridge()
        self.latest_raw_image = None

        self.create_subscription(Image, "/camera/camera/color/image_raw", self.raw_image_callback, 10)
        self.create_subscription(String, "/vlm_user_command", self.command_callback, 10)

        self.cli = self.create_client(UserRequest, "/vlm_request")

        self.get_logger().info("VLM Logic 노드 가동 완료. 명령 대기 중...")

    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def command_callback(self, msg):
        user_command = msg.data.strip()
        self.get_logger().info(f"\n[{user_command}] 명령 수신됨. 다중 타겟 분석 시작...")

        if self.latest_raw_image is None:
            self.get_logger().error("아직 카메라 원본 영상이 들어오지 않았습니다.")
            return

        self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        target_list = get_target_from_gemini(self.latest_raw_image, user_command)
        
        if not target_list:
            self.get_logger().error("유효한 타겟을 찾지 못했거나 응답이 비어있습니다.")
            return

        self.get_logger().info(f"▶ Gemini 파싱 완료: {target_list}")

        # =========================================================
        # 리스트(Array)로 묶어서 한 번에 전송
        # =========================================================
        class_names_list = []
        iterations_list = []

        # 1. 파싱된 딕셔너리 리스트에서 데이터를 뽑아 각각의 배열로 만듭니다.
        for target in target_list:
            class_names_list.append(target["class_name"])
            iterations_list.append(int(target["iteration"]))

        # 2. 서비스 서버가 준비되었는지 확인
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_request)가 준비되지 않았습니다.")
            return

        self.get_logger().info(f"▶ 일괄 서비스 요청 전송 중... [물품: {class_names_list}, 개수: {iterations_list}]")
        
        # 3. Request 객체 생성 후 리스트 데이터 대입
        req = UserRequest.Request()
        req.class_names = class_names_list
        req.iterations = iterations_list

        # 4. 서비스 '한 번' 호출
        future = self.cli.call_async(req)
        
        # 5. 여러 번 보낼 필요가 없으므로 콜백도 단순해집니다.
        future.add_done_callback(self.response_callback)

    # 콜백 함수도 단순하게 원상복구
    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f"✅ 서비스 응답 수신! [성공 여부: {response.success}, 메시지: {response.message}]"
            )
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
#!/usr/bin/env python3

import os
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
from dotenv import load_dotenv

# abc_interfaces 패키지의 UserRequest 서비스 임포트
from abc_interfaces.srv import UserRequest

# 설정 파라미터
TARGET_CLASSES = [
    "Kancho", "pepero_original", "pepsi", "pocarisweat",
    "soy_milk", "chocopie", "pepero_almond"
]

# KEYWORD_MAP = {
#     "칸초": "Kancho", "칸쵸": "Kancho",
#     "오리지널 빼빼로": "pepero_original", "빼빼로": "pepero_original",
#     "펩시": "pepsi", "콜라": "pepsi",
#     "포카리스웨트": "pocarisweat", "포카리": "pocarisweat",
#     "소이밀크": "soy_milk", "두유": "soy_milk",
#     "초코파이": "chocopie",
#     "아몬드 빼빼로": "pepero_almond", "아몬드": "pepero_almond",
# }

# .env 파일 로드 (환경 변수 적용)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)


# def get_targets_from_keywords(user_command):
#     targets = []
#     selected_classes = set()
#     remaining_command = user_command
#
#     for kr_word in sorted(KEYWORD_MAP, key=len, reverse=True):
#         if kr_word in remaining_command:
#             class_name = KEYWORD_MAP[kr_word]
#             if class_name not in selected_classes:
#                 targets.append({"class_name": class_name, "iteration": 1})
#                 selected_classes.add(class_name)
#             remaining_command = remaining_command.replace(kr_word, " ")
#
#     return targets


def get_target_from_gemini(cv_image, user_command):

    cv_image = cv2.resize(cv_image, (640, 480))
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
        "너는 로봇 명령을 YOLO 탐지 요청으로 변환하는 모듈이다.\n"
        "카메라 이미지는 참고용이며, 가장 중요한 입력은 사용자 명령이다.\n"
        "사용자 명령에 들어있는 물품명과 개수를 분석해서 로봇이 찾아야 할 물품 목록을 만들어라.\n"
        f"반드시 다음 제공된 리스트 내의 영문 클래스명만 선택해야 하며, 리스트에 없는 물품은 철저히 무시해라.\n"
        f"리스트: {TARGET_CLASSES}\n\n"
        "개수가 명시되지 않은 경우 iteration은 1로 취급하라.\n"
        "사용자가 여러 물품을 말하면 각 물품을 모두 포함하라.\n"
        "사용자가 리스트에 없는 물품만 말하면 빈 배열 []을 반환하라.\n"
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
                response_mime_type="application/json", 
            ),
            request_options={"timeout": 15.0} 
        )
        
        if not response.parts:
            print("[VLM 오류] 텍스트가 생성되지 않았습니다.")
            return []
            
        text = response.text.strip()
        ###debug
        print("===== Gemini Raw Response =====")
        print(text)
        print("================================")
        ###
        
        parsed_data = json.loads(text)
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        if not isinstance(parsed_data, list):
            print(f"[VLM 파싱 실패] JSON 배열이 아닙니다: {parsed_data}")
            return []
        
        # 클래스명이 리스트에 있는지 한 번 더 검증 (안전 장치)
        valid_targets = []
        for item in parsed_data:
            if not isinstance(item, dict):
                print(f"[VLM 경고] 딕셔너리가 아닌 항목이 필터링 되었습니다: {item}")
                continue

            c_name = item.get("class_name", "")
            iteration = int(item.get("iteration", 1))
            if iteration <= 0:
                iteration = 1

            if c_name in TARGET_CLASSES:
                valid_targets.append({
                    "class_name": c_name,
                    "iteration": iteration,
                })
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

        # 키워드 매칭은 VLM 담당자 확인 전까지 비활성화.
        # target_list = get_targets_from_keywords(user_command)
        #
        # if target_list:
        #     self.get_logger().info(f"▶ 키워드 매칭 완료: {target_list}")
        # else:
        #     self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        #     target_list = get_target_from_gemini(self.latest_raw_image, user_command)

        self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        target_list = get_target_from_gemini(self.latest_raw_image, user_command)
        
        if not target_list:
            self.get_logger().error("유효한 타겟을 찾지 못했거나 응답이 비어있습니다.")
            return

        self.get_logger().info(f"▶ 타겟 분석 완료: {target_list}")

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
        req.class_name = class_names_list
        req.iteration = iterations_list

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

#!/usr/bin/env python3

import os
import json
import cv2
import rclpy
import requests
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import google.generativeai as genai
from dotenv import load_dotenv

# abc_interfaces 패키지의 서비스 임포트
# (Stt 서비스도 동일한 패키지에 존재한다고 가정합니다)
from abc_interfaces.srv import UserRequest, Stt

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
        print("===== Gemini Raw Response =====")
        print(text)
        print("================================")
        
        parsed_data = json.loads(text)
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        if not isinstance(parsed_data, list):
            print(f"[VLM 파싱 실패] JSON 배열이 아닙니다: {parsed_data}")
            return []
        
        valid_targets = []
        for item in parsed_data:
            if not isinstance(item, dict):
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
                
        return valid_targets

    except json.JSONDecodeError as e:
        print(f"\n[VLM 파싱 실패] JSON 형식이 올바르지 않습니다: {e}\n")
        return []
    except Exception as e:
        print(f"\n[VLM SDK 통신 실패] {e}\n")
        return []

def check_stock_from_db(class_name):
    """
    HTTP GET 요청을 통해 DB에서 재고를 조회합니다.
    """
    url = f"http://127.0.0.1:8000/api/v1/stock/{class_name}"
    try:
        response = requests.get(url, timeout=5.0)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"[DB 통신 오류] {e}")
        return None


class VlmLogicNode(Node):
    def __init__(self):
        super().__init__("vlm_logic_node")
        self.bridge = CvBridge()
        self.latest_raw_image = None

        # 카메라 이미지 구독
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.raw_image_callback, 10)

        # 1. 터미널 명령(Topic) 대신 STT 결과를 받는 서비스 서버 생성
        self.stt_srv = self.create_service(Stt, "/stt_results", self.stt_callback)

        # 2. VLM 요청(성공 시 바코드 리스트) 및 TTS 요청(부족 시 클래스 리스트) 클라이언트
        self.vlm_cli = self.create_client(UserRequest, "/vlm_request")
        self.tts_cli = self.create_client(UserRequest, "/vlm_to_tts")

        self.get_logger().info("VLM Logic 노드 가동 완료. STT 명령 수신 대기 중...")

    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def stt_callback(self, request, response):
        """
        /stt_results 서비스 요청이 들어왔을 때 실행되는 메인 콜백 함수입니다.
        """
        if not request.success:
            self.get_logger().error("STT 노드에서 인식 실패 상태를 전달받았습니다.")
            return response

        user_command = request.raw_text.strip()
        self.get_logger().info(f"\n[음성 인식 결과: '{user_command}'] 다중 타겟 분석 시작...")

        if self.latest_raw_image is None:
            self.get_logger().error("아직 카메라 원본 영상이 들어오지 않았습니다.")
            return response

        self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        target_list = get_target_from_gemini(self.latest_raw_image, user_command)
        
        if not target_list:
            self.get_logger().error("유효한 타겟을 찾지 못했거나 응답이 비어있습니다.")
            return response

        self.get_logger().info(f"▶ 타겟 분석 완료: {target_list}")

        # 리스트 추적용 변수 초기화
        insufficient_classes = []
        insufficient_stocks = []
        
        valid_barcodes = []
        valid_iterations = []
        
        is_any_stock_insufficient = False

        # DB 재고 확인 로직 수행
        for target in target_list:
            c_name = target["class_name"]
            req_qty = int(target["iteration"])
            
            db_res = check_stock_from_db(c_name)
            
            if db_res is None:
                self.get_logger().error(f"[{c_name}] DB 정보를 불러오지 못했습니다. 재고 부족으로 간주합니다.")
                is_any_stock_insufficient = True
                insufficient_classes.append(c_name)
                insufficient_stocks.append(0)
                continue
                
            stock = db_res.get("remaining_stock", 0)
            barcode = db_res.get("barcode_data", "")
            
            if req_qty > stock:
                # 하나라도 재고가 부족하다면 플래그를 변경하고 부족한 상품 리스트에 추가
                is_any_stock_insufficient = True
                insufficient_classes.append(c_name)
                insufficient_stocks.append(stock)
            else:
                # 재고가 충분한 상품은 바코드 데이터로 변환하여 리스트업
                valid_barcodes.append(barcode)
                valid_iterations.append(req_qty)

        # 재고 수량 비교 후 분기 처리
        if is_any_stock_insufficient:
            self._send_tts_request(insufficient_classes, insufficient_stocks)
        else:
            self._send_vlm_request(valid_barcodes, valid_iterations)

        return response

    def _send_tts_request(self, classes, stocks):
        """재고 부족 시 부족한 상품의 정보만 TTS 서비스로 전달"""
        if not self.tts_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_to_tts)가 준비되지 않았습니다.")
            return

        self.get_logger().warn(
            f"▶ 재고 부족 상품 발생. TTS 서비스 요청 전송 중... [물품: {classes}, DB잔여재고: {stocks}]"
        )
        
        req = UserRequest.Request()
        req.class_name = classes
        req.iteration = stocks

        future = self.tts_cli.call_async(req)
        future.add_done_callback(lambda f: self.response_callback(f, "TTS"))

    def _send_vlm_request(self, barcodes, iterations):
        """모든 재고가 충분할 시 바코드 기반으로 메인 서비스에 전달"""
        if not self.vlm_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_request)가 준비되지 않았습니다.")
            return

        self.get_logger().info(
            f"▶ 모든 재고 충분. VLM 서비스 요청 전송 중... [바코드: {barcodes}, 개수: {iterations}]"
        )
        
        req = UserRequest.Request()
        req.class_name = barcodes
        req.iteration = iterations

        future = self.vlm_cli.call_async(req)
        future.add_done_callback(lambda f: self.response_callback(f, "VLM"))

    def response_callback(self, future, node_type):
        try:
            response = future.result()
            self.get_logger().info(
                f"✅ {node_type} 서비스 응답 수신! [성공 여부: {response.success}, 메시지: {response.message}]"
            )
        except Exception as e:
            self.get_logger().error(f"❌ {node_type} 서비스 호출 실패: {e}")

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
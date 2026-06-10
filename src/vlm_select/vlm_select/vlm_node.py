#!/usr/bin/env python3

import os
import json
import cv2
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import google.generativeai as genai
from dotenv import load_dotenv

# abc_interfaces 패키지의 UserRequest 서비스 임포트
from abc_interfaces.srv import UserRequest

# stt_interfaces 패키지의 Stt 서비스 임포트
from abc_interfaces.srv import Stt

# 설정 파라미터
TARGET_CLASSES = [
    "Kancho", "pepero_original", "pepsi", "pocarisweat",
    "soy_milk", "chocopie", "pepero_almond"
]

# DB API 엔드포인트
# DB_BASE_URL = "http://127.0.0.1:8000"
DB_BASE_URL = "https://ssu-abc-store-api.ssammwu.info"
DB_STOCK_ENDPOINT = "/api/v1/stock/{class_name}"

# .env 파일 로드 (환경 변수 적용)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)


# ============================================================
# Gemini VLM: 사용자 명령 → 타겟 클래스 + 개수 리스트 변환
# ============================================================
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


# ============================================================
# DB API: 단일 상품 재고 조회
# ============================================================
def fetch_stock_from_db(class_name: str) -> dict:
    """
    DB REST API에서 class_name에 해당하는 재고 정보를 조회한다.

    호출 대상 URL을 로그로 함께 출력하며, 반환값(dict)은 항상 "status" 키를 포함한다.
      - {"status": "ok",        "data": {...}}   : 정상 조회 (data 안에 remaining_stock, barcode_data 등)
      - {"status": "not_found", "detail": str}   : 상품이 DB에 등록되지 않음 (HTTP 404)
      - {"status": "error",     "detail": str}   : DB 통신/HTTP/파싱 오류 (API 호출 실패)
    """
    # 어떤 주소에서 받아오는지 명시적으로 로그에 남긴다.
    url = DB_BASE_URL + DB_STOCK_ENDPOINT.format(class_name=class_name)
    print(f"[DB 요청] {class_name} → GET {url}")

    # ── API 호출 실패: 연결/타임아웃 등 통신 예외 처리 ──────────────
    try:
        resp = requests.get(url, timeout=5.0)
    except requests.exceptions.Timeout:
        detail = f"DB 응답 시간 초과(timeout) - {url}"
        print(f"[DB 통신 오류] {class_name}: {detail}")
        return {"status": "error", "detail": detail}
    except requests.exceptions.ConnectionError:
        detail = f"DB 서버에 연결할 수 없습니다(connection error) - {url}"
        print(f"[DB 통신 오류] {class_name}: {detail}")
        return {"status": "error", "detail": detail}
    except requests.exceptions.RequestException as e:
        detail = f"DB 요청 중 예외 발생: {e} - {url}"
        print(f"[DB 통신 오류] {class_name}: {detail}")
        return {"status": "error", "detail": detail}

    # ── 정상 응답: JSON 파싱 실패까지 방어 ─────────────────────────
    if resp.status_code == 200:
        try:
            return {"status": "ok", "data": resp.json()}
        except ValueError as e:
            detail = f"DB 응답 JSON 파싱 실패: {e} / 원문: {resp.text}"
            print(f"[DB 오류] {class_name}: {detail}")
            return {"status": "error", "detail": detail}

    # ── 상품 미등록(404)은 통신 실패와 구분한다 ────────────────────
    if resp.status_code == 404:
        detail = f"DB에 등록되지 않은 상품(HTTP 404) - {url}"
        print(f"[DB 오류] {class_name}: {detail}")
        return {"status": "not_found", "detail": detail}

    # ── 그 외 HTTP 오류는 API 호출 실패로 처리 ─────────────────────
    detail = f"DB 조회 실패 (HTTP {resp.status_code}): {resp.text}"
    print(f"[DB 오류] {class_name}: {detail}")
    return {"status": "error", "detail": detail}


# ============================================================
# VLM 메인 노드
# ============================================================
class VlmLogicNode(Node):
    def __init__(self):
        super().__init__("vlm_logic_node")
        self.bridge = CvBridge()
        self.latest_raw_image = None

        # 카메라 이미지 구독
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.raw_image_callback,
            10
        )

        # ── 요구사항 1 ──────────────────────────────────────────────
        # /stt_results 서비스 서버 생성 (Stt.srv 타입)
        # vlm_command_node.py의 키보드 입력 대신 STT 결과를 서비스로 수신
        self.stt_srv = self.create_service(
            Stt,
            "/stt_results",
            self.stt_service_callback
        )

        # ── 요구사항 3 ──────────────────────────────────────────────
        # /vlm_to_tts 서비스 클라이언트 (재고 부족 시 TTS 알림용)
        self.tts_cli = self.create_client(UserRequest, "/vlm_to_tts")

        # ── 요구사항 4 ──────────────────────────────────────────────
        # /vlm_request 서비스 클라이언트 (재고 충분 시 YOLO 요청용)
        self.vlm_cli = self.create_client(UserRequest, "/vlm_request")

        self.get_logger().info("VLM Logic 노드 가동 완료. STT 서비스(/stt_results) 대기 중...")

    # ── 카메라 콜백 ─────────────────────────────────────────────────
    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    # ── 요구사항 1: STT 서비스 콜백 ────────────────────────────────
    def stt_service_callback(self, request: Stt.Request, response: Stt.Response):
        """
        /stt_results 서비스 요청을 처리한다.
        request.raw_text : STT로 인식된 사용자 명령 문자열
        """
        user_command = request.raw_text.strip()

        if not user_command:
            self.get_logger().warn("빈 STT 명령 수신. 무시합니다.")
            response.success = False
            response.message = "빈 명령입니다."
            return response

        self.get_logger().info(f"\n[{user_command}] STT 명령 수신됨. 타겟 분석 시작...")

        if self.latest_raw_image is None:
            self.get_logger().error("아직 카메라 원본 영상이 들어오지 않았습니다.")
            response.success = False
            response.message = "카메라 영상 미수신 상태입니다."
            return response

        # 1단계: Gemini로 명령 분석
        self.get_logger().info("▶ Gemini API로 문맥 분석 요청 중...")
        target_list = get_target_from_gemini(self.latest_raw_image, user_command)

        if not target_list:
            self.get_logger().error("유효한 타겟을 찾지 못했거나 응답이 비어있습니다.")
            response.success = False
            response.message = "유효한 타겟을 파악하지 못했습니다."
            return response

        self.get_logger().info(f"▶ 타겟 분석 완료: {target_list}")

        # 2단계: DB 재고 조회 및 재고 판단
        self.process_targets_with_db(target_list)

        response.success = True
        response.message = "명령 처리를 시작했습니다."
        return response

    # ── 요구사항 2~4: DB 조회 + 재고 판단 + 서비스 전송 ────────────
    def process_targets_with_db(self, target_list: list):
        """
        target_list: [{"class_name": str, "iteration": int}, ...]

        1. 각 상품의 DB 재고를 조회한다.
        2. 재고가 부족한 상품이 하나라도 있으면 /vlm_to_tts 서비스를 호출한다.
        3. 모든 상품의 재고가 충분하면 /vlm_request 서비스를 호출한다.
           이때 class_name 대신 DB에서 받은 barcode_data를 전송한다.
        """
        self.get_logger().info("▶ DB 재고 조회 시작...")

        insufficient_class_names = []   # 부족한 상품 class_name
        insufficient_stocks     = []    # 부족한 상품의 현재 DB 재고

        sufficient_barcodes    = []     # 충분한 상품의 barcode_data
        sufficient_iterations  = []     # 충분한 상품의 요구 개수

        db_results = {}  # class_name → DB 응답 dict

        # ── 요구사항 2: 상품별 재고 조회 ──────────────────────────
        for target in target_list:
            c_name    = target["class_name"]
            requested = target["iteration"]

            stock_info = fetch_stock_from_db(c_name)

            if stock_info["status"] != "ok":
                # DB 조회 실패(통신 오류/미등록) → 재고 0으로 간주
                self.get_logger().error(
                    f"  [{c_name}] DB 조회 실패({stock_info['status']}) → 재고 0으로 처리"
                )
                insufficient_class_names.append(c_name)
                insufficient_stocks.append(0)
                continue

            data      = stock_info["data"]
            remaining = data.get("remaining_stock", 0)
            barcode   = data.get("barcode_data", "")
            db_results[c_name] = data

            self.get_logger().info(
                f"  [{c_name}] 요구: {requested}개 / 재고: {remaining}개 / 바코드: {barcode}"
            )

            # ── 요구사항 3: 재고 부족 판단 ────────────────────────
            if remaining < requested:
                self.get_logger().warn(
                    f"  [{c_name}] 재고 부족! (재고 {remaining}개 < 요구 {requested}개)"
                )
                insufficient_class_names.append(c_name)
                insufficient_stocks.append(remaining)
            else:
                sufficient_barcodes.append(barcode)
                sufficient_iterations.append(requested)

        # ── 요구사항 3: 부족 상품이 있으면 /vlm_to_tts 호출 ───────
        if insufficient_class_names:
            self.get_logger().warn(
                f"▶ 재고 부족 상품 발생 → /vlm_to_tts 서비스 전송\n"
                f"  부족 상품: {insufficient_class_names}\n"
                f"  현재 재고: {insufficient_stocks}"
            )
            self.call_tts_service(insufficient_class_names, insufficient_stocks)
            return  # 재고 부족이 있으면 YOLO 요청은 보내지 않음

        # ── 요구사항 4: 모든 재고 충분 → /vlm_request 호출 ────────
        self.get_logger().info(
            f"▶ 모든 상품 재고 충분 → /vlm_request 서비스 전송\n"
            f"  바코드 목록: {sufficient_barcodes}\n"
            f"  요구 개수:   {sufficient_iterations}"
        )
        self.call_vlm_request_service(sufficient_barcodes, sufficient_iterations)

    # ── /vlm_to_tts 서비스 호출 ─────────────────────────────────────
    def call_tts_service(self, class_names: list, stocks: list):
        if not self.tts_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_to_tts)가 준비되지 않았습니다.")
            return

        req = UserRequest.Request()
        req.class_name = class_names          # 부족한 상품 class_name 리스트
        req.iteration  = [int(s) for s in stocks]  # 현재 DB 재고 리스트

        future = self.tts_cli.call_async(req)
        future.add_done_callback(self.tts_response_callback)

    def tts_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f"✅ /vlm_to_tts 응답 수신! [성공: {response.success}, 메시지: {response.message}]"
            )
        except Exception as e:
            self.get_logger().error(f"❌ /vlm_to_tts 서비스 호출 실패: {e}")

    # ── /vlm_request 서비스 호출 ────────────────────────────────────
    def call_vlm_request_service(self, barcodes: list, iterations: list):
        if not self.vlm_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_request)가 준비되지 않았습니다.")
            return

        req = UserRequest.Request()
        req.class_name = barcodes             # barcode_data 리스트 (요구사항 4)
        req.iteration  = [int(i) for i in iterations]

        future = self.vlm_cli.call_async(req)
        future.add_done_callback(self.vlm_request_response_callback)

    def vlm_request_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(
                f"✅ /vlm_request 응답 수신! [성공: {response.success}, 메시지: {response.message}]"
            )
        except Exception as e:
            self.get_logger().error(f"❌ /vlm_request 서비스 호출 실패: {e}")


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

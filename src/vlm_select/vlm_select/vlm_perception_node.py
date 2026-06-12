#!/usr/bin/env python3
"""
VLM Perception Node (단일 프롬프트 통합 버전)
===================================================================
기존 두 노드의 역할을 하나로 통합한 노드.

  1) vlm_select/vlm_node.py (VlmLogicNode)
       - STT 명령(/stt_results) → Gemini 문맥 분석 → DB 재고 조회
       - 재고 부족 시 /vlm_to_tts, 재고 충분 시 /vlm_request 호출

  2) abc_perception/yolo_node.py (YoloNode)
       - /request_item 으로 받은 바코드(class_name)에 해당하는 물품의
         박스 좌표를 /response_item 으로 전송
       - /yolo_tracking 으로 추적 좌표 퍼블리시

[기존 2-call 구조와의 차이]
  기존에는 ① 명령→타겟 분석(Gemini)과 ② 이미지→박스 탐지(YOLO/Gemini)를
  각각 별도 API 로 호출했다. 이 버전은 둘을 **하나의 프롬프트/한 번의 Gemini
  호출**로 합쳐, 명령을 받은 시점에 "어떤 물품을 몇 개 + 화면 어디에" 를 한꺼번에
  받아온다. 탐지 결과는 캐시되어, 이후 /request_item 요청과 추적 퍼블리시는
  추가 API 호출 없이 캐시에서 응답한다.

  → 트레이드오프: 박스 좌표가 "명령 시점" 기준이므로, 명령~집기 사이에 카메라나
    물체가 크게 움직이는 환경에서는 좌표가 낡을 수 있다. (정적 진열 가정)

[설정]
  탐지 클래스명과 바코드는 코드가 아니라 config/config.yaml 에서 읽어 온다.
  물품을 추가/변경할 때 코드 수정 없이 config.yaml 만 고치면 된다.

서비스/토픽 이름은 모두 기존과 동일하게 유지하여 두 노드의 드롭인 대체로 동작한다.
"""

import os
import json

import cv2
import yaml
import requests
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import google.generativeai as genai
from dotenv import load_dotenv

# abc_interfaces 패키지의 서비스 임포트
from abc_interfaces.srv import UserRequest     # /vlm_to_tts, /vlm_request
from abc_interfaces.srv import Stt             # /stt_results
from abc_interfaces.srv import RequestItem     # /request_item (YOLO 통합)
from abc_interfaces.srv import ResponseItem    # /response_item (YOLO 통합)


# ============================================================
# 설정 파라미터
# ============================================================
# Gemini 모델 (명령 분석 + 박스 좌표 탐지를 한 번에 수행)
GEMINI_MODEL = "gemini-3-flash-preview"

# DB API 엔드포인트
# DB_BASE_URL = "http://127.0.0.1:8000"
DB_BASE_URL = "https://ssu-abc-store-api.ssammwu.info"
DB_STOCK_ENDPOINT = "/api/v1/stock/{class_name}"

# .env 파일 로드 (환경 변수 적용)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)


# ============================================================
# config.yaml 로드: 클래스명 ↔ 바코드 매핑
# ============================================================
def resolve_config_path(param_path: str) -> str:
    """config.yaml 경로를 결정한다.
    1) ROS 파라미터(config_file)로 직접 지정한 경로
    2) 설치된 share 디렉터리(share/vlm_select/config/config.yaml)
    3) 소스 트리 기준 상대 경로(개발 중 colcon 설치 없이 실행할 때)
    """
    candidates = []
    if param_path:
        candidates.append(param_path)
    try:
        share_dir = get_package_share_directory("vlm_select")
        candidates.append(os.path.join(share_dir, "config", "config.yaml"))
    except Exception:
        pass
    candidates.append(os.path.join(current_dir, "..", "config", "config.yaml"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    # 못 찾으면 마지막 후보 경로를 반환(에러 메시지에 활용)
    return candidates[-1]


def load_class_config(path: str):
    """config.yaml 을 읽어 (target_classes, class_to_barcode, barcode_to_class) 반환."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    classes = data.get("classes", [])
    target_classes = []
    class_to_barcode = {}
    for entry in classes:
        name = str(entry["name"])
        barcode = str(entry["barcode"])
        target_classes.append(name)
        class_to_barcode[name] = barcode

    barcode_to_class = {barcode: name for name, barcode in class_to_barcode.items()}
    return target_classes, class_to_barcode, barcode_to_class


# ============================================================
# Gemini 모델 핸들 (1회만 구성)
# ============================================================
def build_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        print("[오류] .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다!")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


# ============================================================
# Gemini VLM (단일 호출): 명령 + 이미지 → 타겟(개수) + 박스 좌표
# ============================================================
def analyze_command_and_detect(model, cv_image, user_command, target_classes):
    """
    한 번의 Gemini 호출로 다음을 동시에 수행한다.
      (1) 사용자 명령을 해석해 찾아야 할 물품과 개수(iteration)를 정한다.
      (2) 그 물품들이 현재 이미지의 어디에 있는지 박스 좌표를 탐지한다.

    반환값: [
        {
            "class_name": <영문 클래스명>,
            "iteration":  <int, 명령에서 파악한 요구 개수>,
            "confidence": <float, 화면에 안 보이면 0.0>,
            # 화면에서 탐지된 경우에만 픽셀 좌표(원본 해상도 기준)가 채워진다.
            "center_x", "center_y", "width", "height": <float | None>,
        }, ...
    ]
    실패/오류 시 빈 리스트를 반환한다.
    """
    if model is None:
        return None

    # 원본 해상도(픽셀 좌표 환산 기준)를 보관. API 전송용으로만 리사이즈한다.
    orig_h, orig_w = cv_image.shape[:2]
    send_image = cv2.resize(cv_image, (640, 480))
    ok, buf = cv2.imencode('.jpg', send_image)
    if not ok:
        print("[VLM 오류] 이미지 인코딩 실패")
        return None
    image_bytes = buf.tobytes()

    prompt = (
        "너는 로봇 명령을 해석하면서 동시에 카메라 이미지에서 물품 위치를 찾는 모듈이다.\n"
        "한 번에 두 가지 일을 한다:\n"
        "  (1) 사용자 명령을 분석해 찾아야 할 물품과 개수를 정한다.\n"
        "  (2) 그 물품이 이미지의 어디에 있는지 박스 좌표를 찾는다.\n\n"
        f"반드시 다음 리스트 내의 영문 클래스명만 사용하고, 리스트에 없는 물품은 무시하라.\n"
        f"리스트: {target_classes}\n\n"
        "규칙:\n"
        "- 사용자 명령에 등장하는 (리스트 내) 물품마다 하나의 항목을 만든다.\n"
        "- iteration 은 명령에서 파악한 요구 개수이며, 명시가 없으면 1 로 한다.\n"
        "- 사용자가 여러 물품을 말하면 모두 포함한다.\n"
        "- 리스트에 없는 물품만 말하면 빈 배열 []을 반환한다.\n"
        "- 박스 좌표(box_2d)는 [ymin, xmin, ymax, xmax] 순서이며, 이미지 왼쪽/위를 0,\n"
        "  오른쪽/아래를 1000 으로 하는 0~1000 정규화 정수 좌표로 출력한다.\n"
        "- 해당 물품이 이미지에 보이면 box_2d 와 0.0~1.0 사이 confidence 를 채운다.\n"
        "- 명령에는 있으나 이미지에 보이지 않으면 box_2d 는 null, confidence 는 0.0 으로 한다.\n\n"
        "출력은 반드시 아래 구조의 JSON 배열(Array) 형식으로만 반환하라.\n"
        "예시:\n"
        "[\n"
        "  {\"class_name\": \"pepero_original\", \"iteration\": 2, \"box_2d\": [120, 340, 600, 520], \"confidence\": 0.93},\n"
        "  {\"class_name\": \"pepsi\", \"iteration\": 1, \"box_2d\": null, \"confidence\": 0.0}\n"
        "]\n\n"
        f"사용자 명령: {user_command}"
    )

    text = ""
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
            print("[VLM 오류] 응답이 비어있습니다.")
            return []

        text = response.text.strip()
        print("===== Gemini Raw Response =====")
        print(text)
        print("===============================")

        parsed_data = json.loads(text)
        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]
        if not isinstance(parsed_data, list):
            print(f"[VLM 파싱 실패] JSON 배열이 아닙니다: {parsed_data}")
            return []

        results = []
        for item in parsed_data:
            if not isinstance(item, dict):
                print(f"[VLM 경고] 딕셔너리가 아닌 항목 필터링: {item}")
                continue

            c_name = item.get("class_name", "")
            if c_name not in target_classes:
                print(f"[VLM 경고] 리스트에 없는 값 필터링: '{c_name}'")
                continue

            iteration = int(item.get("iteration", 1))
            if iteration <= 0:
                iteration = 1

            result = {
                "class_name": c_name,
                "iteration": iteration,
                "confidence": float(item.get("confidence", 0.0)),
                "center_x": None,
                "center_y": None,
                "width": None,
                "height": None,
            }

            box = item.get("box_2d")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                # [ymin, xmin, ymax, xmax] (0~1000) → 원본 픽셀 좌표
                ymin, xmin, ymax, xmax = [float(v) for v in box]
                x1 = xmin / 1000.0 * orig_w
                y1 = ymin / 1000.0 * orig_h
                x2 = xmax / 1000.0 * orig_w
                y2 = ymax / 1000.0 * orig_h

                # 좌표 정렬(역전 방어) 및 클램핑
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                x1 = max(0.0, min(x1, orig_w))
                x2 = max(0.0, min(x2, orig_w))
                y1 = max(0.0, min(y1, orig_h))
                y2 = max(0.0, min(y2, orig_h))

                result["center_x"] = float((x1 + x2) / 2.0)
                result["center_y"] = float((y1 + y2) / 2.0)
                result["width"] = float(x2 - x1)
                result["height"] = float(y2 - y1)

            results.append(result)

        return results

    except json.JSONDecodeError as e:
        print(f"\n[VLM 파싱 실패] JSON 형식 오류: {e}\n응답 데이터: {text}\n")
        return []
    except Exception as e:
        print(f"\n[VLM SDK 통신 실패] {e}\n")
        return []


# ============================================================
# DB API: 단일 상품 재고 조회
# (기존 vlm_node.fetch_stock_from_db 와 동일 로직)
# ============================================================
def fetch_stock_from_db(class_name: str) -> dict:
    """
    DB REST API에서 class_name에 해당하는 재고 정보를 조회한다.

    반환값(dict)은 항상 "status" 키를 포함한다.
      - {"status": "ok",        "data": {...}}   : 정상 조회
      - {"status": "not_found", "detail": str}   : 상품 미등록 (HTTP 404)
      - {"status": "error",     "detail": str}   : DB 통신/HTTP/파싱 오류
    """
    url = DB_BASE_URL + DB_STOCK_ENDPOINT.format(class_name=class_name)
    print(f"[DB 요청] {class_name} → GET {url}")

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

    if resp.status_code == 200:
        try:
            return {"status": "ok", "data": resp.json()}
        except ValueError as e:
            detail = f"DB 응답 JSON 파싱 실패: {e} / 원문: {resp.text}"
            print(f"[DB 오류] {class_name}: {detail}")
            return {"status": "error", "detail": detail}

    if resp.status_code == 404:
        detail = f"DB에 등록되지 않은 상품(HTTP 404) - {url}"
        print(f"[DB 오류] {class_name}: {detail}")
        return {"status": "not_found", "detail": detail}

    detail = f"DB 조회 실패 (HTTP {resp.status_code}): {resp.text}"
    print(f"[DB 오류] {class_name}: {detail}")
    return {"status": "error", "detail": detail}


# ============================================================
# 통합 VLM Perception 노드
# ============================================================
class VlmPerceptionNode(Node):
    def __init__(self):
        super().__init__("vlm_perception_node")
        self.bridge = CvBridge()
        self.latest_raw_image = None

        # ── config.yaml 로드 (클래스명 ↔ 바코드 매핑) ───────────────
        config_file = (
            self.declare_parameter("config_file", "")
            .get_parameter_value().string_value
        )
        config_path = resolve_config_path(config_file)
        try:
            (self.target_classes,
             self.class_to_barcode,
             self.barcode_to_class) = load_class_config(config_path)
            self.get_logger().info(
                f"config 로드 완료: {config_path}\n  클래스: {self.target_classes}"
            )
        except Exception as e:
            self.get_logger().error(f"config 로드 실패({config_path}): {e}")
            self.target_classes, self.class_to_barcode, self.barcode_to_class = [], {}, {}

        # Gemini 모델 1회 구성 후 재사용
        self.model = build_gemini_model()
        if self.model is None:
            self.get_logger().error("Gemini 모델 초기화 실패 (GEMINI_API_KEY 확인 필요).")

        # ── 파라미터 ────────────────────────────────────────────────
        self.request_service_name = (
            self.declare_parameter("request_service_name", "/request_item")
            .get_parameter_value().string_value
        )
        self.response_service_name = (
            self.declare_parameter("response_service_name", "/response_item")
            .get_parameter_value().string_value
        )
        # YOLO 와의 호환을 위해 추적 토픽 이름 기본값은 유지한다.
        self.tracking_topic_name = (
            self.declare_parameter("tracking_topic_name", "/yolo_tracking")
            .get_parameter_value().string_value
        )
        # 추적 좌표 퍼블리시 주기(초). 단일 호출 구조이므로 추가 API 호출 없이
        # 캐시된 박스를 이 주기로 재퍼블리시한다.
        self.tracking_period = (
            self.declare_parameter("tracking_period", 1.0)
            .get_parameter_value().double_value
        )

        # 단일 Gemini 호출로 받은 탐지 결과 캐시: class_name → detection dict
        self.cached_detections = {}

        # ── 카메라 이미지 구독 ──────────────────────────────────────
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.raw_image_callback,
            10
        )

        # ── (vlm_node 역할) STT 명령 서비스 서버 ───────────────────
        self.stt_srv = self.create_service(
            Stt,
            "/stt_results",
            self.stt_service_callback
        )

        # ── (vlm_node 역할) TTS / vlm_request 클라이언트 ───────────
        self.tts_cli = self.create_client(UserRequest, "/vlm_to_tts")
        self.vlm_cli = self.create_client(UserRequest, "/vlm_request")

        # ── (YOLO 역할) 물품 탐지 요청 서비스 서버 ─────────────────
        self.request_srv = self.create_service(
            RequestItem,
            self.request_service_name,
            self.handle_item_request
        )

        # ── (YOLO 역할) 탐지 결과 응답 클라이언트 ──────────────────
        self.response_cli = self.create_client(
            ResponseItem,
            self.response_service_name
        )

        # ── (YOLO 역할) 추적 좌표 퍼블리셔 ─────────────────────────
        self.tracking_pub = self.create_publisher(Vector3, self.tracking_topic_name, 10)
        self.tracking_class = None   # 현재 추적 중인 영문 클래스명
        self.tracking_timer = self.create_timer(
            max(0.1, self.tracking_period),
            self.tracking_timer_callback
        )

        self.get_logger().info(
            "VLM Perception 노드 가동 완료 (단일 프롬프트 통합).\n"
            f"  STT 서비스:      /stt_results\n"
            f"  물품 요청 서비스: {self.request_service_name}\n"
            f"  탐지 응답 서비스: {self.response_service_name}\n"
            f"  추적 토픽:        {self.tracking_topic_name}"
        )

    # ── 카메라 콜백 ─────────────────────────────────────────────────
    def raw_image_callback(self, msg):
        try:
            self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 실패: {e}")

    # ================================================================
    # (1) STT → [단일 Gemini 호출: 타겟+박스] → DB 재고 → TTS / vlm_request
    # ================================================================
    def stt_service_callback(self, request: Stt.Request, response: Stt.Response):
        user_command = request.raw_text.strip()

        if not user_command:
            self.get_logger().warn("빈 STT 명령 수신. 무시합니다.")
            response.success = False
            response.message = "빈 명령입니다."
            return response

        self.get_logger().info(f"\n[{user_command}] STT 명령 수신됨. 분석 시작...")

        if self.latest_raw_image is None:
            self.get_logger().error("아직 카메라 원본 영상이 들어오지 않았습니다.")
            response.success = False
            response.message = "카메라 영상 미수신 상태입니다."
            return response

        # ── 단일 Gemini 호출: 명령 해석 + 박스 탐지 동시 수행 ──────
        self.get_logger().info("▶ Gemini API로 명령 분석 + 박스 탐지 요청 중 (단일 호출)...")
        analyzed = analyze_command_and_detect(
            self.model, self.latest_raw_image, user_command, self.target_classes
        )

        if not analyzed:
            self.get_logger().error("유효한 타겟을 찾지 못했거나 응답이 비어있습니다.")
            response.success = False
            response.message = "유효한 타겟을 파악하지 못했습니다."
            return response

        # 박스가 탐지된 항목만 바코드 키로 캐시(이후 /request_item 응답에 사용)
        self.update_detection_cache(analyzed)

        # DB 재고 판단에는 class_name + iteration 만 사용
        target_list = [
            {"class_name": a["class_name"], "iteration": a["iteration"]}
            for a in analyzed
        ]
        self.get_logger().info(f"▶ 분석 완료: {target_list}")

        self.process_targets_with_db(target_list)

        response.success = True
        response.message = "명령 처리를 시작했습니다."
        return response

    def update_detection_cache(self, analyzed: list):
        """단일 호출 결과 중 박스가 있는 항목을 class_name 키로 캐시한다."""
        self.cached_detections = {}
        for a in analyzed:
            if a["center_x"] is None:
                self.get_logger().warn(f"  [{a['class_name']}] 화면에서 탐지되지 않음(박스 없음).")
                continue
            self.cached_detections[a["class_name"]] = {
                "class_name": a["class_name"],
                "confidence": a["confidence"],
                "center_x": a["center_x"],
                "center_y": a["center_y"],
                "width": a["width"],
                "height": a["height"],
            }

    def process_targets_with_db(self, target_list: list):
        """
        target_list: [{"class_name": str, "iteration": int}, ...]

        1. 각 상품의 DB 재고를 조회한다.
        2. 재고가 부족한 상품이 하나라도 있으면 /vlm_to_tts 서비스를 호출한다.
        3. 모든 상품의 재고가 충분하면 /vlm_request 서비스를 호출한다 (바코드 전송).
        """
        self.get_logger().info("▶ DB 재고 조회 시작...")

        insufficient_class_names = []
        insufficient_stocks = []

        sufficient_barcodes = []
        sufficient_iterations = []

        for target in target_list:
            c_name = target["class_name"]
            requested = target["iteration"]

            stock_info = fetch_stock_from_db(c_name)

            if stock_info["status"] != "ok":
                self.get_logger().error(
                    f"  [{c_name}] DB 조회 실패({stock_info['status']}) → 재고 0으로 처리"
                )
                insufficient_class_names.append(c_name)
                insufficient_stocks.append(0)
                continue

            data = stock_info["data"]
            remaining = data.get("remaining_stock", 0)
            barcode = data.get("barcode_data", "")

            self.get_logger().info(
                f"  [{c_name}] 요구: {requested}개 / 재고: {remaining}개 / 바코드: {barcode}"
            )

            if remaining < requested:
                self.get_logger().warn(
                    f"  [{c_name}] 재고 부족! (재고 {remaining}개 < 요구 {requested}개)"
                )
                insufficient_class_names.append(c_name)
                insufficient_stocks.append(remaining)
            else:
                sufficient_barcodes.append(barcode)
                sufficient_iterations.append(requested)

        if insufficient_class_names:
            self.get_logger().warn(
                f"▶ 재고 부족 상품 발생 → /vlm_to_tts 서비스 전송\n"
                f"  부족 상품: {insufficient_class_names}\n"
                f"  현재 재고: {insufficient_stocks}"
            )
            self.call_tts_service(insufficient_class_names, insufficient_stocks)
            return  # 재고 부족이 있으면 물품 요청은 보내지 않음

        self.get_logger().info(
            f"▶ 모든 상품 재고 충분 → /vlm_request 서비스 전송\n"
            f"  바코드 목록: {sufficient_barcodes}\n"
            f"  요구 개수:   {sufficient_iterations}"
        )
        self.call_vlm_request_service(sufficient_barcodes, sufficient_iterations)

    def call_tts_service(self, class_names: list, stocks: list):
        if not self.tts_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_to_tts)가 준비되지 않았습니다.")
            return

        req = UserRequest.Request()
        req.class_name = class_names
        req.iteration = [int(s) for s in stocks]

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

    def call_vlm_request_service(self, barcodes: list, iterations: list):
        if not self.vlm_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("서비스 서버(/vlm_request)가 준비되지 않았습니다.")
            return

        req = UserRequest.Request()
        req.class_name = barcodes
        req.iteration = [int(i) for i in iterations]

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

    # ================================================================
    # (2) YOLO 역할: /request_item → 캐시 조회 → /response_item
    #     (추가 API 호출 없이, 단일 호출로 받아둔 박스를 그대로 사용)
    # ================================================================
    def handle_item_request(self, request, response):
        # /request_item 의 class_name 은 바코드(또는 'unknown') 값으로 들어온다.
        requested_barcode = request.class_name.strip()

        if not requested_barcode:
            response.success = False
            response.message = "에러: 요청된 클래스가 없습니다."
            self.get_logger().error(response.message)
            return response

        # 바코드 → 영문 클래스명 변환
        target_class = self.barcode_to_class.get(requested_barcode)
        if target_class is None:
            known = sorted(self.barcode_to_class.keys())
            response.success = False
            response.message = (
                f"에러: '{requested_barcode}'은(는) config 에 없는 클래스(바코드)입니다. "
                f"사용 가능: {known}"
            )
            self.get_logger().error(response.message)
            return response

        # 단일 호출에서 캐시해 둔 박스를 조회
        detection = self.cached_detections.get(target_class)
        if detection is None:
            response.success = False
            response.message = (
                f"'{target_class}' 박스 캐시 없음 "
                f"(최근 명령에서 탐지되지 않았거나 화면에 없음)"
            )
            self.get_logger().warn(response.message)
            return response

        # /response_item 으로는 기존 YOLO 와 동일하게 바코드를 class_name 으로 전송
        if not self.send_item_response(detection, requested_barcode):
            response.success = False
            response.message = (
                f"'{target_class}' 위치는 있지만 "
                f"{self.response_service_name} 서비스 전송에 실패했습니다."
            )
            return response

        # 추적은 영문 클래스명으로 관리한다.
        self.tracking_class = target_class
        response.success = True
        response.message = f"'{target_class}' 위치 응답 전송 완료"
        self.get_logger().info(response.message)
        return response

    def send_item_response(self, detection, barcode):
        if not self.response_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f"서비스 서버({self.response_service_name})가 준비되지 않았습니다."
            )
            return False

        req = ResponseItem.Request()
        req.class_name = barcode   # 기존 YOLO 호환: 바코드 전송
        req.confidence = detection["confidence"]
        req.center_x = detection["center_x"]
        req.center_y = detection["center_y"]
        req.width = detection["width"]
        req.height = detection["height"]

        future = self.response_cli.call_async(req)
        future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(
                    f"{self.response_service_name} 응답 성공: {response.message}"
                )
            else:
                self.get_logger().warn(
                    f"{self.response_service_name} 응답 실패: {response.message}"
                )
        except Exception as e:
            self.get_logger().error(f"{self.response_service_name} 호출 실패: {e}")

    # ── (YOLO 역할) 추적 좌표 퍼블리시 ─────────────────────────────
    def tracking_timer_callback(self):
        """
        단일 호출 구조이므로 추가 Gemini 호출 없이, 캐시된 박스를 주기적으로
        /yolo_tracking 으로 재퍼블리시한다. (정적 진열 가정)
        """
        if self.tracking_class is None:
            return
        det = self.cached_detections.get(self.tracking_class)
        if det is None:
            return

        tracking_msg = Vector3()
        tracking_msg.x = det["center_x"]
        tracking_msg.y = det["center_y"]
        tracking_msg.z = det["confidence"]
        self.tracking_pub.publish(tracking_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VlmPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

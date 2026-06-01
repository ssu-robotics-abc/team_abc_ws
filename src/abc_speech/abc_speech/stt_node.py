import rclpy
from rclpy.node import Node

import re
from rapidfuzz import process, fuzz
from jamo import h2j, j2hcj
import time
import json
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
from abc_interfaces.srv import Stt, SttStart, Tts


class SttNode(Node):

    def __init__(self):
        super().__init__("stt_node")

        self.declare_parameter("language", "ko-KR")
        self.declare_parameter("device_index", -1)
        self.declare_parameter("energy_threshold", 50.0)
        # self.declare_parameter("pause_threshold", 0.8)
        # self.declare_parameter("phrase_time_limit", 5.0)
        self.declare_parameter("pause_threshold", 1.5)
        self.declare_parameter("phrase_time_limit", 10.0)
        self.declare_parameter("dynamic_energy", True)
        self.declare_parameter("ambient_duration", 2.0)

        self._lang = self.get_parameter("language").get_parameter_value().string_value
        self._device_idx = self.get_parameter("device_index").get_parameter_value().integer_value
        energy_thresh = self.get_parameter("energy_threshold").get_parameter_value().double_value
        pause_thresh = self.get_parameter("pause_threshold").get_parameter_value().double_value
        self._phrase_lim = self.get_parameter("phrase_time_limit").get_parameter_value().double_value
        dynamic_energy = self.get_parameter("dynamic_energy").get_parameter_value().bool_value
        ambient_duration = self.get_parameter("ambient_duration").get_parameter_value().double_value

        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = energy_thresh
        self._recognizer.pause_threshold = pause_thresh
        self._recognizer.dynamic_energy_threshold = dynamic_energy

        self._log_devices()

        device = self._device_idx if self._device_idx >= 0 else None
        
        try:
            self._mic = sr.Microphone(device_index=device)
        except Exception as e:
            self.get_logger().error(f"마이크 열기 실패: {e}")
            raise

        with self._mic as source:
            self.get_logger().info(f"주변 소음 측정 중 ({ambient_duration:.1f}s)...")
            self._recognizer.adjust_for_ambient_noise(
                source,
                duration=ambient_duration
            )
            self.get_logger().info(
                f"energy_threshold={self._recognizer.energy_threshold:.1f}"
            )

        #-----------------------------------------------------------
        # 기존 STT 결과 전달 client
        self._vlm_client = self.create_client(Stt, "/stt_results")

        # stt_start 서비스 서버
        self._start_srv = self.create_service(
            SttStart,
            "/stt_start",
            self._start_callback,
        )

        self._tts_client = self.create_client(Tts, "/stt_to_tts")      
        
        while not self._tts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("TTS service waiting...")

        # listening 상태 관리
        self._stop_listen = None
        self._is_listening = False

        self._products = [
            "칸쵸",
            "초코파이",
            "펩시",
            "포카리",
            "두유",
            "빼빼로 아몬드",
            # "빼빼로 오리지널",
        ]

        self._alias_map = {
            "do you":               "두유",
            "soy milk":             "두유",
            "우유":                  "두유",
            "fc":                   "펩시",
            "pepsi":                "펩시",
            "pocari":               "포카리",
            "포카리스웨트":            "포카리",
            "보카리":                 "포카리",
            "보카리스웨트":            "포카리",
            "간초":                 "칸쵸",
            "관초":                 "칸쵸",
            "칸초":                 "칸쵸",
            "칸죠":                 "칸쵸",
            "한초":                  "칸쵸",
            # "함초":                  "칸쵸",
            "pepero almond":        "빼빼로",
            "아몬드 빼빼로":           "빼빼로",
            # "pepero original":      "빼빼로 오리지널",
            # "오리지널 빼빼로":          "빼빼로 오리지널"
        }

        # self._ambiguous_products = {
        #     "빼빼로": [
        #         "빼빼로 아몬드",
        #         "빼빼로 오리지널",
        #     ]
        # }
        self._ambiguous_products = {}

        # pending 주문 상태
        self._pending_orders = []
        self._pending_disambiguation = None
        self._pending_confirmation = None

        #상품명 jamo 저장
        self._product_jamo = {}

        for product in self._products:
            self._product_jamo[product] = self._to_jamo(product)

        self.get_logger().info("STT node 준비 완료")

    
    def _log_devices(self):
        self.get_logger().info("=== 마이크 장치 목록 ===")
        for idx, name in enumerate(sr.Microphone.list_microphone_names()):
            mark = " ◀" if idx == self._device_idx else ""
            self.get_logger().info(f"[{idx}] {name}{mark}")

    
    def _start_callback(self, request, response):
        if not request.start:
            response.success = False
            return response

        if self._is_listening:
            self.get_logger().warning("이미 STT listening 중")
            response.success = True
            return response

        try:
            self._stop_listen = self._recognizer.listen_in_background(
                self._mic,
                self._on_audio,
                phrase_time_limit=self._phrase_lim,
            )
            self.get_logger().info(
                f"listening start {time.time()}"
            )

            self._is_listening = True

            self.get_logger().info("STT listening 시작")

            response.success = True

        except Exception as e:
            self.get_logger().error(f"STT 시작 실패: {e}")
            response.success = False

        return response

    
    def _stop_background_listening(self):
        if self._stop_listen:
            try:
                self._stop_listen(wait_for_stop=False)
            except Exception:
                pass

            self._stop_listen = None
            self._is_listening = False


    def _save_parse_failed_log(
        self,
        raw_text,
        best_candidate=None,
        score=None,
        method=None,
    ):
        
        try:

            log_data = {
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "raw": raw_text,
                "best_candidate": best_candidate,
                "score": score,
                "method": method,
            }

            log_path = Path.home() / "team_abc_ws/src/abc_speech/abc_speech/parse_failed_log.jsonl"
            self.get_logger().info(
                f"log path = {log_path}"
            )
            
            with open(
                log_path,
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        log_data,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except Exception as e:
            self.get_logger().error(
                f"로그 저장 실패: {e}"
            )


    def _normalize_stt(self, text: str) -> str:
        text = text.strip().lower()

        # alias exact replace
        for alias, canonical in self._alias_map.items():
            if alias in text:
                text = text.replace(alias, canonical)

        filler_words = [
        "랑",
        "이랑",
        "그리고",
        "주세요",
        "줘",
        "죠",
        "조",
        "하고",
        ]

        for word in filler_words:
            text = text.replace(word, " ")

        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def _to_jamo(self, text: str):
        return j2hcj(h2j(text))

    
    def _jamo_match(self, word: str):

        input_jamo = self._to_jamo(word)

        best_product = None
        best_score = 0

        for product in self._products:

            score = fuzz.ratio(
                input_jamo,
                self._product_jamo[product]
            )

            if score > best_score:
                best_score = score
                best_product = product

        return best_product, best_score


    def _extract_product_and_quantity(self, text: str):
        orders = []

        number_map = {
            "한": 1,
            "하나": 1,
            "1": 1,
            "두": 2,
            "둘": 2,
            "2": 2,
            "세": 3,
            "셋": 3,
            "3": 3,
        }

        remaining = text.strip()

        # 긴 상품명 먼저 찾기
        products = sorted(self._products, key=len, reverse=True)

        while remaining:
            remaining = remaining.strip()

            product = None

            # 1. exact product match
            self.get_logger().info(f"[===exact product match")
            for p in products:
                if p in remaining:
                    product = p
                    self.get_logger().info(f"[product] {product}")
                    remaining = remaining.replace(p, "", 1)
                    break

            # # 2. ambiguous base product
            # self.get_logger().info(f"[===ambiguous base product")
            # if not product:
            #     for base in self._ambiguous_products.keys():
            #         if base in remaining:
            #             quantity = 1

            #             for word, num in number_map.items():
            #                 if word in remaining:
            #                     quantity = num
            #                     break

                        
            #             self.get_logger().info(f"[STATUS] need_disambiguation")
            #             return {
            #                 "status": "need_disambiguation",
            #                 "base_product": base,
            #                 "candidates": self._ambiguous_products[base],
            #                 "orders": orders,
            #                 "quantity": quantity,
            #             }

            best_candidate = None
            best_score = 0
            best_method = None

            # 3. fuzzy match
            self.get_logger().info(f"[===fuzzy match]")
            if not product:
                tokens = remaining.split()

                if not tokens:
                    break

                token = tokens[0]
                self.get_logger().info(f"[product] {token}")
                    
                match = process.extractOne(
                    token,
                    self._products,
                    scorer=fuzz.ratio
                )

                if match:
                    candidate, score, _ = match
                    self.get_logger().info(f"[product, score] {candidate}, {score}")

                    if score > best_score:
                        best_candidate = candidate
                        best_score = score
                        best_method = "fuzzy"

                    if score >= 60:
                        product = candidate
                        remaining = remaining.replace(token, "", 1)

            # 4. jamo match
            if not product:
                tokens = remaining.split()

                if not tokens:
                    break

                token = tokens[0]
                self.get_logger().info(f"[product] {token}")

                candidate, score = self._jamo_match(token)

                if score > best_score:
                    best_candidate = candidate
                    best_score = score
                    best_method = "jamo"

                self.get_logger().info(
                    f"[JAMO] {token} -> {candidate}, {score}"
                )

                if score >= 60:
                    product = candidate

                    remaining = remaining.replace(
                        token,
                        "",
                        1
                    )


            if not product:
                if best_score >= 20:
                    self.get_logger().info(f"[STATUS] parse_failed")
                    return {
                        "status": "parse_failed",
                        "failed_text": remaining,
                        "orders": orders,
                        "best_candidate": best_candidate,
                        "best_score": best_score,
                        "method": best_method,
                    }

                else:
                    self.get_logger().info(f"[STATUS] need_confirmation")
                    return {
                        "status": "need_confirmation",
                        "original_text": text,
                    }

            self.get_logger().info(f"[product] {product}")
                

            # 4. quantity match
            self.get_logger().info(f"[===quantity match]")
            quantity = None

            for word, num in number_map.items():
                if word in remaining:
                    quantity = num
                    remaining = remaining.replace(word, "", 1)
                    remaining = remaining.replace("개", "", 1)
                    break

            if quantity is None:
                quantity = 1

            orders.append((product, quantity))

        self.get_logger().info(f"[STATUS] success")
        return {
            "status": "success",
            "orders": orders,
        }


    # def _handle_pending_disambiguation(self, normalized: str) -> bool:
    #     option_map = self._pending_disambiguation["option_map"]
    #     candidates = self._pending_disambiguation["candidates"]
    #     quantity = self._pending_disambiguation["quantity"]

    #     if normalized in option_map:
    #         selected_product = option_map[normalized]

    #     elif normalized in candidates:
    #         selected_product = normalized

    #     else:
    #         match = process.extractOne(
    #             normalized,
    #             list(option_map.keys()) + candidates,
    #             scorer=fuzz.partial_ratio
    #         )

    #         if not match:
    #             return False

    #         candidate, score, _ = match

    #         if score < 70:
    #             return False

    #         selected_product = option_map.get(candidate, candidate)

    #     self._pending_orders.append((selected_product, quantity))

    #     final_text = " ".join(
    #         f"{product} {qty}개"
    #         for product, qty in self._pending_orders
    #     )

    #     self._pending_orders = []
    #     self._pending_disambiguation = None

    #     self._send_order(final_text)

    #     return True
    
    
    def _handle_confirmation_request(self, result):

        self._pending_confirmation = result["original_text"]

        tts_text = (
            f"{result['original_text']}라고 말씀하신 게 맞나요?"
        )

        self.get_logger().info(
            f"[CONFIRMATION] {tts_text}"
        )

        self._stop_background_listening()
        self._send_tts(tts_text)


    # def _handle_disambiguation_request(self, result):
    #     self._pending_orders = result["orders"]

    #     base_product = result["base_product"]

    #     option_map = {
    #         candidate.replace(base_product, "").strip(): candidate
    #         for candidate in result["candidates"]
    #     }

    #     self._pending_disambiguation = {
    #         "base_product": base_product,
    #         "option_map": option_map,
    #         "candidates": result["candidates"],
    #         "quantity": result["quantity"],
    #     }

    #     candidate_names = [
    #         item.replace(base_product, "").strip()
    #         for item in result["candidates"]
    #     ]

    #     tts_text = (
    #         f"{', '.join(candidate_names)} 중 "
    #         f"어떤 {base_product}를 원하시나요?"
    #     )

    #     self.get_logger().info(f"[TTS 질문] {tts_text}")

    #     self._stop_background_listening()
    #     self._send_tts(tts_text)


    def _send_order(self, final_text: str):
        self.get_logger().info(f"[최종 주문] {final_text}")

        self._stop_background_listening()

        req = Stt.Request()
        req.raw_text = final_text

        future = self._vlm_client.call_async(req)
        future.add_done_callback(self._service_response_callback)


    def _send_tts(self, text: str):
        req = Tts.Request()
        req.text = text

        future = self._tts_client.call_async(req)
        future.add_done_callback(self._tts_response_callback)
    
    
    def _tts_response_callback(self, future):
        try:
            response = future.result()

            if response.success:
                self.get_logger().info("TTS request 성공")

                if not self._is_listening:
                    self._stop_listen = self._recognizer.listen_in_background(
                        self._mic,
                        self._on_audio,
                        phrase_time_limit=self._phrase_lim,
                    )
                    self._is_listening = True
                    self.get_logger().info("STT listening 시작")

            else:
                self.get_logger().warning("TTS request 실패")

        except Exception as e:
            self.get_logger().error(f"TTS service call 실패: {e}")  


    def _on_audio(self, recognizer, audio):
        self.get_logger().info(
            f"audio callback {time.time()}"
        )
        
        try:
            text = recognizer.recognize_google(
                audio,
                language=self._lang
            ).strip()

            if not text:
                return

            # self.get_logger().info(f"[STT 결과] {text}")

            # # 한 번 인식했으면 listening 중지
            # self._stop_background_listening()

            # req = Stt.Request()
            # req.raw_text = text
            self.get_logger().info(f"[STT 원본] {text}")

            normalized = self._normalize_stt(text)

            if self._pending_confirmation:

                normalized = normalized.strip()
                self.get_logger().info(
                    f"[normalized] '{normalized}'"
                )

                yes_words = [
                    "네",
                    "예",
                    "맞아",
                    "맞습니다",
                    "응",
                    "어",
                ]

                no_words = [
                    "아니",
                    "아니요",
                    "틀려",
                ]

                if any(word in normalized for word in no_words):

                    self._pending_confirmation = None

                    self._stop_background_listening()
                    self._send_tts(
                        "주문을 다시 말씀해주세요."
                    )
                    
                    return
            
                elif any(word in normalized for word in yes_words):

                    self.get_logger().info(
                        f"[VLM 주문] {self._pending_confirmation}"
                    )

                    req = Stt.Request()
                    req.raw_text = self._pending_confirmation

                    future = self._vlm_client.call_async(req)
                    future.add_done_callback(
                        self._service_response_callback
                    )

                    self._pending_confirmation = None

                    self._stop_background_listening()

                    return

                else:
                    self.get_logger().warning(
                        "confirmation 응답 대기 중"
                    )
                    # pending 상태 유지
                    return
            
            
            # if self._pending_disambiguation:
            #     handled = self._handle_pending_disambiguation(normalized)
            #     if handled:
            #         return

            #     self.get_logger().warning("후보 선택 실패")

            #     self._stop_background_listening()
            #     self._send_tts("다시 말씀해주세요.")
            #     return
            
            result = self._extract_product_and_quantity(normalized)
            
            if result["status"] == "success":
                orders = result["orders"]

                if orders:
                    final_text = " ".join(
                        f"{product} {quantity}개"
                        for product, quantity in orders
                    )
                    self._send_order(final_text)

                #상품 추출 실패
                else:
                    self._stop_background_listening()
                    self._send_tts("상품을 다시 말씀해주세요.")
                    return
            
            elif result["status"] == "parse_failed":

                self._save_parse_failed_log(
                   raw_text=text,
                   best_candidate=result.get("best_candidate"),
                   score=result.get("best_score"),
                   method=result.get("method"),
                )           
                
                failed_product = result["failed_text"]

                self.get_logger().warning(
                    f"주문 파싱 실패: {result['failed_text']}"
                )

                self._stop_background_listening()

                self._send_tts(
                    f"{failed_product}는 등록되지 않은 상품입니다. "
                    "전체 주문을 다시 말씀해주세요."
                )

                return

            elif result["status"] == "need_confirmation":

                self._handle_confirmation_request(
                    result
                )
                                        
            # elif result["status"] == "need_disambiguation":
            #     self._handle_disambiguation_request(result)
                

        except sr.UnknownValueError:
            self.get_logger().warning("음성을 인식하지 못했습니다.")

        except sr.RequestError as e:
            self.get_logger().error(f"Google STT 요청 실패: {e}")

        except Exception as e:
            self.get_logger().error(f"STT 실패: {e}")

    
    def _service_response_callback(self, future):
        try:
            response = future.result()

            if response.success:
                self.get_logger().info("service request 성공")
            else:
                self.get_logger().warning("service request 실패")

        except Exception as e:
            self.get_logger().error(f"service call 실패: {e}")

    
    def destroy_node(self):
        self._stop_background_listening()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = SttNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
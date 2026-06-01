import os
import tempfile
import threading

import rclpy
from rclpy.node import Node

from gtts import gTTS
from abc_interfaces.srv import UserRequest, SttStart, Tts


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")

        self._vlm_srv = self.create_service(
            UserRequest,
            "/vlm_to_tts",
            self.vlm_callback,
        )

        # STT 서비스 클라이언트 추가
        self.stt_client = self.create_client(
            SttStart,
            "/stt_start",
        )

        self._stt_srv = self.create_service(
            Tts,
            "/stt_to_tts",
            self.stt_callback,
        )

        while not self.stt_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/stt_start 서비스 대기 중...")

        # 키보드 입력 스레드 추가
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_listener,
            daemon=True,
        )
        self.keyboard_thread.start()

        self.get_logger().info("TTS service 준비 완료 (/vlm_to_tts)")

    def play_tts(self, text):
        """TTS 재생 공통 함수"""
        tts = gTTS(
            text=text,
            lang="ko",
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False,
        ) as fp:
            temp_path = fp.name

        tts.save(temp_path)

        os.system(f"mpg123 {temp_path} > /dev/null 2>&1")

        os.remove(temp_path)

    def trigger_stt(self):
        """STT 시작 서비스 호출"""
        req = SttStart.Request()
        req.start = True

        future = self.stt_client.call_async(req)

        def callback(fut):
            try:
                result = fut.result()
                if result.success:
                    self.get_logger().info("STT 시작 성공")
                else:
                    self.get_logger().warning("STT 시작 실패")
            except Exception as e:
                self.get_logger().error(f"STT 서비스 호출 실패: {e}")

        future.add_done_callback(callback)

    def keyboard_listener(self):
        """키보드 입력 감지"""
        while rclpy.ok():
            key = input()

            if key.lower() == "s":
                self.get_logger().info("'s' 입력 감지")

                self.play_tts("주문을 시작해주세요")
                self.trigger_stt()

    def vlm_callback(self, request, response):
        class_names = request.class_name
        iterations = request.iteration

        if len(class_names) != len(iterations):
            msg = "class_name과 iteration 길이가 다릅니다."
            self.get_logger().error(msg)

            response.success = False
            response.message = msg
            return response

        if not class_names:
            msg = "[TTS 재생 완료] 요청된 상품이 없습니다."
            self.get_logger().warning(msg)
            self.play_tts("요청된 상품이 없습니다")

            # 재주문 위해 STT 다시 시작
            self.trigger_stt()

            response.success = True
            response.message = msg
            return response

        try:
            items = []

            for name, count in zip(class_names, iterations):
                items.append(f"{name} 상품이 {count}개")

            items_text = ", ".join(items)

            full_text = (
                f"{items_text} 있어 재고가 부족합니다. "
                f"해당 상품만 재주문 해주세요."
            )

            self.get_logger().info(f"[TTS 요청] {full_text}")

            self.play_tts(full_text)

            # 재주문 위해 STT 다시 시작
            self.trigger_stt()

            response.success = True
            response.message = "[TTS 재생 완료] 재고가 부족합니다."

        except Exception as e:
            error_msg = f"TTS 실패: {e}"
            self.get_logger().error(error_msg)

            response.success = False
            response.message = error_msg

        return response
    
    #----------------------------------------------------
    def stt_callback(self, request, response):
        try:
            self.get_logger().info(f"[STT 요청] {request.text}")

            self.play_tts(request.text)

            response.success = True

        except Exception as e:
            error_msg = f"Ambiguous TTS 실패: {e}"
            self.get_logger().error(error_msg)

            response.success = False

        return response


def main(args=None):
    rclpy.init(args=args)

    node = TtsNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
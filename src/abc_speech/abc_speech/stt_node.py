import rclpy
from rclpy.node import Node

import speech_recognition as sr
from abc_interfaces.srv import Stt, SttStart


class SttNode(Node):

    def __init__(self):
        super().__init__("stt_node")

        self.declare_parameter("language", "ko-KR")
        self.declare_parameter("device_index", -1)
        self.declare_parameter("energy_threshold", 50.0)
        self.declare_parameter("pause_threshold", 0.8)
        self.declare_parameter("phrase_time_limit", 5.0)
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

        # 기존 STT 결과 전달 client
        self._client = self.create_client(Stt, "/stt_results")

        # while not self._client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info("STT result service waiting...")

        # stt_start 서비스 서버
        self._start_srv = self.create_service(
            SttStart,
            "/stt_start",
            self._start_callback,
        )

        # listening 상태 관리
        self._stop_listen = None
        self._is_listening = False

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

    def _on_audio(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(
                audio,
                language=self._lang
            ).strip()

            if not text:
                return

            self.get_logger().info(f"[STT 결과] {text}")

            # 한 번 인식했으면 listening 중지
            self._stop_background_listening()

            req = Stt.Request()
            req.raw_text = text

            future = self._client.call_async(req)
            future.add_done_callback(self._service_response_callback)

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
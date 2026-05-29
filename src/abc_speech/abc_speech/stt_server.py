import rclpy
from rclpy.node import Node

from abc_interfaces.srv import Stt, UserRequest


class TestSttResultServer(Node):

    def __init__(self):
        super().__init__("test_stt_result_server")

        # /stt_result server
        self._srv = self.create_service(
            Stt,
            "/stt_results",
            self.stt_callback,
        )

        # /vlm_to_tts client
        self._tts_client = self.create_client(
            UserRequest,
            "/vlm_to_tts",
        )

        self.get_logger().info("test stt_result server 준비 완료")

    def stt_callback(self, request, response):
        self.get_logger().info(f"[STT 입력] {request.raw_text}")

        if not self._tts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("/vlm_to_tts 서비스 없음")

            response.success = False
            return response

        tts_req = UserRequest.Request()

        # 테스트용 고정 데이터
        tts_req.class_name = [
            "칸초",
            "초코파이",
        ]

        tts_req.iteration = [
            1,
            2,
        ]

        future = self._tts_client.call_async(tts_req)
        future.add_done_callback(self.tts_response_callback)

        response.success = True
        return response

    def tts_response_callback(self, future):
        try:
            result = future.result()

            if result.success:
                self.get_logger().info(
                    f"TTS 성공: {result.message}"
                )
            else:
                self.get_logger().warning(
                    f"TTS 실패: {result.message}"
                )

        except Exception as e:
            self.get_logger().error(f"TTS 호출 실패: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TestSttResultServer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
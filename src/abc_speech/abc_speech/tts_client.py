import rclpy
from rclpy.node import Node

from abc_interfaces.srv import UserRequest


class TtsTestClient(Node):
    def __init__(self):
        super().__init__("tts_test_client")

        self._client = self.create_client(
            UserRequest,
            "/vlm_to_tts"
        )

        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("TTS service waiting...")

        self.send_request()

    def send_request(self):
        req = UserRequest.Request()

        req.class_name = [
            "초코파이",
            "콜라",
            "포카리스웨트",
        ]

        req.iteration = [
            1,
            2,
            3,
        ]

        self.get_logger().info("TTS request 전송")

        future = self._client.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()

            self.get_logger().info(
                f"success={response.success}, "
                f"message={response.message}"
            )

        except Exception as e:
            self.get_logger().error(f"service call 실패: {e}")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = TtsTestClient()

    rclpy.spin(node)


if __name__ == "__main__":
    main()
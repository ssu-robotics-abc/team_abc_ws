import rclpy
from rclpy.node import Node

from abc_interfaces.srv import UserRequest


class TtsTestClient(Node):
    def __init__(self):
        super().__init__("tts_test_client")
        self.cli = self.create_client(UserRequest, "/vlm_to_tts")
        self.get_logger().info("/vlm_to_tts 서비스 대기 중...")

        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/vlm_to_tts 서비스가 준비되지 않았습니다.")
            raise RuntimeError("/vlm_to_tts service not available")

        self.get_logger().info("/vlm_to_tts 서비스 준비 완료")

    def send_request(self, class_names, iterations):
        req = UserRequest.Request()
        req.class_name = class_names
        req.iteration = iterations

        self.get_logger().info(
            f"요청 전송 class_name={class_names}, iteration={iterations}"
        )

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.done():
            res = future.result()
            self.get_logger().info(
                f"응답 success={res.success}, message='{res.message}'"
            )
            return res

        self.get_logger().error("/vlm_to_tts 서비스 호출 타임아웃")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = TtsTestClient()

    try:
        node.send_request([], [])
        # node.send_request(["pepsi", "chocopie"], [0, 1])
    finally:
        node.destroy_node()
        rclpy.shutdown()

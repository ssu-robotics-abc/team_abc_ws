import os
import tempfile

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from gtts import gTTS


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")

        self.create_subscription(
            String,
            "/tts_text",
            self.tts_callback,
            10,
        )

        self.get_logger().info("TTS node 준비 완료 (/tts_text)")

    def tts_callback(self, msg: String):
        text = msg.data.strip()

        if not text:
            self.get_logger().warning("빈 TTS 메시지 수신")
            return

        self.get_logger().info(f"[TTS 요청] {text}")

        try:
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

        except Exception as e:
            self.get_logger().error(f"TTS 실패: {e}")


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
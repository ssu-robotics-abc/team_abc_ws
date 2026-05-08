#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

class VlmCommandPublisher(Node):
    def __init__(self):
        super().__init__("vlm_command_publisher")
        # VLM 노드와 통신할 토픽 생성
        self.publisher_ = self.create_publisher(String, "/vlm_user_command", 10)

    def publish_command(self, user_input: str):
        msg = String()
        msg.data = user_input
        self.publisher_.publish(msg)
        self.get_logger().info(f"로봇에 명령 전송 완료: '{user_input}'")

def main(args=None):
    rclpy.init(args=args)
    node = VlmCommandPublisher()

    # 노드의 ROS 통신(콜백 등)을 백그라운드에서 처리하기 위한 스레드
    # 퍼블리셔만 있다면 필수적이지는 않지만, 확장을 위해 관행적으로 넣어줍니다.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n" + "="*50)
    print(" 🎙️ VLM 로봇 제어 터미널이 시작되었습니다.")
    print(" 원하는 물건을 텍스트로 입력하세요 (예: 칸쵸 찾아줘)")
    print(" 종료하시려면 'exit' 또는 'quit'을 입력하세요.")
    print("="*50 + "\n")

    try:
        while rclpy.ok():
            try:
                user_input = input("명령 입력 >> ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("명령 입력을 종료합니다.")
                    break
                
                if user_input:
                    node.publish_command(user_input)
                    
            except EOFError:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
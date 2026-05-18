import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from abc_interfaces.srv import Stt
from abc_interfaces.srv import ParseOrder
from abc_interfaces.srv import CheckInventory
from abc_interfaces.msg import OrderList


class OrderManager(Node):
    def __init__(self):
        super().__init__("order_manager")

        self.is_processing = False

        self.tts_pub = self.create_publisher(String, "/tts_text", 10)
        self.vlm_pub = self.create_publisher(
            OrderList,
            "/confirmed_order",
            10,
        )
        
        self.stt_client = self.create_client(Stt, "/stt")
        self.parse_client = self.create_client(ParseOrder, "/parse_order")
        self.inventory_client = self.create_client(CheckInventory, "/inventory/check")

        self._wait_for_services()

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True,
        )
        self.keyboard_thread.start()

        self.get_logger().info("OrderManager 준비 완료")
        self.get_logger().info("s 입력 후 Enter를 누르면 주문을 시작합니다.")

    def _wait_for_services(self):
        services = [
            (self.stt_client, "/stt"),
            (self.parse_client, "/parse_order"),
            # (self.inventory_client, "/inventory/check"),
        ]

        for client, name in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"{name} service 대기 중...")

    # def keyboard_loop(self):
    #     while rclpy.ok():
    #         cmd = input("주문 시작: s 입력 후 Enter > ").strip().lower()

    #         self.get_logger().info(f"입력값=[{cmd}]")

    #         if cmd == "s":
    #             if self.is_processing:
    #                 self.get_logger().warning("이미 주문 처리 중입니다.")
    #                 continue

    #             self.is_processing = True
    #             self.start_order_flow()

    #         elif cmd == "q":
    #             self.get_logger().info("종료합니다.")
    #             rclpy.shutdown()
    #             break
    def keyboard_loop(self):
        while rclpy.ok():
            cmd = input("주문 시작: s 입력 후 Enter > ")

            print(f"RAW=[{repr(cmd)}]", flush=True)

            cmd = cmd.strip().lower()

            print(f"PARSED=[{repr(cmd)}]", flush=True)

            if cmd == "s":
                print("S DETECTED", flush=True)

                if self.is_processing:
                    print("already processing", flush=True)
                    continue

                self.is_processing = True
                self.start_order_flow()

    def publish_tts(self, text: str):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f"[TTS 요청] {text}")

    def start_order_flow(self):
        self.publish_tts("주문을 말씀해주세요.")

        req = Stt.Request()
        # req.raw_text = "주문을 말씀해주세요."
        req.raw_text = "초코파이 하나 줘"

        future = self.stt_client.call_async(req)
        future.add_done_callback(self.on_stt_response)

    def on_stt_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"STT service 호출 실패: {e}")
            self.publish_tts("음성 인식 중 오류가 발생했습니다.")
            self.is_processing = False
            return

        if not response.success:
            self.publish_tts("음성을 인식하지 못했습니다. 다시 주문해주세요.")
            self.is_processing = False
            return

        recognized_text = response.recognized_text.strip()
        self.get_logger().info(f"[STT 결과] {recognized_text}")

        parse_req = ParseOrder.Request()
        parse_req.raw_text = recognized_text

        future = self.parse_client.call_async(parse_req)
        future.add_done_callback(self.on_parse_response)

    def on_parse_response(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"ParseOrder service 호출 실패: {e}")
            self.publish_tts("주문 분석 중 오류가 발생했습니다.")
            self.is_processing = False
            return

        if not response.parse_success or len(response.items) == 0:
            self.publish_tts("주문 내용을 이해하지 못했습니다. 다시 주문해주세요.")
            self.is_processing = False
            return

        self.order_items = list(response.items)
        self.inventory_results = []
        self.current_index = 0

        self.check_next_inventory()

    def check_next_inventory(self):
        if self.current_index >= len(self.order_items):
            self.finish_inventory_check()
            return

        item = self.order_items[self.current_index]

        req = CheckInventory.Request()
        req.item_name = item.product_id
        req.quantity = item.quantity

        self.get_logger().info(
            f"[재고 확인 요청] product_id={item.product_id}, quantity={item.quantity}"
        )

        # future = self.inventory_client.call_async(req)
        # future.add_done_callback(self.on_inventory_response)

    # def on_inventory_response(self, future):
    #     item = self.order_items[self.current_index]

    #     try:
    #         response = future.result()
    #     except Exception as e:
    #         self.get_logger().error(f"Inventory service 호출 실패: {e}")
    #         self.publish_tts("재고 확인 중 오류가 발생했습니다.")
    #         self.is_processing = False
    #         return

    #     self.inventory_results.append({
    #         "product_id": item.product_id,
    #         "quantity": item.quantity,
    #         "available": response.available,
    #         "current_quantity": response.current_quantity,
    #         "message": response.message,
    #     })

    #     self.current_index += 1
    #     self.check_next_inventory()

    # def finish_inventory_check(self):
    #     unavailable_items = [
    #         item for item in self.inventory_results
    #         if not item["available"]
    #     ]

    #     if unavailable_items:
    #         messages = []

    #         for item in unavailable_items:
    #             messages.append(
    #                 f"{item['product_id']}는 현재 {item['current_quantity']}개 남아 있어 "
    #                 f"{item['quantity']}개 주문이 불가합니다."
    #             )

    #         self.publish_tts(" ".join(messages) + " 다시 주문해주세요.")
    #         self.is_processing = False
    #         return

    #     order_summary = ", ".join(
    #         [
    #             f"{item['product_id']} {item['quantity']}개"
    #             for item in self.inventory_results
    #         ]
    #     )

    #     self.publish_tts(f"{order_summary} 주문이 확인되었습니다.")

    #     self.publish_confirmed_order()

    #     self.is_processing = False

    # def publish_confirmed_order(self):
    #     msg = OrderList()
    #     msg.items = self.order_items

    #     self.vlm_pub.publish(msg)

    #     self.get_logger().info(
    #         "[VLM 전달] 최종 주문 publish 완료"
    #     )

def main(args=None):
    rclpy.init(args=args)
    node = OrderManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
import os
import json

import rclpy
from rclpy.node import Node

from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory
import os
import google.generativeai as genai

from abc_interfaces.srv import ParseOrder
from abc_interfaces.msg import OrderItem


TARGET_CLASSES = [
    "Kancho",
    "pepero_original",
    "pepsi",
    "pocarisweat",
    "soy_milk",
    "chocopie",
    "pepero_almond",
]


def parse_order_with_gemini(raw_text: str):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    prompt = f"""
너는 주문 파싱 AI다.

사용자의 주문 문장을 보고 상품명과 수량을 추출해라.

허용 가능한 상품 목록:
{TARGET_CLASSES}

반드시 JSON 배열만 출력해라.
설명 금지.
마크다운 금지.

형식:
[
  {{"product_id": "chocopie", "quantity": 2}},
  {{"product_id": "pepsi", "quantity": 1}}
]

규칙:
- 상품명이 없으면 빈 배열
- 수량 없으면 1
- 반드시 product_id는 위 리스트 중 하나
- 한국어 상품명을 영문 product_id로 변환

사용자 주문:
{raw_text}
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0
        ),
    )

    text = response.text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(text)

    return parsed


class ParseOrderServer(Node):
    def __init__(self):
        super().__init__("parse_order_server")

        load_dotenv()

        self.create_service(
            ParseOrder,
            "/parse_order",
            self.callback,
        )

        self.get_logger().info("/parse_order service ready")

    def callback(self, request, response):
        raw_text = request.raw_text

        self.get_logger().info(f"[Parse 요청] {raw_text}")

        try:
            # 테스트용 mock parser
            parsed_items = [
                {
                    "product_id": "chocopie",
                    "quantity": 1,
                }
            ]

            ros_items = []

            for item in parsed_items:
                msg = OrderItem()
                msg.product_id = item["product_id"]
                msg.quantity = item["quantity"]
                ros_items.append(msg)

            response.parse_success = True
            response.items = ros_items

            self.get_logger().info(
                f"[Parse 성공] {len(ros_items)} items"
            )

        except Exception as e:
            response.parse_success = False
            response.items = []

            self.get_logger().error(f"Parse 실패: {e}")

        return response

    # def callback(self, request, response):
    #     raw_text = request.raw_text

    #     self.get_logger().info(f"[Parse 요청] {raw_text}")

    #     try:
    #         parsed_items = parse_order_with_gemini(raw_text)

    #         ros_items = []

    #         for item in parsed_items:
    #             msg = OrderItem()
    #             msg.product_id = item["product_id"]
    #             msg.quantity = item["quantity"]
    #             ros_items.append(msg)

    #         response.parse_success = True
    #         response.items = ros_items

    #         self.get_logger().info(
    #             f"[Parse 성공] {len(ros_items)} items"
    #         )

    #     except Exception as e:
    #         response.parse_success = False
    #         response.items = []

    #         self.get_logger().error(f"Parse 실패: {e}")

    #     return response


def main(args=None):
    rclpy.init(args=args)

    node = ParseOrderServer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
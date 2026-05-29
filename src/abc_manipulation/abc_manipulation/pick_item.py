import os
import rclpy
import threading
import numpy as np
from scipy.spatial.transform import Rotation

from abc_manipulation.realsense import ImgNode
from abc_manipulation.onrobot import RG
from ament_index_python.packages import get_package_share_directory
import DR_init

from abc_interfaces.msg import DetectionArray
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from abc_interfaces.action import PickItem

# ======================
# 1. 로봇 및 ROS2 초기 설정
# ======================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

rclpy.init()
node = rclpy.create_node("dsr_auto_pick", namespace=ROBOT_ID)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = node

try:
    from DSR_ROBOT2 import get_current_posx, movej, movel, wait
    from DR_common2 import posx, posj
except ImportError:
    print("두산 로봇 라이브러리 로드 실패")
    exit()

# ======================
# 2. 수평 집기 파라미터 설정
# ======================
X_OFFSET = 185.0            # 그리퍼 중심 보정 (mm)
SIDE_APPROACH_DIST = 100.0  # 물체 정면 대기 거리 (mm)
SAFE_Z = 400.0              # 이동 안전 높이
SQUEEZE_RATIO = 0.95        # 파지 보정 (박스 재질 고려)

class PickItemServer(Node):
    def __init__(self):
        super().__init__('pick_item_server')
        cb_group = ReentrantCallbackGroup()

        # 이미지 노드 초기화 및 인트린식(카메라 고유 파라미터) 대기
        self.img_node = ImgNode()
        self.get_logger().info("카메라 및 비전 파라미터 대기 중")
        while rclpy.ok() and self.img_node.get_camera_intrinsic() is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)

        self.intrinsics = self.img_node.get_camera_intrinsic()
        
        # 그리퍼-카메라 변환 행렬 로드
        package_path = get_package_share_directory("abc_manipulation")
        gripper2cam_file_path = os.path.join(package_path, 'T_gripper2camera.npy')
        
        if not os.path.exists(gripper2cam_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gripper2cam_file_path = os.path.join(current_dir, "T_gripper2camera.npy")
            
        self.gripper2cam = np.load(gripper2cam_file_path)
        self.gripper = RG("rg2", "192.168.1.1", 502)

        self.JReady = posj([0, 0, 135, 0, -45, 90])
        self.home_pose = None

        # 액션 서버 생성 (태스크 플래너의 요청을 받을 창구)
        self._action_server = ActionServer(
            self,
            PickItem,
            'pick_item',
            self.execute_callback,
            callback_group=cb_group
        )
        self.get_logger().info("🚀 pick_item 서버가 준비되었습니다. 플래너의 명령을 기다립니다.")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, capture_pose):
        coord = np.append(np.array(camera_coords, dtype=float), 1.0)
        base2gripper = self.get_robot_pose_matrix(*capture_pose)
        base2cam = base2gripper @ self.gripper2cam
        td_coord = base2cam @ coord
        return td_coord[:3]

    def execute_callback(self, goal_handle):
        """ 태스크 플래너로부터 목표 픽셀 좌표를 받아 실행되는 핵심 콜백 """
        # 1) 플래너가 보내준 Goal 데이터 꺼내기
        cx = int(goal_handle.request.center_x)
        cy = int(goal_handle.request.center_y)
        w = goal_handle.request.width
        
        self.get_logger().info(f"[액션 수신] 픽셀 좌표 기반 피킹 개시 -> 중심:({cx}, {cy}), 너비:{w}")

        # 2) 피드백 발행: 좌표 변환 단계
        feedback_msg = PickItem.Feedback()
        feedback_msg.state = "받은 픽셀 좌표를 기반으로 3D 위치 계산 중"
        goal_handle.publish_feedback(feedback_msg)

        # 3) 최신 뎁스 프레임 읽기 및 3D 변환 수행
        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            self.get_logger().error("❌ 뎁스 이미지를 가져올 수 없습니다.")
            goal_handle.abort()
            return PickItem.Result(success=False)

        z_dist = depth_frame[cy, cx]
        if z_dist == 0:
            self.get_logger().error("❌ 선택된 좌표의 뎁스(거리) 값이 0입니다. 파지를 취소합니다.")
            goal_handle.abort()
            return PickItem.Result(success=False)

        # 카메라 기준 좌표 -> 로봇 베이스 기준 3D 좌표 변환
        capture_pose = get_current_posx()[0]
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]
        
        cam_coords = ((cx - ppx) * z_dist / fx, (cy - ppy) * z_dist / fy, z_dist)
        base_xyz = self.transform_to_base(cam_coords, capture_pose)

        # 실제 물체 두께 기반 그리퍼 목표 너비(mm) 계산
        obj_width_mm = (w * z_dist) / fx
        target_w = int(obj_width_mm * 10 * SQUEEZE_RATIO)
        target_w = max(0, min(target_w, 1100))

        self.get_logger().info(f"🎯 변환 완료 -> 베이스 좌표: {base_xyz} | 예상 두께: {obj_width_mm:.1f}mm")

        # 4) 로봇 실제 구동 (블로킹 방식으로 순차 실행)
        try:
            self.execute_pick_motion(*base_xyz, target_w, goal_handle)
            
            # 모든 동작 성공적 종료 시
            goal_handle.succeed()
            self.get_logger().info("✅ >>> 피킹 작업 성공 및 플래너에 완료 보고")
            return PickItem.Result(success=True)
            
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 구동 중 에러 발생: {e}")
            goal_handle.abort()
            return PickItem.Result(success=False)

    def execute_pick_motion(self, x, y, z, target_width, goal_handle):
        """ 실제 로봇과 그리퍼를 움직이는 시퀀스 """
        cur = get_current_posx()[0]
        fixed_rx, fixed_ry, fixed_rz = cur[3:]
        hx, hy, hz, hrx, hry, hrz = self.home_pose
        
        self.gripper.open_gripper()
        feedback_msg = PickItem.Feedback()

        # Step 1. 접근 대기 (X축 후방 대기)
        feedback_msg.state = "물품 정면(접근 위치)으로 이동 중"
        goal_handle.publish_feedback(feedback_msg)
        
        wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
        movel(posx([wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        wait(0.5)

        # Step 2. 진입 (수평 찌르기)
        feedback_msg.state = "그립 처리를 위해 물품으로 진입 중"
        goal_handle.publish_feedback(feedback_msg)
        
        pick_x = x - X_OFFSET
        movel(posx([pick_x, y, z, fixed_rx, fixed_ry, fixed_rz]), 20, 20)
        
        # Step 3. 파지
        feedback_msg.state = f"물품 파지 중 (목표 너비: {target_width/10:.1f}mm)"
        goal_handle.publish_feedback(feedback_msg)
        
        self.gripper.move_gripper(width_val=target_width)
        wait(1.2)

        # Step 4. 후퇴 (뒤로 빠지기)
        feedback_msg.state = "파지 완료 후 후방으로 후퇴 중"
        goal_handle.publish_feedback(feedback_msg)
        movel(posx([wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)

        # Step 5. 상승 및 홈 이동
        feedback_msg.state = "안전 높이 및 홈 위치로 복귀 중"
        goal_handle.publish_feedback(feedback_msg)
        movel(posx([wait_x, y, SAFE_Z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        movel(posx([hx, hy, SAFE_Z, hrx, hry, hrz]), VELOCITY, ACC)
        
        # Step 6. 최종 초기화 및 대기 자세
        wait(1.0)
        movej(self.JReady, VELOCITY, ACC)

    def run(self):
        """ 로봇 초기 자세 세팅 및 ROS2 멀티스레드 스핀 가동 """
        self.get_logger().info("로봇 초기 위치(JReady) 설정 중")
        movej(self.JReady, VELOCITY, ACC)
        wait(1.0)
        self.home_pose = get_current_posx()[0]
        self.gripper.open_gripper()
        self.get_logger().info("🤖 로봇 준비 완료. 플래너 신호를 수신할 수 있습니다.")

        # 메인 스레드에서 익스큐터를 직접 돌려 노드를 상시 켜둡니다.
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.add_node(self.img_node)
        executor.spin()

def main():
    try:
        server = PickItemServer()
        server.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
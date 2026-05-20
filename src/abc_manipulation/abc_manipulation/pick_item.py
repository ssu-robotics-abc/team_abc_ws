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

# 클라이언트의 product_id와 비전의 클래스 이름 매핑 정보(임시)
PRODUCT_MAP = {
    1234: 'chocopie',
    5678: 'pepsi',
    2345: 'pocarisweat',
    # 임시 물품들
}

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

        self.img_node = ImgNode()
        print("카메라 및 비전 파라미터 대기 중...")
        while rclpy.ok() and self.img_node.get_camera_intrinsic() is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)

        self.intrinsics = self.img_node.get_camera_intrinsic()
        
        # [표준 방식] 그리퍼-카메라 변환 행렬 로드
        package_path = get_package_share_directory("abc_manipulation")
        gripper2cam_file_path = os.path.join(package_path, 'T_gripper2camera.npy')
        
        if not os.path.exists(gripper2cam_file_path):
            # 빌드 전 src 폴더에서 찾기 위한 예외 처리
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gripper2cam_file_path = os.path.join(current_dir, "T_gripper2camera.npy")
            
        self.gripper2cam = np.load(gripper2cam_file_path)
        self.gripper = RG("rg2", "192.168.1.1", 502)

        self.JReady = posj([0, 0, 135, 0, -45, 90])
        self.home_pose = None
        self.target_object = None
        self.is_picking = False
        self.action_done_event = threading.Event()

        # 액션 서버 생성
        self._action_server = ActionServer(
            self,
            PickItem,
            'pick_item',
            self.execute_callback,
            callback_group=cb_group
        )
        self.get_logger().info("pick_item 서버가 시작되었습니다.")

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
        target_id = goal_handle.request.product_id
        target_name = PRODUCT_MAP.get(target_id)

        if not target_name:
            self.get_logger().error("알 수 없는 물품 ID입니다.")
            goal_handle.abort()
            return PickItem.Result(success=False)

        self.get_logger().info(f"[액션 수신] 탐색 목표 설정: {target_name}")
        
        self.action_done_event.clear()

        # 물품 이름 전달
        self.target_object = target_name

        # 파지 작업 완료 대기
        self.action_done_event.wait() 

        # 성공 처리
        goal_handle.succeed()
        self.get_logger().info(">>> 작업 완료 및 통신 성공")
        return PickItem.Result(success=True)

    def pick_and_place(self, x, y, z, target_width, target_name):
        cur = get_current_posx()[0]
        fixed_rx, fixed_ry, fixed_rz = cur[3:]
        print(f"\n>>> {target_name} 수평 집기 수행")
        
        hx, hy, hz, hrx, hry, hrz = self.home_pose
        self.gripper.open_gripper()
        
        # 1. 접근 대기 (X축 후방 대기)
        wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
        movel(posx([wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        wait(0.5)

        # 2. 진입 (수평 찌르기)
        pick_x = x - X_OFFSET
        movel(posx([pick_x, y, z, fixed_rx, fixed_ry, fixed_rz]), 20, 20)
        
        # 3. 파지
        print(f"[파지] 목표 너비: {target_width/10:.1f}mm")
        self.gripper.move_gripper(width_val=target_width)
        wait(1.2)

        # 4. 후퇴 (뒤로 빠지기)
        movel(posx([wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)

        # 5. 상승 및 홈 이동
        movel(posx([wait_x, y, SAFE_Z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        movel(posx([hx, hy, SAFE_Z, hrx, hry, hrz]), VELOCITY, ACC)
        
        # 6. 초기화
        wait(1.0)
        movej(self.JReady, VELOCITY, ACC)
        self.is_picking = False
        print(">>> 작업 완료.")

        self.action_done_event.set()

    def run(self):
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self)
        executor.add_node(self.img_node)
        threading.Thread(target=executor.spin, daemon=True).start()

        movej(self.JReady, VELOCITY, ACC)
        wait(1.0)
        self.home_pose = get_current_posx()[0]
        self.gripper.open_gripper()

        print("루프 시작 (YOLO 노드를 먼저 실행해주세요)...")
        while True:
            img_frame = self.img_node.get_color_frame()
            depth_frame = self.img_node.get_depth_frame()
            detection_msg = self.img_node.get_latest_detection_msg()

            # 타겟 물체가 지정되었고, 현재 작업 중이 아닐 때만 실행
            if self.target_object is not None and not self.is_picking and detection_msg is not None:
                for obj in detection_msg.detections:
                    if obj.class_name == self.target_object and obj.confidence >= 0.6:
                        cx, cy = int(obj.center_x), int(obj.center_y)
                        z_dist = depth_frame[cy, cx]
                        if z_dist == 0: 
                            continue

                        capture_pose = get_current_posx()[0]
                        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                        ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]
                        
                        cam_coords = ((cx - ppx) * z_dist / fx, (cy - ppy) * z_dist / fy, z_dist)
                        base_xyz = self.transform_to_base(cam_coords, capture_pose)

                        # 너비 계산
                        obj_width_mm = (obj.width * z_dist) / fx
                        target_w = int(obj_width_mm * 10 * SQUEEZE_RATIO)
                        target_w = max(0, min(target_w, 1100))

                        print(f"\n\n[Found] {obj.class_name} | Dist: {z_dist:.1f}mm | Base: {base_xyz}")
                        
                        self.is_picking = True
                        current_target = self.target_object
                        self.target_object = None
                        
                        threading.Thread(
                            target=self.pick_and_place, 
                            args=(*base_xyz, target_w, current_target), 
                            daemon=True
                        ).start()
                        break

def main():
    try:
        test = PickItemServer()
        test.run()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
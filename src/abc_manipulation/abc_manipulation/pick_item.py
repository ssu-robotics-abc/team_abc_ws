import cv2
import rclpy
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation

from abc_manipulation.realsense import ImgNode
from abc_manipulation.onrobot import RG
import DR_init
# [수정] 커스텀 메시지 임포트 (패키지명에 맞게 확인 필요)
# from abc_interfaces.msg import DetectedObjectArray 
from abc_interfaces.msg import DetectionArray

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
X_OFFSET           = 185.0   # 그리퍼 중심 보정 (mm)
SIDE_APPROACH_DIST = 100.0   # 물체 정면 대기 거리 (mm)
SAFE_Z             = 400.0   # 이동 안전 높이
SQUEEZE_RATIO      = 0.95    # 파지 보정
GRIPPER_FORCE      = 5       # 파지 힘 (N)

class TestNode:
    def __init__(self):
        self.img_node = ImgNode()
        while rclpy.ok() and self.img_node.get_camera_intrinsic() is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)

        self.intrinsics = self.img_node.get_camera_intrinsic()
        self.gripper2cam = np.array(
            [[-9.99419954e-01, 2.64817788e-02, 2.14119182e-02, 2.67871780e+01],
 [-2.61544385e-02,  -9.99538886e-01,  1.54259848e-02,  4.30822834e+01],
 [ 2.18105524e-02,  1.48570203e-02,  9.99651724e-01,  1.71277841e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        self.gripper = RG("rg2", "192.168.1.1", 502)

        # 시작 포즈 설정
        self.JReady = posj([0, 0, 135, 0, -45, 90])
        self.home_pose = None  
        self.target_object = None  
        self.is_picking = False

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

    def input_command_thread(self):
        valid_classes = ['chocopie', 'Kancho', 'pepero_almond', 'pepero_original', 'pepsi', 'pocarisweat', 'soy_milk']
        while rclpy.ok():
            cmd = input(f"\n[명령 대기] {valid_classes}\n대상 입력 > ").strip()
            if cmd in valid_classes:
                self.target_object = cmd

    def pick_and_place(self, x, y, z, target_width):
        # [각도 고정] 현재 위치의 각도를 가져와서 작업 내내 고정 (수평 유지)
        cur = get_current_posx()[0]
        fixed_rx, fixed_ry, fixed_rz = cur[3:]
        
        print(f"\n>>> {self.target_object} 수평 집기 수행")
        print(f"각도 유지: [{fixed_rx}, {fixed_ry}, {fixed_rz}]")
        
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
        self.gripper.move_gripper(width_val=target_width, force=GRIPPER_FORCE) 
        time.sleep(1.2) 

        # 4. 후퇴 (뒤로 빠지기)
        movel(posx([wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)

        # 5. 상승 및 홈 이동
        movel(posx([wait_x, y, SAFE_Z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        movel(posx([hx, hy, SAFE_Z, hrx, hry, hrz]), VELOCITY, ACC)
        
        # 6. 복귀
        # self.gripper.open_gripper() # 투하가 필요하면 주석 해제
        time.sleep(1.0)
        movej(self.JReady, VELOCITY, ACC)
        
        self.is_picking = False
        print(">>> 작업 완료.")

    def run(self):
        # [수정] 이제 모델을 로드하지 않고 ImgNode의 메시지를 사용합니다.
        
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.img_node)
        threading.Thread(target=executor.spin, daemon=True).start()
        threading.Thread(target=self.input_command_thread, daemon=True).start()

        movej(self.JReady, VELOCITY, ACC)
        wait(1.0)
        self.home_pose = get_current_posx()[0]
        self.gripper.open_gripper()

        while True:
            img_frame = self.img_node.get_color_frame()
            depth_frame = self.img_node.get_depth_frame()
            
            detection_msg = self.img_node.get_latest_detection_msg()

            if img_frame is None or depth_frame is None or detection_msg is None:
                continue

            # 타겟 물체가 지정되었고, 현재 작업 중이 아닐 때만 실행
            if self.target_object is not None and not self.is_picking:
                # 전달받은 객체 리스트 순회
                for obj in detection_msg.objects:
                    name = obj.class_name
                    conf = obj.confidence
                    
                    if name == self.target_object and conf >= 0.6:
                        # 1. 메시지 필드 데이터 추출
                        cx, cy = int(obj.center_x), int(obj.center_y)
                        pixel_width = obj.width
                        # obj.heigh 는 필요 시 사용 (사용자 메시지 정의의 오타 반영)
                        
                        # 2. 깊이값 확인
                        z_dist = depth_frame[cy, cx]
                        if z_dist == 0: continue

                        # 3. 좌표 변환
                        capture_pose = get_current_posx()[0]
                        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                        ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]
                        
                        cam_coords = ((cx - ppx) * z_dist / fx, (cy - ppy) * z_dist / fy, z_dist)
                        base_xyz = self.transform_to_base(cam_coords, capture_pose)

                        # 4. 너비 계산 (Side Picking용)
                        obj_width_mm = (pixel_width * z_dist) / fx
                        target_w = int(obj_width_mm * 10 * SQUEEZE_RATIO)
                        target_w = max(0, min(target_w, 1100))

                        # === 디버깅 출력 ===
                        print("\n" + "="*50)
                        print(f"[Target Found] {name} (Conf: {conf:.2f})")
                        print(f"Distance: {z_dist:.2f}mm | Width: {obj_width_mm:.1f}mm")
                        print(f"Robot Pose: X={base_xyz[0]:.2f}, Y={base_xyz[1]:.2f}, Z={base_xyz[2]:.2f}")
                        print("="*50 + "\n")

                        # 5. 스레드 실행
                        self.is_picking = True
                        self.target_object = None 
                        threading.Thread(target=self.pick_and_place, args=(*base_xyz, target_w), daemon=True).start()
                        break

            # 화면 출력 (YOLO가 그린 그림 대신 원본 프레임 표시)
            cv2.imshow("Webcam View", img_frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        cv2.destroyAllWindows()
def main():

    try:

        test = TestNode()

        test.run()

    except KeyboardInterrupt:

        print("\n종료")

    finally:

        rclpy.shutdown()



if __name__ == "__main__":

    main()
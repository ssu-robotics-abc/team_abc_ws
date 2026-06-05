import os
import rclpy
import threading
import numpy as np
import time
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

import cv2
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
    from DSR_ROBOT2 import (get_current_posx, get_current_posj, movej, movel, wait, amovel, 
                            get_tool_force, task_compliance_ctrl, release_compliance_ctrl)
    from DR_common2 import posx, posj
except ImportError:
    print("두산 로봇 라이브러리 로드 실패")
    exit()

# ======================
# 2. 시스템 파라미터 설정
# ======================
X_OFFSET = 185.0            # 그리퍼 중심 보정 (mm)
SIDE_APPROACH_DIST = 180.0  # 물체 정면 대기 거리 (mm)
SAFE_Z = 400.0              # 이동 안전 높이
SQUEEZE_RATIO = 0.95
REAL_TABLE_Z = 5.0          # 티칭 펜던트로 측정한 실제 진열대 바닥의 Z 좌표

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
        '''
        self.pos_home_horiz = posj([1, 38.89, 123.21, -0.08, -71.09, 89.94])  # 수평 시작 자세             
        self.pos_home_vert  = posj([0, -20, 120, 0, 15, 90]) 
        '''
        self.pos_home_horiz = posx([431.39, 13.76, 112.90, 178.23, -90.00, -86.81])    
        
        self.pos_home_vert = posj([
            0.228,
            40.153,
            48.268,
            0.155,
            89.955,
            92.466
        ]) 
        '''
        self.pos_home_vert = posj([
            3,
            26.06,
            44.64,
            -2.57,
            109.11,
            94.13
        ])   
        '''       
        #self.pos_home_vert  = posx([635.02, 8.2, 337.79, 1.04, 178.40, 93.34])

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
        slots = goal_handle.request.get_fields_and_field_types().keys()
        self.get_logger().info(f"🚨 [인터페이스 디버깅] PickItem_Goal 내부 실제 변수 목록: {list(slots)}")
        """ 태스크 플래너로부터 목표 픽셀 좌표를 받아 실행되는 핵심 콜백 """
        # 1) 플래너가 보내준 Goal 데이터 꺼내기 (class_id 변수의 바코드 스트링 추출)
        cx = int(goal_handle.request.center_x)
        cy = int(goal_handle.request.center_y)
        w = goal_handle.request.width
        target_h = goal_handle.request.height # 제품의 실제 총 높이 (세로 길이)
        
        self.get_logger().info(f"[액션 수신] 피킹 개시 -> 중심:({cx}, {cy})")

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

        # -----------------------------------------------------------------
        # 🎯 [실측값 일대일 매칭 수립] 제품 바코드별 정밀 목표 파지 너비 지정 (0.1mm 단위)
        # -----------------------------------------------------------------
        obj_width_mm = (w * z_dist) / fx

        target_w = int(obj_width_mm * 10 * SQUEEZE_RATIO)

        # RG2 허용 범위 제한
        target_w = max(0, min(target_w, 1100))

        self.get_logger().info(
            f"🎯 변환 완료 -> "
            f"베이스 좌표: {base_xyz} | "
            f"예상 두께: {obj_width_mm:.1f}mm | "
            f"목표 폭: {target_w/10:.1f}mm"
        )

        # 4) 로봇 실제 구동 (블로킹 방식으로 순차 실행)
        try:
            # 수평 피킹부터 시작하여 바닥 안착 후 수직 피킹까지 원스톱으로 이어지는 마스터 시퀀스 실행

            
            self.execute_pick_motion(*base_xyz, target_w, target_h, z_dist, goal_handle)
            
            # 모든 동작 성공적 종료 시
            goal_handle.succeed()
            self.get_logger().info("✅ >>> 전체 피킹/안착/수직 재피킹 작업 성공 및 플래너에 완료 보고")
            return PickItem.Result(success=True)
            
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 구동 중 에러 발생: {e}")
            goal_handle.abort()
            return PickItem.Result(success=False)

    def execute_pick_motion(self, x, y, z, target_width, target_height, z_dist, goal_handle):
        """ 수평 피킹 -> 바닥 안착 -> 수직 전환 측정 -> 수직 파지 통합 시퀀스 """
        cur = get_current_posx()[0]
        fixed_rx, fixed_ry, fixed_rz = cur[3:]
        
        # 언팩 에러 예방용 예외 처리
        if self.home_pose is None:
            self.home_pose = cur
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
        
        pick_x = x - X_OFFSET  + 5 
        movel(posx([pick_x, y, z, fixed_rx, fixed_ry, fixed_rz]), VELOCITY, ACC)
        
        # Step 3. 파지
        feedback_msg.state = f"물품 파지 중 (실측 보정 너비: {target_width/10:.1f}mm)"
        goal_handle.publish_feedback(feedback_msg)
        self.get_logger().info(
            f"[DEBUG] grip done current_pos={get_current_posx()[0]}"
        )
        


        self.get_logger().info(f"[DEBUG] target_width={target_width} ({target_width/10:.1f}mm)")
        self.gripper.move_gripper(width_val=target_width)
        wait(1.2)
        self.get_logger().info(
            f"wait_x={wait_x}, pick_x={pick_x}"
        )
        # Step 4. 후퇴 (뒤로 빠지기)
        feedback_msg.state = "파지 완료 후 후방으로 후퇴 중"
        goal_handle.publish_feedback(feedback_msg)

        movel(
            posx([wait_x, y,z,
                fixed_rx, fixed_ry, fixed_rz]),
            VELOCITY,
            ACC
        )
        wait(1.2)
        self.get_logger().info(
            f"[DEBUG] retreat finished current={get_current_posx()[0]}"
        )

        # ---------------------------------------------------------------------
        # 로직 1) 수평으로 잡은 물품을 바닥에 사뿐히 내려놓는다
        # ---------------------------------------------------------------------
        feedback_msg.state = "후퇴 위치에서 내려놓는 중..."
        goal_handle.publish_feedback(feedback_msg)
        wait(1.0)

        fy = self.intrinsics["fy"]
        obj_height_mm = (target_height * z_dist) / fy
        place_z = REAL_TABLE_Z + (obj_height_mm / 2.0) +20.0  # 반높이보다 20mm 더 아래
    
        self.get_logger().info(f"📏 물체높이: {obj_height_mm:.1f}mm | place_z: {place_z:.1f}mm")
        place_pose = posx([wait_x, 13.76, place_z, fixed_rx, fixed_ry, fixed_rz])
        movel(place_pose, VELOCITY, ACC)
        wait(0.5)


        # ---------------------------------------------------------------------
        # 로직 2) 그리퍼를 연다
        # ---------------------------------------------------------------------
        feedback_msg.state = "그리퍼 개방 중..."
        goal_handle.publish_feedback(feedback_msg)
        self.gripper.open_gripper()
        wait(1.0) 

        # ---------------------------------------------------------------------
        # 로직 3) 수직 관측 자세로 전환 중
        # ---------------------------------------------------------------------
        feedback_msg.state = "수직 관측 자세로 전환 중..."
        goal_handle.publish_feedback(feedback_msg)
        
        cur = get_current_posx()[0]

        movel(posx([
            cur[0],
            cur[1],
            cur[2] + 300,
            cur[3],
            cur[4],
            cur[5]
        ]), VELOCITY, ACC)

        movej(self.pos_home_vert, VELOCITY, ACC)
        jpos = get_current_posj()[0]

        feedback_msg.state = f"{get_current_posj()}"
        goal_handle.publish_feedback(feedback_msg)

        wait(1.0) # 비전 카메라 상 흔들림 억제를 위한 안정화 대기 추가


        # ---------------------------------------------------------------------
        # 로직 4) Depth 기반 캔(또는 재배치 물품) 위치 측정
        # ---------------------------------------------------------------------
        feedback_msg.state = "Depth 기반 물체 검출 중..."
        goal_handle.publish_feedback(feedback_msg)
 
        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Depth frame 없음")
 
        h_d, w_d = depth_frame.shape
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]
 
        # 테이블 depth 추정: 프레임 가장자리 50px 띠의 중앙값
        border = 50
        edges = np.concatenate([
            depth_frame[:border, :].ravel(),
            depth_frame[-border:, :].ravel(),
            depth_frame[:, :border].ravel(),
            depth_frame[:, -border:].ravel(),
        ])
        edges_valid = edges[edges > 0]
        if len(edges_valid) == 0:
            raise RuntimeError("테이블 depth 추정 실패")
        floor_mm = float(np.median(edges_valid))
 
        # 물체 마스크: 테이블보다 20mm 이상 가까운 영역
        diff = floor_mm - depth_frame.astype(np.float32)
        mask = ((diff > 20) & (depth_frame > 0)).astype(np.uint8) * 255
 
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("물체 검출 실패")
 
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 300:
            raise RuntimeError("검출 영역 너무 작음")
 
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx_px = bx + bw // 2
        cy_px = by + bh // 2
 
        # 물체 표면 depth (중앙값)
        roi_depth = depth_frame[by:by+bh, bx:bx+bw]
        roi_valid = roi_depth[roi_depth > 0]
        if len(roi_valid) == 0:
            raise RuntimeError("물체 ROI 내 유효 depth 없음")
        obj_surface_mm = float(np.median(roi_valid))
 
        # 물체 실제 너비 & 높이 (mm)
        obj_width_mm = (bw * obj_surface_mm) / fx
        obj_height_mm = floor_mm - obj_surface_mm  # 물체 수직 높이
 
        self.get_logger().info(
            f"📦 물체 검출: center_px=({cx_px},{cy_px}) "
            f"너비={obj_width_mm:.1f}mm 높이={obj_height_mm:.1f}mm "
            f"표면depth={obj_surface_mm:.1f}mm 바닥depth={floor_mm:.1f}mm"
        )
 
        # 카메라 좌표 → 로봇 베이스 좌표 변환
        cam_coords = (
            (cx_px - ppx) * obj_surface_mm / fx,
            (cy_px - ppy) * obj_surface_mm / fy,
            obj_surface_mm
        )
        capture_pose = get_current_posx()[0]
        base_xyz = self.transform_to_base(cam_coords, capture_pose)
 
        target_x = base_xyz[0]
        target_y = base_xyz[1]
        target_z = base_xyz[2]
 
        # 그리퍼 너비 계산 (depth 측정 너비 기준)
 
        self.get_logger().info(
            f"🎯 수직 피킹 목표: X={target_x:.1f} Y={target_y:.1f} Z={target_z:.1f} "
        )
 
        # =================================================================
        # Step 9. 물체 중심 상부 접근
        # =================================================================
        feedback_msg.state = "물체 상부 접근 중..."
        goal_handle.publish_feedback(feedback_msg)
 
        vert_rx, vert_ry, vert_rz = capture_pose[3], capture_pose[4], capture_pose[5]
        


 
        # =================================================================
        # Step 10. 하강 & 파지 (물체 높이 기반)
        # =================================================================
        feedback_msg.state = f"하강 파지 중... 너비={obj_width_mm:.1f}mm 높이={obj_height_mm:.1f}mm"
        goal_handle.publish_feedback(feedback_msg)


        Z_SYSTEM_MARGIN = 200.0
        # 물체 높이의 절반 지점을 잡도록 하강
        grip_z = REAL_TABLE_Z + obj_height_mm + Z_SYSTEM_MARGIN

        
        self.get_logger().info(
            f"📐 파지 높이 계산: 표면Z={target_z:.1f} - 높이절반={obj_height_mm/2.0:.1f} = grip_z={grip_z:.1f}"
        )
 
        pick_pose = posx([
            target_x, target_y, grip_z,
            vert_rx, vert_ry, vert_rz
        ])
        movel(pick_pose, 10, 10)
        wait(0.3)
 
        # depth 기반 너비로 파지
        self.gripper.move_gripper(width_val=target_width)
        wait(1.2)
        # ---------------------------------------------------------------------
        # 🔥 [필수 추가] 로직 7) 들어올리기ter
        # ---------------------------------------------------------------------
        feedback_msg.state = "수직 피킹 완료 후 들어올리는 중..."
        goal_handle.publish_feedback(feedback_msg)

        # 파지한 상태 그대로 일직선으로 들어 올릴 수 있도록 회전각을 1.04, 178.40, 93.34로 변경합니다.
        p_lift = get_current_posx()[0]

        # 💡 X, Y축과 손목 각도(rx, ry, rz)는 100% 고정한 상태에서 오직 Z축만 +120mm 수직 상승시킵니다.
        lift_pose = posx([
            p_lift[0],              # 현재 X 그대로 유지
            p_lift[1],              # 현재 Y 그대로 유지
            p_lift[2] + 120.0,      # 현재 움켜쥔 높이 기준에서 정확히 120mm 수직 상승
            p_lift[3],              # 현재 rx 그대로 (관절 비틀림 원천 차단)
            p_lift[4],              # 현재 ry 그대로
            p_lift[5]               # 현재 rz 그대로
        ])

        movel(lift_pose, VELOCITY, ACC)
        wait(0.5)

    def run(self):
        """ 로봇 초기 자세 세팅 및 ROS2 멀티스레드 스핀 가동 """
        self.get_logger().info("로봇 초기 위치(JReady) 설정 중")
        movej(self.JReady, VELOCITY, ACC)
        wait(1.0)
        self.home_pose = get_current_posx()[0]
        self.gripper.open_gripper()
        self.get_logger().info("🤖 연쇄 하이브리드 피킹 시스템 준비 완료. 플래너 명령을 기다립니다.")

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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

import math
import os
import sys
import time
import threading

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from abc_interfaces.action import PickItem
from abc_manipulation.onrobot import RG
from abc_manipulation.realsense import ImgNode

GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK = "link_6"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

VELOCITY_SCALE = 0.1
ACCELERATION_SCALE = 0.2
SLOW_VELOCITY_SCALE = 1.0   # 수직 하강 파지 시 저속

X_OFFSET = 185.0
SIDE_APPROACH_DIST = 180.0
SQUEEZE_RATIO = 0.95
REAL_TABLE_Z = 5.0

# pos_home_horiz = [431.39, 13.76, 112.90, 178.23, -90.00, -86.81] 에서 추출한 수평 자세 오리엔테이션
HORIZ_RX = 178.23
HORIZ_RY = -90.00
HORIZ_RZ = -90.00


class PickItemServer(Node):
    def __init__(self):
        super().__init__("pick_item_action_helper")
        cb_group = ReentrantCallbackGroup()

        self.img_node = ImgNode()
        self.get_logger().info("카메라 및 비전 파라미터 대기 중")
        while rclpy.ok() and self.img_node.get_camera_intrinsic() is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)

        self.intrinsics = self.img_node.get_camera_intrinsic()

        package_path = get_package_share_directory("abc_manipulation")
        gripper2cam_file_path = os.path.join(package_path, 'T_gripper2camera.npy')
        if not os.path.exists(gripper2cam_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gripper2cam_file_path = os.path.join(current_dir, "T_gripper2camera.npy")
        self.gripper2cam = np.load(gripper2cam_file_path)

        try:
            self.gripper = RG("rg2", "192.168.1.1", 502)
            self.gripper.open_gripper()
            self.get_logger().info("[Pick_Item] 그리퍼 연결 성공 및 최대 개방")
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] 그리퍼 연결 실패: {e}")
            self.gripper = None

        try:
            self.robot = MoveItPy(node_name="pick_item_server")
            self.arm = self.robot.get_planning_component(GROUP_NAME)
            self.get_logger().info("[Pick_Item] MoveItPy 엔진 로드 완료")
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] MoveItPy 초기화 실패: {e}")
            sys.exit(1)

        self.joints_jready = {
            "joint_1": math.radians(0.0),
            "joint_2": math.radians(0.0),
            "joint_3": math.radians(135.0),
            "joint_4": math.radians(0.0),
            "joint_5": math.radians(-45.0),
            "joint_6": math.radians(90.0),
        }

        self.joints_home_vert = {
            "joint_1": math.radians(0.32),
            "joint_2": math.radians(36.83),
            "joint_3": math.radians(42.95),
            "joint_4": math.radians(0.15),
            "joint_5": math.radians(98.66),
            "joint_6": math.radians(92.58),
        }

        self._action_server = ActionServer(
            self,
            PickItem,
            "pick_item",
            self.execute_callback,
            callback_group=cb_group,
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

    def get_current_pose_dsr(self):
        """현재 EE 위치를 DSR 포맷 [x_mm, y_mm, z_mm, rx, ry, rz]으로 반환."""
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            state = scene.current_state
            pose = state.get_pose(EE_LINK)
        x_mm = pose.position.x * 1000.0
        y_mm = pose.position.y * 1000.0
        z_mm = pose.position.z * 1000.0
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rx, ry, rz = Rotation.from_quat(quat).as_euler('ZYZ', degrees=True)
        return [x_mm, y_mm, z_mm, rx, ry, rz]

    def to_pose_stamped(self, dsr_pose):
        quat = Rotation.from_euler("ZYZ", dsr_pose[3:], degrees=True).as_quat()
        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = dsr_pose[0] / 1000.0
        pose.pose.position.y = dsr_pose[1] / 1000.0
        pose.pose.position.z = dsr_pose[2] / 1000.0
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def build_joint_constraints(self, joint_goal):
        constraints = Constraints()
        for joint_name in JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = joint_goal[joint_name]
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        return [constraints]

    def make_plan_params(self, planner_id, velocity_scale=None):
        req_params = PlanRequestParameters(self.robot)
        req_params.planning_pipeline = "pilz_industrial_motion_planner"
        req_params.planner_id = planner_id
        req_params.max_velocity_scaling_factor = velocity_scale if velocity_scale is not None else VELOCITY_SCALE
        req_params.max_acceleration_scaling_factor = ACCELERATION_SCALE
        req_params.planning_time = 2.0
        return req_params

    def plan_and_execute_pose(self, dsr_pose, planner_id="PTP", velocity_scale=None):
        try:
            pose_stamped = self.to_pose_stamped(dsr_pose)
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=EE_LINK)
            return self.plan_and_execute(self.make_plan_params(planner_id, velocity_scale))
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] pose 계획/실행 중 예외: {e}")
            return False

    def plan_and_execute_joints(self, joint_goal):
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=self.build_joint_constraints(joint_goal))
            return self.plan_and_execute(self.make_plan_params("PTP"))
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] joint 계획/실행 중 예외: {e}")
            return False

    def plan_and_execute(self, req_params):
        plan_result = self.arm.plan(parameters=req_params)
        if not plan_result:
            self.get_logger().error("[Pick_Item] Planning failed")
            return False
        self.robot.execute(
            group_name=GROUP_NAME,
            robot_trajectory=plan_result.trajectory,
            blocking=True,
        )
        return True

    def execute_callback(self, goal_handle):
        slots = goal_handle.request.get_fields_and_field_types().keys()
        self.get_logger().info(f"🚨 [인터페이스 디버깅] PickItem_Goal 내부 실제 변수 목록: {list(slots)}")

        cx = int(goal_handle.request.center_x)
        cy = int(goal_handle.request.center_y)
        w = goal_handle.request.width
        target_h = goal_handle.request.height

        self.get_logger().info(f"[액션 수신] 피킹 개시 -> 중심:({cx}, {cy})")

        feedback_msg = PickItem.Feedback()
        feedback_msg.state = "받은 픽셀 좌표를 기반으로 3D 위치 계산 중"
        goal_handle.publish_feedback(feedback_msg)

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

        capture_pose = self.get_current_pose_dsr()
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        ppx, ppy = self.intrinsics["ppx"], self.intrinsics["ppy"]

        cam_coords = ((cx - ppx) * z_dist / fx, (cy - ppy) * z_dist / fy, z_dist)
        base_xyz = self.transform_to_base(cam_coords, capture_pose)

        obj_width_mm = (w * z_dist) / fx
        target_w = int(obj_width_mm * 10 * SQUEEZE_RATIO)
        target_w = max(0, min(target_w, 1100))

        self.get_logger().info(
            f"🎯 변환 완료 -> "
            f"베이스 좌표: {base_xyz} | "
            f"예상 두께: {obj_width_mm:.1f}mm | "
            f"목표 폭: {target_w/10:.1f}mm"
        )

        try:
            self.execute_pick_motion(*base_xyz, target_w, target_h, z_dist, goal_handle)
            goal_handle.succeed()
            self.get_logger().info("✅ >>> 전체 피킹/안착/수직 재피킹 작업 성공 및 플래너에 완료 보고")
            return PickItem.Result(success=True)
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 구동 중 에러 발생: {e}")
            goal_handle.abort()
            return PickItem.Result(success=False)

    def execute_pick_motion(self, x, y, z, target_width, target_height, z_dist, goal_handle):
        """수평 피킹 -> 바닥 안착 -> 수직 전환 측정 -> 수직 파지 통합 시퀀스"""
        self.gripper.open_gripper()
        feedback_msg = PickItem.Feedback()

        # Step 1: 접근 대기 (X축 후방 대기)
        feedback_msg.state = "물품 정면(접근 위치)으로 이동 중"
        goal_handle.publish_feedback(feedback_msg)

        wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
        grip_z = z + 30.0  # 중심보다 30mm 위를 파지
        self.plan_and_execute_pose([wait_x, y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "PTP")
        time.sleep(0.5)

        # Step 2: 진입 (수평 찌르기)
        feedback_msg.state = "그립 처리를 위해 물품으로 진입 중"
        goal_handle.publish_feedback(feedback_msg)

        pick_x = x - X_OFFSET
        self.plan_and_execute_pose([pick_x, y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN")

        # Step 3: 파지
        feedback_msg.state = f"물품 파지 중 (실측 보정 너비: {target_width/10:.1f}mm)"
        goal_handle.publish_feedback(feedback_msg)
        self.get_logger().info(f"[DEBUG] target_width={target_width} ({target_width/10:.1f}mm)")
        self.gripper.move_gripper(width_val=target_width)
        time.sleep(1.2)
        self.get_logger().info(f"wait_x={wait_x}, pick_x={pick_x}")

        # Step 4: 후퇴 (뒤로 빠지기)
        feedback_msg.state = "파지 완료 후 후방으로 후퇴 중"
        goal_handle.publish_feedback(feedback_msg)

        self.plan_and_execute_pose([wait_x, y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN")
        time.sleep(1.0)
        self.get_logger().info("[DEBUG] retreat finished")

        # 로직 1: 수평으로 잡은 물품을 바닥에 사뿐히 내려놓는다
        feedback_msg.state = "후퇴 위치에서 내려놓는 중..."
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.0)

        fy = self.intrinsics["fy"]
        obj_height_mm = (target_height * z_dist) / fy
        place_z = REAL_TABLE_Z + (obj_height_mm / 2.0) -5.0

        self.get_logger().info(f"📏 물체높이: {obj_height_mm:.1f}mm | place_z: {place_z:.1f}mm")
        self.plan_and_execute_pose([wait_x, 13.76, place_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN", velocity_scale=0.05)
        time.sleep(0.5)

        # 로직 2: 그리퍼를 연다
        feedback_msg.state = "그리퍼 개방 중..."
        goal_handle.publish_feedback(feedback_msg)
        self.gripper.open_gripper()
        time.sleep(1.0)

        # 로직 3: 수직 관측 자세로 전환
        feedback_msg.state = "수직 관측 자세로 전환 중..."
        goal_handle.publish_feedback(feedback_msg)

        cur = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [cur[0], cur[1], cur[2] + 400, cur[3], cur[4], cur[5]], "LIN"
        )
        self.plan_and_execute_joints(self.joints_home_vert)

        feedback_msg.state = "수직 관측 자세 도달"
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(1.0)

        # 로직 4: Depth 기반 캔(또는 재배치 물품) 위치 측정
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

        # minAreaRect로 물체 기울기 각도 검출
        rect = cv2.minAreaRect(cnt)
        (cx_f, cy_f), (rect_w, rect_h), rect_angle = rect
        cx_px = int(cx_f)
        cy_px = int(cy_f)

        # minAreaRect angle: [-90, 0) → 긴 축 기준으로 정규화
        if rect_w < rect_h:
            obj_angle_deg = rect_angle + 90.0
        else:
            obj_angle_deg = rect_angle

        # ROI depth는 axis-aligned bounding box로 계산
        bx, by, bw, bh = cv2.boundingRect(cnt)
        roi_depth = depth_frame[by:by+bh, bx:bx+bw]
        roi_valid = roi_depth[roi_depth > 0]
        if len(roi_valid) == 0:
            raise RuntimeError("물체 ROI 내 유효 depth 없음")
        obj_surface_mm = float(np.median(roi_valid))

        obj_width_mm = (min(rect_w, rect_h) * obj_surface_mm) / fx
        obj_height_mm = floor_mm - obj_surface_mm

        self.get_logger().info(
            f"📦 물체 검출: center_px=({cx_px},{cy_px}) "
            f"너비={obj_width_mm:.1f}mm 높이={obj_height_mm:.1f}mm "
            f"각도={obj_angle_deg:.1f}° "
            f"표면depth={obj_surface_mm:.1f}mm 바닥depth={floor_mm:.1f}mm"
        )

        cam_coords = (
            (cx_px - ppx) * obj_surface_mm / fx,
            (cy_px - ppy) * obj_surface_mm / fy,
            obj_surface_mm
        )
        capture_pose = self.get_current_pose_dsr()
        base_xyz = self.transform_to_base(cam_coords, capture_pose)

        target_x = base_xyz[0]
        target_y = base_xyz[1]
        target_z = base_xyz[2]

        self.get_logger().info(
            f"🎯 수직 피킹 목표: X={target_x:.1f} Y={target_y:.1f} Z={target_z:.1f} 각도={obj_angle_deg:.1f}°"
        )

        # Step 9: 물체 중심 상부 접근 (수직 자세 오리엔테이션 추출 + 각도 반영)
        feedback_msg.state = "물체 상부 접근 중..."
        goal_handle.publish_feedback(feedback_msg)

        vert_rx, vert_ry, vert_rz = capture_pose[3], capture_pose[4], capture_pose[5]
        vert_rz = vert_rz + obj_angle_deg

        # Step 10: 하강 & 파지 (물체 높이 기반)
        feedback_msg.state = f"하강 파지 중... 너비={obj_width_mm:.1f}mm 높이={obj_height_mm:.1f}mm"
        goal_handle.publish_feedback(feedback_msg)

        Z_SYSTEM_MARGIN = 200.0
        grip_z = REAL_TABLE_Z + obj_height_mm + Z_SYSTEM_MARGIN

        self.get_logger().info(
            f"📐 파지 높이 계산: 표면Z={target_z:.1f} - 높이절반={obj_height_mm/2.0:.1f} = grip_z={grip_z:.1f}"
        )

        self.plan_and_execute_pose(
            [target_x, target_y, grip_z, vert_rx, vert_ry, vert_rz],
            "LIN",
            velocity_scale=SLOW_VELOCITY_SCALE,
        )
        time.sleep(0.3)

        self.gripper.move_gripper(width_val=target_width)
        time.sleep(1.2)

        # 로직 7: 들어올리기
        feedback_msg.state = "수직 피킹 완료 후 들어올리는 중..."
        goal_handle.publish_feedback(feedback_msg)

        p_lift = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [p_lift[0], p_lift[1], p_lift[2] + 120.0, p_lift[3], p_lift[4], p_lift[5]],
            "LIN",
        )
        time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    server = PickItemServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(server.img_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    server.get_logger().info("로봇 초기 위치(JReady) 설정 중")
    server.plan_and_execute_joints(server.joints_jready)
    if server.gripper is not None:
        server.gripper.open_gripper()
    server.get_logger().info("🤖 연쇄 하이브리드 피킹 시스템 준비 완료. 플래너 명령을 기다립니다.")

    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

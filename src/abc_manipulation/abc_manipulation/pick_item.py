import math
import os
import sys
import time
import threading

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Vector3
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from abc_interfaces.action import PickItem
from abc_interfaces.srv import DetectItem
from abc_manipulation.onrobot import RG, close_until_grip_detected
from abc_manipulation.realsense import ImgNode

GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK = "link_6"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

VELOCITY_SCALE = 0.5
ACCELERATION_SCALE = 0.2
SLOW_VELOCITY_SCALE = 0.4   # 수직 하강 파지 시 저속

X_OFFSET = 185.0
SIDE_APPROACH_DIST = 180.0
SQUEEZE_RATIO = 0.95
REAL_TABLE_Z = 5.0
Y_PICK_OFFSET = 5.0   # 수평 파지 Y 보정 (오른쪽으로 치우치면 음수, 왼쪽이면 양수)
HORIZ_MIN_GRIP_WIDTH_RATIO = 0.5
VERT_MIN_GRIP_WIDTH_RATIO = 0.5

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

        self._latest_yolo_cx: float | None = None
        self._latest_yolo_cy: float | None = None
        self.create_subscription(
            Vector3, "/yolo_tracking", self._yolo_tracking_cb, 10,
            callback_group=cb_group,
        )

        # 접근 위치 라이브 재탐지 서비스 클라이언트 (/recapture_item)
        # 대기 위치 도착 후 현재 프레임에서 물품을 다시 탐지받아 Y 를 보정한다.
        self.recapture_cli = self.create_client(
            DetectItem, "/recapture_item", callback_group=cb_group,
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

    def _yolo_tracking_cb(self, msg: Vector3):
        self._latest_yolo_cx = msg.x
        self._latest_yolo_cy = msg.y

    def request_recapture(self, barcode: str):
        """[vlm_perception_node 시나리오] /recapture_item 서비스로 '지금 프레임'
        라이브 재탐지를 요청한다. 성공 시 (center_x, center_y, width, height) 픽셀, 실패 시 None.
        서비스가 없으면(=yolo_node 시나리오) None 을 반환해 토픽 폴백으로 넘긴다."""
        if not barcode:
            self.get_logger().warn("⚠️ 바코드가 비어 재탐지 불가 → 폴백")
            return None
        if not self.recapture_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("⚠️ /recapture_item 서비스 없음 → /yolo_tracking 폴백")
            return None

        req = DetectItem.Request()
        req.class_name = barcode
        future = self.recapture_cli.call_async(req)

        # execute_callback 은 worker 스레드에서 실행되고 executor 는 별도 스레드에서
        # spin 중이므로, 여기서 future 완료까지 동기 대기해도 응답이 처리된다.
        start = time.time()
        while not future.done():
            if time.time() - start > 20.0:
                self.get_logger().warn("⚠️ 재탐지 응답 타임아웃 → fallback")
                return None
            time.sleep(0.02)

        result = future.result()
        if result is None or not result.success:
            msg = result.message if result is not None else "응답 없음"
            self.get_logger().warn(f"⚠️ 재탐지 실패({msg}) → fallback")
            return None
        return (
            float(result.center_x),
            float(result.center_y),
            float(result.width),
            float(result.height),
        )

    def recapture_from_tracking_topic(self):
        """[yolo_node 시나리오] /yolo_tracking 의 최신 픽셀을 사용 (없으면 None).
        yolo_node 는 매 프레임 라이브로 발행하므로, 대기 위치 시점의 좌표가 들어온다."""
        cx, cy = self._latest_yolo_cx, self._latest_yolo_cy
        if cx is None or cy is None:
            return None
        return float(cx), float(cy), None, None

    def recapture_pick_target(
        self,
        fallback_x: float,
        fallback_y: float,
        fallback_z: float,
        fallback_target_width: int,
        fallback_target_height: float,
        fallback_z_dist: float,
        barcode: str,
    ):
        """현재 프레임 재탐지로 수평 파지용 X/Y/Z/depth/width/height 를 다시 계산한다."""
        recap = self.request_recapture(barcode)          # 1순위: 서비스 (vlm_perception_node)
        if recap is None:
            recap = self.recapture_from_tracking_topic()  # 2순위: 토픽 폴백 (yolo_node)
        if recap is None:
            self.get_logger().warn("⚠️ 재탐지 좌표 없음(서비스/토픽 모두) → 기존 좌표/depth 사용")
            return (
                fallback_x,
                fallback_y,
                fallback_z,
                fallback_target_width,
                fallback_target_height,
                fallback_z_dist,
            )
        cx_px, cy_px, width_px, height_px = recap

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            self.get_logger().warn("⚠️ depth frame 없음 → 기존 좌표/depth 사용")
            return (
                fallback_x,
                fallback_y,
                fallback_z,
                fallback_target_width,
                fallback_target_height,
                fallback_z_dist,
            )

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        h_d, w_d = depth_frame.shape
        cx_i = max(0, min(int(cx_px), w_d - 1))
        cy_i = max(0, min(int(cy_px), h_d - 1))
        roi_r = 5
        y0 = max(0, cy_i - roi_r); y1 = min(h_d, cy_i + roi_r + 1)
        x0 = max(0, cx_i - roi_r); x1 = min(w_d, cx_i + roi_r + 1)
        roi = depth_frame[y0:y1, x0:x1]
        valid = roi[roi > 0]
        if len(valid) == 0:
            self.get_logger().warn("⚠️ ROI depth 유효값 없음 → 기존 좌표/depth 사용")
            return (
                fallback_x,
                fallback_y,
                fallback_z,
                fallback_target_width,
                fallback_target_height,
                fallback_z_dist,
            )

        z_dist = float(np.median(valid))
        current_pose = self.get_current_pose_dsr()
        cam_coords = ((cx_i - ppx) * z_dist / fx, (cy_i - ppy) * z_dist / fy, z_dist)
        base_xyz = self.transform_to_base(cam_coords, current_pose)

        target_width = fallback_target_width
        if width_px is not None and width_px > 0:
            obj_width_mm = (width_px * z_dist) / fx
            target_width = int(obj_width_mm * 10 * SQUEEZE_RATIO)
            target_width = max(0, min(target_width, 1100))

        target_height = fallback_target_height
        if height_px is not None and height_px > 0:
            target_height = height_px

        self.get_logger().info(
            f"🎯 재탐지 pick target: "
            f"XYZ=({fallback_x:.1f},{fallback_y:.1f},{fallback_z:.1f}) → "
            f"({base_xyz[0]:.1f},{base_xyz[1]:.1f},{base_xyz[2]:.1f}) mm, "
            f"depth={fallback_z_dist:.0f}→{z_dist:.0f}mm, "
            f"target_width={fallback_target_width/10:.1f}→{target_width/10:.1f}mm"
        )
        return (
            float(base_xyz[0]),
            float(base_xyz[1]),
            float(base_xyz[2]),
            target_width,
            target_height,
            z_dist,
        )

    def recapture_y(self, fallback_y: float, barcode: str) -> float:
        """wait_x 도달 후 라이브 재탐지 픽셀로 Y 재계산 (eye-in-hand 시점 보정).

        두 실행 시나리오를 모두 지원한다:
          - vlm_perception_node 단독: /recapture_item 서비스(라이브 재탐지) 사용
          - yolo_node + vlm_node:     서비스가 없으므로 /yolo_tracking 토픽(라이브) 폴백
        둘 다 실패하면 초기 y(fallback_y) 를 그대로 쓴다.
        """
        recap = self.request_recapture(barcode)          # 1순위: 서비스 (vlm_perception_node)
        if recap is None:
            recap = self.recapture_from_tracking_topic()  # 2순위: 토픽 폴백 (yolo_node)
        if recap is None:
            self.get_logger().warn("⚠️ 재탐지 좌표 없음(서비스/토픽 모두) → fallback Y 사용")
            return fallback_y
        cx_px, cy_px = recap[:2]

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            self.get_logger().warn("⚠️ depth frame 없음 → fallback Y 사용")
            return fallback_y

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        h_d, w_d = depth_frame.shape
        cx_i = max(0, min(int(cx_px), w_d - 1))
        cy_i = max(0, min(int(cy_px), h_d - 1))
        roi_r = 5
        y0 = max(0, cy_i - roi_r); y1 = min(h_d, cy_i + roi_r + 1)
        x0 = max(0, cx_i - roi_r); x1 = min(w_d, cx_i + roi_r + 1)
        roi = depth_frame[y0:y1, x0:x1]
        valid = roi[roi > 0]
        if len(valid) == 0:
            self.get_logger().warn("⚠️ ROI depth 유효값 없음 → fallback Y 사용")
            return fallback_y

        z_mm = float(np.median(valid))
        current_pose = self.get_current_pose_dsr()
        cam_coords = ((cx_i - ppx) * z_mm / fx, (cy_i - ppy) * z_mm / fy, z_mm)
        base_xyz = self.transform_to_base(cam_coords, current_pose)

        self.get_logger().info(
            f"🎯 재탐지 Y: {fallback_y:.1f} → {base_xyz[1]:.1f} mm "
            f"(pixel=({cx_i},{cy_i}), depth={z_mm:.0f}mm)"
        )
        return float(base_xyz[1])

    def execute_callback(self, goal_handle):
        slots = goal_handle.request.get_fields_and_field_types().keys()
        self.get_logger().info(f"🚨 [인터페이스 디버깅] PickItem_Goal 내부 실제 변수 목록: {list(slots)}")

        cx = int(goal_handle.request.center_x)
        cy = int(goal_handle.request.center_y)
        w = goal_handle.request.width
        target_h = goal_handle.request.height
        barcode = goal_handle.request.class_name.strip()

        self.get_logger().info(f"[액션 수신] 피킹 개시 -> 중심:({cx}, {cy}) 바코드:{barcode}")

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
            self.execute_pick_motion(*base_xyz, target_w, target_h, z_dist, goal_handle, barcode)
            goal_handle.succeed()
            self.get_logger().info("✅ >>> 전체 피킹/안착/수직 재피킹 작업 성공 및 플래너에 완료 보고")
            return PickItem.Result(success=True)
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 구동 중 에러 발생: {e}")
            goal_handle.abort()
            return PickItem.Result(success=False)

    def _is_gripping(self, min_width_mm: float = 2.0, refresh: bool = False) -> bool:
        """그리퍼가 물체를 쥐고 있는지 확인. 통신 실패 시 True(유지)로 간주."""
        if self.gripper is None:
            return True
        try:
            if refresh:
                detected = close_until_grip_detected(
                    self.gripper,
                    force_val=40,
                    timeout=0.8,
                    poll_interval=0.02,
                )
                if not detected:
                    width_mm = self.gripper.get_width()
                    self.get_logger().info(
                        f"🔎 그립 확인: 재폐쇄 중 grip 미감지 width={width_mm:.1f}mm "
                        f"refresh={refresh} -> False"
                    )
                    return False
            status = self.gripper.get_status()
            width_mm = self.gripper.get_width()
            gripping = bool(status[1]) and width_mm >= min_width_mm
            self.get_logger().info(
                f"🔎 그립 확인: grip_detected={bool(status[1])} width={width_mm:.1f}mm "
                f"refresh={refresh} -> {gripping}"
            )
            return gripping
        except Exception as e:
            self.get_logger().warn(f"⚠️ 그립 상태 확인 실패: {e} → 유지로 간주")
            return True

    def _do_vertical_pick(self, goal_handle):
        """현재 관측 자세에서 depth 인식 → 수직 파지 → 들어올리기."""
        feedback_msg = PickItem.Feedback()

        feedback_msg.state = "Depth 기반 물체 검출 중..."
        goal_handle.publish_feedback(feedback_msg)

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("Depth frame 없음")

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

        rect = cv2.minAreaRect(cnt)
        (cx_f, cy_f), (rect_w, rect_h), rect_angle = rect
        cx_px = int(cx_f)
        cy_px = int(cy_f)

        if rect_w < rect_h:
            obj_angle_deg = rect_angle + 90.0
        else:
            obj_angle_deg = rect_angle

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
            obj_surface_mm,
        )
        capture_pose = self.get_current_pose_dsr()
        base_xyz = self.transform_to_base(cam_coords, capture_pose)

        target_x, target_y, target_z = base_xyz[0], base_xyz[1], base_xyz[2]
        self.get_logger().info(
            f"🎯 수직 피킹 목표: X={target_x:.1f} Y={target_y:.1f} Z={target_z:.1f} 각도={obj_angle_deg:.1f}°"
        )

        vert_rx, vert_ry, vert_rz = capture_pose[3], capture_pose[4], capture_pose[5]
        vert_rz = vert_rz + obj_angle_deg

        Z_SYSTEM_MARGIN = 200.0
        grip_z = REAL_TABLE_Z + obj_height_mm + Z_SYSTEM_MARGIN

        feedback_msg.state = f"하강 파지 중... 너비={obj_width_mm:.1f}mm 높이={obj_height_mm:.1f}mm"
        goal_handle.publish_feedback(feedback_msg)
        self.plan_and_execute_pose(
            [target_x, target_y, grip_z - 15, vert_rx, vert_ry, vert_rz],
            "LIN",
            velocity_scale=SLOW_VELOCITY_SCALE,
        )
        time.sleep(0.5)

        grip_detected = close_until_grip_detected(
            self.gripper,
            force_val=80,
            timeout=5.0,
            poll_interval=0.02,
        )
        time.sleep(0.5)
        if not grip_detected:
            raise RuntimeError("수직 파지 grip 미감지")
        width_mm = self.gripper.get_width()
        min_grip_width_mm = obj_width_mm * VERT_MIN_GRIP_WIDTH_RATIO
        if width_mm < min_grip_width_mm:
            raise RuntimeError(
                f"수직 파지 폭 부족: width={width_mm:.1f}mm "
                f"min={min_grip_width_mm:.1f}mm expected={obj_width_mm:.1f}mm"
            )

        feedback_msg.state = "수직 피킹 완료 후 들어올리는 중..."
        goal_handle.publish_feedback(feedback_msg)
        p_lift = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [p_lift[0], p_lift[1], p_lift[2] + 120.0, p_lift[3], p_lift[4], p_lift[5]],
            "LIN",
        )
        time.sleep(0.5)

    def _do_vertical_pick_with_retry(self, goal_handle, max_retry: int = 5, context: str = "수직 파지"):
        feedback_msg = PickItem.Feedback()

        for vert_attempt in range(1, max_retry + 1):
            try:
                self._do_vertical_pick(goal_handle)
                if self._is_gripping(refresh=True):
                    return
            except Exception as e:
                self.get_logger().warn(
                    f"⚠️ {context} 예외 (시도 {vert_attempt}/{max_retry}): {e}"
                )

            self.get_logger().warn(
                f"⚠️ {context} 실패 (시도 {vert_attempt}/{max_retry}) → 관측 자세 복귀 후 재인식"
            )
            if vert_attempt >= max_retry:
                raise RuntimeError(f"{context} 최대 재시도 횟수 초과")

            feedback_msg.state = "수직 파지 재시도를 위해 관측 자세로 복귀 중..."
            goal_handle.publish_feedback(feedback_msg)
            self.gripper.open_gripper()
            cur = self.get_current_pose_dsr()
            self.plan_and_execute_pose(
                [cur[0], cur[1], cur[2] + 150, cur[3], cur[4], cur[5]], "LIN"
            )
            self.plan_and_execute_joints(self.joints_home_vert)
            time.sleep(0.5)

    def _vertical_recovery_to_pos_return(self, goal_handle, pos_return):
        """운반 중 낙하 복구: 수직 관측 자세 → 수직 파지 → pos_return 내려놓기 → jready."""
        feedback_msg = PickItem.Feedback()

        feedback_msg.state = "낙하 복구: 수직 관측 자세로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        cur = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [cur[0], cur[1], cur[2] + 200, cur[3], cur[4], cur[5]], "LIN"
        )
        self.plan_and_execute_joints(self.joints_home_vert)
        time.sleep(0.5)

        self.gripper.open_gripper()
        time.sleep(0.5)
        self._do_vertical_pick_with_retry(goal_handle, max_retry=5, context="이탈 복구 수직 파지")

        feedback_msg.state = "복구 파지 후 반납 위치로 이동 중..."
        goal_handle.publish_feedback(feedback_msg)
        approach = list(pos_return)
        approach[2] = approach[2] + 100.0
        self.plan_and_execute_pose(approach, "PTP")
        self.plan_and_execute_pose(pos_return, "LIN", velocity_scale=0.05)
        time.sleep(0.3)
        self.gripper.open_gripper()
        time.sleep(0.5)

        cur = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [cur[0], cur[1], cur[2] + 150, cur[3], cur[4], cur[5]], "LIN"
        )
        self.plan_and_execute_joints(self.joints_jready)
        time.sleep(0.5)

    def execute_pick_motion(self, x, y, z, target_width, target_height, z_dist, goal_handle, barcode=""):
        """수평 피킹 -> 바닥 안착 -> 수직 전환 측정 -> 수직 파지 통합 시퀀스"""
        POS_RETURN = [408.0, -153.0, 342.0, 133.0, -180.0, 100.0]
        MAX_TRANSPORT_RETRY = 2

        self.gripper.open_gripper()
        feedback_msg = PickItem.Feedback()

        pick_x = x - X_OFFSET
        pick_y = y + Y_PICK_OFFSET
        grip_z = z + 30.0

        for attempt in range(1, MAX_TRANSPORT_RETRY + 1):
            wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
            grip_z = z + 30.0
            pick_y = y + Y_PICK_OFFSET
            feedback_msg.state = f"물품 정면으로 이동 중 (시도 {attempt})"
            goal_handle.publish_feedback(feedback_msg)
            if not self.plan_and_execute_pose([wait_x, pick_y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "PTP"):
                raise RuntimeError("접근 대기 위치 이동 실패")
            time.sleep(0.5)

            old_wait_x = wait_x
            x, y, z, target_width, target_height, z_dist = self.recapture_pick_target(
                x, y, z, target_width, target_height, z_dist, barcode
            )
            wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
            grip_z = z + 30.0
            pick_y = y + Y_PICK_OFFSET
            if abs(wait_x - old_wait_x) > 1.0:
                if not self.plan_and_execute_pose([wait_x, pick_y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "PTP"):
                    raise RuntimeError("재탐지 접근 대기 위치 이동 실패")
                time.sleep(0.5)

            feedback_msg.state = "물품으로 진입 중"
            goal_handle.publish_feedback(feedback_msg)
            pick_x = x - X_OFFSET
            if not self.plan_and_execute_pose([pick_x + 10, pick_y, grip_z, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN"):
                raise RuntimeError("물품 진입 이동 실패")

            feedback_msg.state = f"물품 파지 중 (실측 보정 너비: {target_width/10:.1f}mm)"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"[DEBUG] target_width={target_width} ({target_width/10:.1f}mm)")
            grip_detected = close_until_grip_detected(
                self.gripper,
                force_val=100,
                timeout=5.0,
                poll_interval=0.02,
            )
            time.sleep(0.5)
            self.get_logger().info(f"wait_x={wait_x}, pick_x={pick_x}, pick_y={pick_y}")

            width_mm = self.gripper.get_width()
            expected_width_mm = (target_width / 10.0) / SQUEEZE_RATIO
            min_grip_width_mm = expected_width_mm * HORIZ_MIN_GRIP_WIDTH_RATIO
            if (not grip_detected) or width_mm < min_grip_width_mm:
                self.get_logger().warn(
                    f"⚠️ 수평 파지 직후 미파지 감지: "
                    f"grip_detected={grip_detected} width={width_mm:.1f}mm "
                    f"(min={min_grip_width_mm:.1f}mm, expected={expected_width_mm:.1f}mm, "
                    f"시도 {attempt}/{MAX_TRANSPORT_RETRY})"
                )
                self.gripper.open_gripper()
                time.sleep(0.5)
                self.plan_and_execute_joints(self.joints_jready)
                time.sleep(0.5)
                if attempt < MAX_TRANSPORT_RETRY:
                    continue
                raise RuntimeError("수평 파지 직후 미파지 재시도 횟수 초과")

            feedback_msg.state = "파지 완료 후 후방으로 후퇴 중"
            goal_handle.publish_feedback(feedback_msg)
            self.plan_and_execute_pose([pick_x + 10, pick_y, grip_z + 5, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN")
            time.sleep(0.5)
            self.plan_and_execute_pose([430, pick_y, grip_z + 5, HORIZ_RX, HORIZ_RY, HORIZ_RZ], "PTP")
            time.sleep(0.5)
            self.get_logger().info("[DEBUG] retreat finished")

            if self._is_gripping(refresh=True):
                feedback_msg.state = "후퇴 위치에서 내려놓는 중..."
                goal_handle.publish_feedback(feedback_msg)
                time.sleep(0.5)

                fy = self.intrinsics["fy"]
                obj_height_mm = (target_height * z_dist) / fy
                place_z = REAL_TABLE_Z + (obj_height_mm / 2.0) - 5.0

                feedback_msg.state = "하강 전 파지 상태 재확인 중..."
                goal_handle.publish_feedback(feedback_msg)
                if not self._is_gripping(refresh=True):
                    self.get_logger().warn(
                        f"⚠️ 하강 전 물품 이탈 감지 (시도 {attempt}/{MAX_TRANSPORT_RETRY}) "
                        "→ 수직 복구 후 반품대로 이동"
                    )
                    self.gripper.open_gripper()
                    self._vertical_recovery_to_pos_return(goal_handle, POS_RETURN)
                    self.gripper.open_gripper()
                    if attempt >= MAX_TRANSPORT_RETRY:
                        raise RuntimeError("하강 전 이탈 복구 최대 횟수 초과")
                    continue

                self.get_logger().info(f"📏 물체높이: {obj_height_mm:.1f}mm | place_z: {place_z:.1f}mm")
                self.plan_and_execute_pose([430, 13.76, max(place_z, 80), HORIZ_RX, HORIZ_RY, HORIZ_RZ], "LIN", velocity_scale=0.05)
                time.sleep(0.5)

                feedback_msg.state = "그리퍼 개방 전 파지 상태 확인 중..."
                goal_handle.publish_feedback(feedback_msg)
                if self._is_gripping(refresh=True):
                    feedback_msg.state = "그리퍼 개방 중..."
                    goal_handle.publish_feedback(feedback_msg)
                    self.gripper.open_gripper()
                    time.sleep(0.5)
                    break

                self.get_logger().warn(
                    f"⚠️ 그리퍼 개방 전 물품 이탈 감지 (시도 {attempt}/{MAX_TRANSPORT_RETRY})"
                )
                if attempt >= MAX_TRANSPORT_RETRY:
                    raise RuntimeError("그리퍼 개방 전 이탈 복구 최대 횟수 초과")
                self._vertical_recovery_to_pos_return(goal_handle, POS_RETURN)
                self.gripper.open_gripper()
                continue

            self.get_logger().warn(f"⚠️ 운반 중 낙하 감지 (시도 {attempt}/{MAX_TRANSPORT_RETRY})")
            self.gripper.open_gripper()
            if attempt < MAX_TRANSPORT_RETRY:
                self._vertical_recovery_to_pos_return(goal_handle, POS_RETURN)
                self.gripper.open_gripper()
                continue
            raise RuntimeError("운반 중 낙하 복구 최대 횟수 초과")

        feedback_msg.state = "수직 관측 자세로 전환 중..."
        goal_handle.publish_feedback(feedback_msg)
        cur = self.get_current_pose_dsr()
        self.plan_and_execute_pose(
            [cur[0], cur[1], cur[2] + 400, cur[3], cur[4], cur[5]], "LIN"
        )
        self.plan_and_execute_joints(self.joints_home_vert)
        feedback_msg.state = "수직 관측 자세 도달"
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(0.5)

        self._do_vertical_pick_with_retry(goal_handle, max_retry=3, context="최종 수직 파지")


def main(args=None):
    rclpy.init(args=args)
    server = PickItemServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(server.img_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(0.5)

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

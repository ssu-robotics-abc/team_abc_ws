import os
import sys
import time
import math

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

VELOCITY_SCALE = 0.15
ACCELERATION_SCALE = 0.1

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

X_OFFSET = 185.0
SIDE_APPROACH_DIST = 150.0 #후퇴하는 거리
SQUEEZE_RATIO = 0.95

REAL_TABLE_Z = 5.0
PLACE_Z_BIAS_MM = 15.0 #수평 파지에서 제품 내려놓을때의 Z 보정값
VERTICAL_RPY = [1.04, 178.40, 93.34]
VERTICAL_X_OFFSET_MM = 0.0 #수직 자세에서 다시 잡을때 X 보정값
VERTICAL_Y_OFFSET_MM = 0.0 #수직 자세에서 다시 잡을때 Y 보정값
VERTICAL_LIFT_BEFORE_ROTATE_MM = 300.0 
DEPTH_CENTER_ROI_PX = 90
DEPTH_FLOOR_RING_PX = 150
MIN_DEPTH_POINTS = 30
OBJECT_MIN_HEIGHT_MM = 20.0
OBJECT_MAX_HEIGHT_MM = 300.0
OBJECT_TOP_PERCENTILE = 92.0
OBJECT_WIDTH_PERCENTILE_LOW = 5.0
OBJECT_WIDTH_PERCENTILE_HIGH = 95.0
VERTICAL_APPROACH_CLEARANCE_MM = 120.0
VERTICAL_TOP_GRASP_DEPTH_MM = 25.0
VERTICAL_DEPTH_Z_BIAS_MM = 0.0
MIN_PICK_ABOVE_FLOOR_MM = 30.0
VERTICAL_GRIP_WIDTH_AXIS = "y"
VERTICAL_GRIP_WIDTH_MARGIN_MM = 8.0
VERTICAL_CENTER_CORRECTION_LIMIT_MM = 120.0
FINAL_LIFT_MM = 100.0


class PickItemServer(Node):
    def __init__(self):
        super().__init__("pick_item_action_helper")
        cb_group = ReentrantCallbackGroup()

        self.img_node = ImgNode()
        self.get_logger().info("[Pick_Item] 카메라 및 비전 파라미터 대기 중")
        while rclpy.ok() and self.img_node.get_camera_intrinsic() is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
        self.intrinsics = self.img_node.get_camera_intrinsic()

        package_path = get_package_share_directory("abc_manipulation")
        gripper2cam_file_path = os.path.join(package_path, "T_gripper2camera.npy")
        if not os.path.exists(gripper2cam_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gripper2cam_file_path = os.path.join(current_dir, "T_gripper2camera.npy")
        self.gripper2cam = np.load(gripper2cam_file_path)

        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Pick_Item] 그리퍼 연결 성공")
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

        self.joints_ready = {
            "joint_1": math.radians(0.0),
            "joint_2": math.radians(0.0),
            "joint_3": math.radians(135.0),
            "joint_4": math.radians(0.0),
            "joint_5": math.radians(-45.0),
            "joint_6": math.radians(90.0),
        }

        if not self.move_to_ready_pose():
            self.get_logger().error("[Pick_Item] JReady 이동 실패. 서버를 종료합니다.")
            sys.exit(1)

        self._action_server = ActionServer(
            self,
            PickItem,
            "pick_item",
            self.execute_callback,
            callback_group=cb_group,
        )
        self.get_logger().info("[Pick_Item] pick_item 서버가 준비되었습니다.")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        rotation = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [x, y, z]
        return transform

    def transform_to_base(self, camera_coords, capture_pose):
        coord = np.append(np.array(camera_coords, dtype=float), 1.0)
        base2gripper = self.get_robot_pose_matrix(*capture_pose)
        base2cam = base2gripper @ self.gripper2cam
        base_coord = base2cam @ coord
        return base_coord[:3]

    def transform_depth_roi_to_base_points(self, depth_frame, roi, capture_pose):
        x1, y1, x2, y2 = roi
        roi_depth = depth_frame[y1:y2, x1:x2].astype(float)
        valid_mask = roi_depth > 0.0
        if np.count_nonzero(valid_mask) == 0:
            return np.empty((0, 3)), np.empty((0, 2), dtype=int)

        ys, xs = np.indices(roi_depth.shape)
        px = xs[valid_mask] + x1
        py = ys[valid_mask] + y1
        depth = roi_depth[valid_mask]

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        cam_x = (px - ppx) * depth / fx
        cam_y = (py - ppy) * depth / fy
        cam_points = np.column_stack((cam_x, cam_y, depth, np.ones_like(depth)))

        base2gripper = self.get_robot_pose_matrix(*capture_pose)
        base2cam = base2gripper @ self.gripper2cam
        base_points = (base2cam @ cam_points.T).T[:, :3]
        pixels = np.column_stack((px, py)).astype(int)
        return base_points, pixels

    def project_base_to_pixel(self, base_xyz, capture_pose):
        base_point = np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=float)
        base2gripper = self.get_robot_pose_matrix(*capture_pose)
        base2cam = base2gripper @ self.gripper2cam
        cam_point = np.linalg.inv(base2cam) @ base_point
        if cam_point[2] <= 0.0:
            raise RuntimeError(f"projection 실패: 카메라 앞쪽 점이 아닙니다. cam_z={cam_point[2]:.1f}")

        u = (cam_point[0] * self.intrinsics["fx"] / cam_point[2]) + self.intrinsics["ppx"]
        v = (cam_point[1] * self.intrinsics["fy"] / cam_point[2]) + self.intrinsics["ppy"]
        return int(round(u)), int(round(v))

    def make_square_roi(self, center_px, radius_px, image_shape):
        cx, cy = center_px
        height, width = image_shape[:2]
        x1 = max(0, cx - radius_px)
        y1 = max(0, cy - radius_px)
        x2 = min(width, cx + radius_px + 1)
        y2 = min(height, cy + radius_px + 1)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"ROI 생성 실패: center=({cx}, {cy}), radius={radius_px}")
        return x1, y1, x2, y2

    def estimate_vertical_grasp_from_depth(self, place_x, place_y, place_z, vertical_pose):
        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("수직 재측정용 depth frame 없음")

        capture_pose = self.get_current_dsr_pose()
        projected_px = self.project_base_to_pixel((place_x, place_y, place_z), capture_pose)
        image_height, image_width = depth_frame.shape[:2]
        if not (0 <= projected_px[0] < image_width and 0 <= projected_px[1] < image_height):
            raise RuntimeError(f"projection pixel이 이미지 밖입니다: {projected_px}")

        center_roi = self.make_square_roi(projected_px, DEPTH_CENTER_ROI_PX, depth_frame.shape)
        outer_roi = self.make_square_roi(projected_px, DEPTH_FLOOR_RING_PX, depth_frame.shape)
        center_points, _ = self.transform_depth_roi_to_base_points(depth_frame, center_roi, capture_pose)
        outer_points, outer_pixels = self.transform_depth_roi_to_base_points(depth_frame, outer_roi, capture_pose)

        if len(center_points) < MIN_DEPTH_POINTS:
            raise RuntimeError(f"물체 후보 ROI depth point 부족: {len(center_points)}개")

        cx1, cy1, cx2, cy2 = center_roi
        ring_mask = (
            (outer_pixels[:, 0] < cx1)
            | (outer_pixels[:, 0] >= cx2)
            | (outer_pixels[:, 1] < cy1)
            | (outer_pixels[:, 1] >= cy2)
        )
        floor_candidates = outer_points[ring_mask]
        if len(floor_candidates) >= MIN_DEPTH_POINTS:
            floor_z = float(np.median(floor_candidates[:, 2]))
            floor_source = "depth"
        else:
            floor_z = REAL_TABLE_Z
            floor_source = "REAL_TABLE_Z fallback"

        object_mask = center_points[:, 2] > (floor_z + OBJECT_MIN_HEIGHT_MM)
        object_points = center_points[object_mask]
        if len(object_points) < MIN_DEPTH_POINTS:
            raise RuntimeError(
                f"물체 후보 point 부족: {len(object_points)}개 "
                f"(floor_z={floor_z:.1f}, source={floor_source})"
            )

        object_top_z = float(np.percentile(object_points[:, 2], OBJECT_TOP_PERCENTILE))
        object_height_mm = object_top_z - floor_z
        if object_height_mm < OBJECT_MIN_HEIGHT_MM or object_height_mm > OBJECT_MAX_HEIGHT_MM:
            raise RuntimeError(
                f"물체 높이 추정값 비정상: height={object_height_mm:.1f}mm, "
                f"floor={floor_z:.1f}, top={object_top_z:.1f}"
            )

        measured_center_x = float(np.median(object_points[:, 0]))
        measured_center_y = float(np.median(object_points[:, 1]))
        center_error = math.hypot(measured_center_x - place_x, measured_center_y - place_y)
        if center_error > VERTICAL_CENTER_CORRECTION_LIMIT_MM:
            raise RuntimeError(
                f"depth 중심 보정량이 너무 큽니다: {center_error:.1f}mm "
                f"(limit={VERTICAL_CENTER_CORRECTION_LIMIT_MM:.1f}mm)"
            )

        base2vertical = self.get_robot_pose_matrix(*vertical_pose)
        object_points_h = np.column_stack((object_points, np.ones(len(object_points))))
        local_points = (np.linalg.inv(base2vertical) @ object_points_h.T).T[:, :3]
        if VERTICAL_GRIP_WIDTH_AXIS == "x":
            width_axis_values = local_points[:, 0]
        elif VERTICAL_GRIP_WIDTH_AXIS == "y":
            width_axis_values = local_points[:, 1]
        else:
            raise RuntimeError(f"지원하지 않는 VERTICAL_GRIP_WIDTH_AXIS: {VERTICAL_GRIP_WIDTH_AXIS}")

        measured_width_mm = float(
            np.percentile(width_axis_values, OBJECT_WIDTH_PERCENTILE_HIGH)
            - np.percentile(width_axis_values, OBJECT_WIDTH_PERCENTILE_LOW)
        )
        vertical_target_width = measured_width_mm + VERTICAL_GRIP_WIDTH_MARGIN_MM
        vertical_target_width_cmd = int(max(0.0, min(vertical_target_width * 10.0, 1100.0)))
        vertical_x = measured_center_x + VERTICAL_X_OFFSET_MM
        vertical_y = measured_center_y + VERTICAL_Y_OFFSET_MM
        vertical_pregrasp_z = object_top_z + VERTICAL_APPROACH_CLEARANCE_MM
        raw_vertical_pick_z = object_top_z - VERTICAL_TOP_GRASP_DEPTH_MM + VERTICAL_DEPTH_Z_BIAS_MM
        vertical_pick_z = max(raw_vertical_pick_z, floor_z + MIN_PICK_ABOVE_FLOOR_MM)

        self.get_logger().info(
            "[Pick_Item] depth 수직 재파지 추정: "
            f"projected_px={projected_px}, floor_z={floor_z:.1f}({floor_source}), "
            f"top_z={object_top_z:.1f}, height={object_height_mm:.1f}, "
            f"center=({measured_center_x:.1f}, {measured_center_y:.1f}), "
            f"center_error={center_error:.1f}, width={measured_width_mm:.1f}, "
            f"gripper={vertical_target_width_cmd / 10.0:.1f}mm, "
            f"pregrasp_z={vertical_pregrasp_z:.1f}, pick_z={vertical_pick_z:.1f}"
        )

        return {
            "vertical_x": vertical_x,
            "vertical_y": vertical_y,
            "vertical_pregrasp_z": vertical_pregrasp_z,
            "vertical_pick_z": vertical_pick_z,
            "vertical_target_width_cmd": vertical_target_width_cmd,
        }

    def get_current_dsr_pose(self):
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            state = scene.current_state
            state.update()
            transform = np.array(state.get_frame_transform(EE_LINK), dtype=float)

        xyz_mm = transform[:3, 3] * 1000.0
        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYZ", degrees=True)
        return [
            float(xyz_mm[0]),
            float(xyz_mm[1]),
            float(xyz_mm[2]),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ]

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

    def make_plan_params(self, planner_id):
        req_params = PlanRequestParameters(self.robot)
        req_params.planning_pipeline = "pilz_industrial_motion_planner"
        req_params.planner_id = planner_id
        req_params.max_velocity_scaling_factor = VELOCITY_SCALE
        req_params.max_acceleration_scaling_factor = ACCELERATION_SCALE
        req_params.planning_time = 2.0
        return req_params

    def build_joint_constraints(self, joint_goal):
        constraints = Constraints()
        for joint_name in JOINT_NAMES:
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = joint_goal[joint_name]
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return [constraints]

    def plan_and_execute_joints(self, joint_goal):
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=self.build_joint_constraints(joint_goal))
            plan_result = self.arm.plan(parameters=self.make_plan_params("PTP"))
            if not plan_result:
                self.get_logger().error(f"[Pick_Item] joint planning failed: {joint_goal}")
                return False

            self.robot.execute(
                group_name=GROUP_NAME,
                robot_trajectory=plan_result.trajectory,
                blocking=True,
            )
            return True
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] joint 계획/실행 중 예외 발생: {e}")
            return False

    def move_to_ready_pose(self):
        self.get_logger().info("[Pick_Item] 카메라 관측 자세(JReady)로 이동 중")
        if not self.plan_and_execute_joints(self.joints_ready):
            return False
        if self.gripper is not None:
            self.gripper.open_gripper()
        self.get_logger().info("[Pick_Item] JReady 이동 완료. YOLO 탐지 대기 자세입니다.")
        return True

    def plan_and_execute_pose(self, dsr_pose, planner_id="PTP"):
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(
                pose_stamped_msg=self.to_pose_stamped(dsr_pose),
                pose_link=EE_LINK,
            )
            plan_result = self.arm.plan(parameters=self.make_plan_params(planner_id))
            if not plan_result:
                self.get_logger().error(f"[Pick_Item] {planner_id} planning failed: {dsr_pose}")
                return False

            self.robot.execute(
                group_name=GROUP_NAME,
                robot_trajectory=plan_result.trajectory,
                blocking=True,
            )
            return True
        except Exception as e:
            self.get_logger().error(f"[Pick_Item] pose 계획/실행 중 예외 발생: {e}")
            return False

    def publish_feedback(self, goal_handle, state):
        feedback_msg = PickItem.Feedback()
        feedback_msg.state = state
        goal_handle.publish_feedback(feedback_msg)

    def abort(self, goal_handle, message):
        self.get_logger().error(message)
        goal_handle.abort()
        return PickItem.Result(success=False)

    def execute_callback(self, goal_handle):
        self.get_logger().info("[Pick_Item] 피킹 시퀀스 시작")

        cx = int(goal_handle.request.center_x)
        cy = int(goal_handle.request.center_y)
        width_px = float(goal_handle.request.width)
        height_px = float(goal_handle.request.height)

        self.publish_feedback(goal_handle, "받은 픽셀 좌표를 기반으로 3D 위치 계산 중")

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None:
            return self.abort(goal_handle, "[Pick_Item] 뎁스 이미지를 가져올 수 없습니다.")
        if not (0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]):
            return self.abort(goal_handle, f"[Pick_Item] 픽셀 좌표가 이미지 범위를 벗어났습니다: ({cx}, {cy})")

        z_dist = float(depth_frame[cy, cx])
        if z_dist <= 0.0:
            return self.abort(goal_handle, "[Pick_Item] 선택된 좌표의 뎁스 값이 0입니다.")

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        capture_pose = self.get_current_dsr_pose()
        cam_coords = (
            (cx - ppx) * z_dist / fx,
            (cy - ppy) * z_dist / fy,
            z_dist,
        )
        base_xyz = self.transform_to_base(cam_coords, capture_pose)

        obj_width_mm = width_px * z_dist / fx
        obj_height_mm = height_px * z_dist / fy
        target_width = int(obj_width_mm * 10.0 * SQUEEZE_RATIO)
        target_width = max(0, min(target_width, 1100))

        self.get_logger().info(
            "[Pick_Item] 변환 완료 -> "
            f"base=({base_xyz[0]:.1f}, {base_xyz[1]:.1f}, {base_xyz[2]:.1f})mm, "
            f"width={obj_width_mm:.1f}mm, height={obj_height_mm:.1f}mm, "
            f"gripper={target_width / 10.0:.1f}mm"
        )

        try:
            self.execute_pick_motion(*base_xyz, target_width, obj_height_mm, goal_handle)
        except Exception as e:
            return self.abort(goal_handle, f"[Pick_Item] 로봇 구동 중 에러 발생: {e}")

        goal_handle.succeed()
        self.get_logger().info("[Pick_Item] 전체 피킹/안착/수직 재피킹 작업 성공")
        return PickItem.Result(success=True)

    def execute_pick_motion(self, x, y, z, target_width, obj_height_mm, goal_handle):
        if self.gripper is None:
            raise RuntimeError("그리퍼가 연결되지 않았습니다.")

        current_pose = self.get_current_dsr_pose()
        fixed_rx, fixed_ry, fixed_rz = current_pose[3:]

        self.gripper.open_gripper()
        time.sleep(0.5)

        wait_x = x - (X_OFFSET + SIDE_APPROACH_DIST)
        pick_x = x - X_OFFSET + 5.0

        approach_pose = [wait_x, y, z, fixed_rx, fixed_ry, fixed_rz]
        pick_pose = [pick_x, y, z, fixed_rx, fixed_ry, fixed_rz]

        self.publish_feedback(goal_handle, "물품 정면 접근 위치로 이동 중")
        if not self.plan_and_execute_pose(approach_pose, planner_id="PTP"):
            raise RuntimeError("접근 대기 위치 이동 실패")
        time.sleep(0.2)

        self.publish_feedback(goal_handle, "그립 처리를 위해 물품으로 진입 중")
        if not self.plan_and_execute_pose(pick_pose, planner_id="LIN"):
            raise RuntimeError("수평 진입 이동 실패")
        time.sleep(0.2)

        self.publish_feedback(goal_handle, f"물품 파지 중 (실측 보정 너비: {target_width / 10.0:.1f}mm)")
        self.gripper.move_gripper(width_val=target_width)
        time.sleep(1.2)

        self.publish_feedback(goal_handle, "파지 완료 후 후방으로 후퇴 중")
        if not self.plan_and_execute_pose(approach_pose, planner_id="LIN"):
            raise RuntimeError("후퇴 이동 실패")
        time.sleep(0.2)

        place_x, place_y = approach_pose[0], approach_pose[1]
        place_z = REAL_TABLE_Z + (obj_height_mm / 2.0) + PLACE_Z_BIAS_MM
        place_pose = [place_x, place_y, place_z, fixed_rx, fixed_ry, fixed_rz]

        self.publish_feedback(goal_handle, "후퇴 위치에서 Z축만 내려 물품을 바닥에 놓는 중")
        if not self.plan_and_execute_pose(place_pose, planner_id="LIN"):
            raise RuntimeError("바닥 내려놓기 이동 실패")
        time.sleep(0.5)

        self.publish_feedback(goal_handle, "그리퍼 개방 중")
        self.gripper.open_gripper()
        time.sleep(1.0)

        lift_after_place_pose = list(place_pose)
        lift_after_place_pose[2] = place_z + VERTICAL_LIFT_BEFORE_ROTATE_MM
        self.publish_feedback(goal_handle, "수직 자세 전환 전 Z축 상승 중")
        if not self.plan_and_execute_pose(lift_after_place_pose, planner_id="LIN"):
            raise RuntimeError("수직 자세 전환 전 상승 실패")
        time.sleep(0.2)

        vertical_safe_pose = [place_x, place_y, lift_after_place_pose[2], *VERTICAL_RPY]
        self.publish_feedback(goal_handle, "그리퍼를 수직 방향으로 전환 중")
        if not self.plan_and_execute_pose(vertical_safe_pose, planner_id="PTP"):
            raise RuntimeError("수직 자세 전환 실패")
        time.sleep(0.5)

        self.publish_feedback(goal_handle, "Depth 기반으로 물품 중심과 높이를 재측정 중")
        vertical_grasp = self.estimate_vertical_grasp_from_depth(
            place_x,
            place_y,
            place_z,
            vertical_safe_pose,
        )

        vertical_pregrasp_pose = [
            vertical_grasp["vertical_x"],
            vertical_grasp["vertical_y"],
            vertical_grasp["vertical_pregrasp_z"],
            *VERTICAL_RPY,
        ]
        vertical_pick_pose = [
            vertical_grasp["vertical_x"],
            vertical_grasp["vertical_y"],
            vertical_grasp["vertical_pick_z"],
            *VERTICAL_RPY,
        ]

        self.publish_feedback(goal_handle, "Depth 측정 위치 기준으로 물품 바로 위로 이동 중")
        if not self.plan_and_execute_pose(vertical_pregrasp_pose, planner_id="PTP"):
            raise RuntimeError("수직 재파지 상단 이동 실패")
        time.sleep(0.2)

        self.publish_feedback(goal_handle, "수직 방향으로 물품을 다시 파지 중")
        if not self.plan_and_execute_pose(vertical_pick_pose, planner_id="LIN"):
            raise RuntimeError("수직 파지 위치 하강 실패")
        time.sleep(0.2)

        self.gripper.move_gripper(width_val=vertical_grasp["vertical_target_width_cmd"])
        time.sleep(1.2)

        final_lift_pose = list(vertical_pick_pose)
        final_lift_pose[2] += FINAL_LIFT_MM
        self.publish_feedback(goal_handle, f"수직 피킹 완료 후 {FINAL_LIFT_MM:.0f}mm 상승 중")
        if not self.plan_and_execute_pose(final_lift_pose, planner_id="LIN"):
            raise RuntimeError("최종 상승 이동 실패")
        time.sleep(0.5)

    def run(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.add_node(self.img_node)
        executor.spin()


def main(args=None):
    rclpy.init(args=args)
    server = PickItemServer()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        server.img_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

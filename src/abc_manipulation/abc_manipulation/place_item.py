import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from abc_interfaces.action import PlaceItem
from abc_manipulation.onrobot import RG


GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK = "link_6"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

VELOCITY_SCALE = 0.3
ACCELERATION_SCALE = 0.3
SAFE_X_MIN = 0.0
SAFE_Y_MIN = -300.0
SAFE_Y_MAX = 300.0
SAFE_Z_MIN = 270.0
SAFE_TRAVEL_Z = 400.0

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502


class PlaceItemServer(Node):
    def __init__(self):
        super().__init__("place_item_action_helper")
        cb_group = ReentrantCallbackGroup()

        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info("[Place_Item] 그리퍼 연결 성공")
        except Exception as e:
            self.get_logger().error(f"[Place_Item] 그리퍼 연결 실패: {e}")
            self.gripper = None

        try:
            self.robot = MoveItPy(node_name="place_item_server")
            self.arm = self.robot.get_planning_component(GROUP_NAME)
            self.get_logger().info("[Place_Item] MoveItPy 엔진 로드 완료")
        except Exception as e:
            self.get_logger().error(f"[Place_Item] MoveItPy 초기화 실패: {e}")
            sys.exit(1)
        '''
        self.joints_home = {
            "joint_1": math.radians(0.0),
            "joint_2": math.radians(-20.0),
            "joint_3": math.radians(120.0),
            "joint_4": math.radians(0.0),
            "joint_5": math.radians(15.0),
            "joint_6": math.radians(90.0),
        }
        '''
        self.joints_home = {
            "joint_1": math.radians(0.0),
            "joint_2": math.radians(0.0),
            "joint_3": math.radians(135.0),
            "joint_4": math.radians(0.0),
            "joint_5": math.radians(-45.0),
            "joint_6": math.radians(90.0),
        }

        self.pos_checkout = [408.0, 153.0, 342.0, 33.0, 180.0, 100.0]
        self.pos_return = [408.0, -153.0, 342.0, 133.0, -172.0, -120.0]
        self.safe_z_offset = 100.0

        self._action_server = ActionServer(
            self,
            PlaceItem,
            "place_item",
            self.execute_callback,
            callback_group=cb_group,
        )

        self.get_logger().info("[Place_Item] Place Item 서버가 시작되었습니다.")

    def clamp_to_safe_workspace(self, pose):
        safe_pose = list(pose)

        if safe_pose[0] < SAFE_X_MIN:
            self.get_logger().warning(
                f"[Place_Item] X 안전 한계 보정: {safe_pose[0]:.1f} -> {SAFE_X_MIN:.1f} mm"
            )
            safe_pose[0] = SAFE_X_MIN

        if safe_pose[1] < SAFE_Y_MIN:
            self.get_logger().warning(
                f"[Place_Item] Y 안전 한계 보정: {safe_pose[1]:.1f} -> {SAFE_Y_MIN:.1f} mm"
            )
            safe_pose[1] = SAFE_Y_MIN
        elif safe_pose[1] > SAFE_Y_MAX:
            self.get_logger().warning(
                f"[Place_Item] Y 안전 한계 보정: {safe_pose[1]:.1f} -> {SAFE_Y_MAX:.1f} mm"
            )
            safe_pose[1] = SAFE_Y_MAX

        if safe_pose[2] < SAFE_Z_MIN:
            self.get_logger().warning(
                f"[Place_Item] Z 안전 한계 보정: {safe_pose[2]:.1f} -> {SAFE_Z_MIN:.1f} mm"
            )
            safe_pose[2] = SAFE_Z_MIN

        return safe_pose

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
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = joint_goal[joint_name]
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)
        return [constraints]

    def make_plan_params(self, planner_id):
        req_params = PlanRequestParameters(self.robot)
        req_params.planning_pipeline = "pilz_industrial_motion_planner"
        req_params.planner_id = planner_id
        req_params.max_velocity_scaling_factor = VELOCITY_SCALE
        req_params.max_acceleration_scaling_factor = ACCELERATION_SCALE
        req_params.planning_time = 2.0
        return req_params

    def plan_and_execute_pose(self, dsr_pose, planner_id="LIN"):
        try:
            pose_stamped = self.to_pose_stamped(self.clamp_to_safe_workspace(dsr_pose))
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=EE_LINK)
            return self.plan_and_execute(self.make_plan_params(planner_id))
        except Exception as e:
            self.get_logger().error(f"[Place_Item] pose 계획/실행 중 예외 발생: {e}")
            return False

    def plan_and_execute_joints(self, joint_goal):
        try:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(motion_plan_constraints=self.build_joint_constraints(joint_goal))
            return self.plan_and_execute(self.make_plan_params("PTP"))
        except Exception as e:
            self.get_logger().error(f"[Place_Item] joint 계획/실행 중 예외 발생: {e}")
            return False

    def plan_and_execute(self, req_params):
        plan_result = self.arm.plan(parameters=req_params)
        if not plan_result:
            self.get_logger().error("[Place_Item] Planning failed")
            return False

        self.robot.execute(
            group_name=GROUP_NAME,
            robot_trajectory=plan_result.trajectory,
            blocking=True,
        )
        return True

    def abort(self, goal_handle, message):
        self.get_logger().error(message)
        goal_handle.abort()
        result = PlaceItem.Result()
        result.success = False
        return result

    def execute_callback(self, goal_handle):
        is_match = goal_handle.request.is_corrected
        dest_name = "판매대" if is_match else "반품대"
        self.get_logger().info(f"\n[Place_Item 시작] 검증 결과({is_match})에 따라 {dest_name}로 이송합니다.")

        feedback_msg = PlaceItem.Feedback()

        target_pose = self.clamp_to_safe_workspace(self.pos_checkout if is_match else self.pos_return)
        approach_target = list(target_pose)
        approach_target[2] = max(approach_target[2] + self.safe_z_offset, SAFE_TRAVEL_Z)

        feedback_msg.state = f"{dest_name} 상단으로 접근 중..."
        goal_handle.publish_feedback(feedback_msg)
        if not self.plan_and_execute_pose(approach_target, planner_id="LIN"):
            return self.abort(goal_handle, "[Place_Item] 목적지 상단 이동 실패")
        time.sleep(0.5)

        feedback_msg.state = "물체 내려놓는 중..."
        goal_handle.publish_feedback(feedback_msg)
        if not self.plan_and_execute_pose(target_pose, planner_id="LIN"):
            return self.abort(goal_handle, "[Place_Item] 내려놓기 위치 이동 실패")
        time.sleep(0.5)

        if self.gripper is not None:
            self.gripper.open_gripper()
        time.sleep(0.5)

        feedback_msg.state = "작업 완료 후 홈 복귀 중..."
        goal_handle.publish_feedback(feedback_msg)
        if not self.plan_and_execute_pose(approach_target, planner_id="LIN"):
            return self.abort(goal_handle, "[Place_Item] 후퇴 이동 실패")
        if not self.plan_and_execute_joints(self.joints_home):
            return self.abort(goal_handle, "[Place_Item] 홈 위치 복귀 실패")

        goal_handle.succeed()
        result = PlaceItem.Result()
        result.success = True
        self.get_logger().info(f"[Place_Item 완료] {dest_name}에 배치 성공")
        return result


def main(args=None):
    rclpy.init(args=args)
    place_server = PlaceItemServer()
    executor = MultiThreadedExecutor()
    executor.add_node(place_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        place_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

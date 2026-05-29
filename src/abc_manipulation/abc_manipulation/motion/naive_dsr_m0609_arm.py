from typing import Dict, Mapping

from geometry_msgs.msg import PoseStamped

from DSR_ROBOT2 import get_current_posj, movej, movel
from DR_common2 import posx, posj

from abc_manipulation.motion.robot_arm import RobotArm, MotionResult


JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)


def joints_dict_generator(
    joint_1: float,
    joint_2: float,
    joint_3: float,
    joint_4: float,
    joint_5: float,
    joint_6: float,
) -> Dict[str, float]:
    return {
        "joint_1": joint_1,
        "joint_2": joint_2,
        "joint_3": joint_3,
        "joint_4": joint_4,
        "joint_5": joint_5,
        "joint_6": joint_6,
    }


def posj_to_dict(joint_pos) -> Dict[str, float]:
    values = list(joint_pos)

    if len(values) != 6:
        raise ValueError(f"Expected 6 joint values, got {len(values)}")

    return {
        name: float(value)
        for name, value in zip(JOINT_NAMES, values)
    }


class DSRM0609Arm(RobotArm):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def move_to_joint(
        self,
        joints: Mapping[str, float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        try:
            missing = [name for name in JOINT_NAMES if name not in joints]
            if missing:
                return MotionResult(
                    success=False,
                    message=f"Missing joint values: {missing}",
                )

            target_joint = posj(
                joints["joint_1"],
                joints["joint_2"],
                joints["joint_3"],
                joints["joint_4"],
                joints["joint_5"],
                joints["joint_6"],
            )

            movej(target_joint, velocity_scale, acceleration_scale)

            return MotionResult(success=True)

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"move_to_joint failed: {e}")

            return MotionResult(success=False, message=str(e))

    def move_to_pose(
        self,
        pose: PoseStamped,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        try:
            x_mm = pose.pose.position.x * 1000.0
            y_mm = pose.pose.position.y * 1000.0
            z_mm = pose.pose.position.z * 1000.0

            # TODO: orientation 변환 필요
            rx = 0.0
            ry = 0.0
            rz = 0.0

            target_pose = posx(x_mm, y_mm, z_mm, rx, ry, rz)

            movel(target_pose, velocity_scale, acceleration_scale)

            return MotionResult(success=True)

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"move_to_pose failed: {e}")

            return MotionResult(success=False, message=str(e))

    def get_current_joints(self) -> Dict[str, float]:
        try:
            current_joint = get_current_posj()
            return posj_to_dict(current_joint)

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"get_current_joints failed: {e}")

            raise
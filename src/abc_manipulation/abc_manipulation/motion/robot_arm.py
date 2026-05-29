# abc_manipulation/arm/robot_arm.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from geometry_msgs.msg import PoseStamped


@dataclass
class MotionResult:
    success: bool
    message: str = ""
    error_code: Optional[int] = None


class RobotArm(ABC):
    @abstractmethod
    def move_named(
        self,
        name: str,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        """
        미리 정의된 named pose로 이동한다.

        예:
            arm.move_named("home")
            arm.move_named("ready")
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_joint(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def move_joint(
        self,
        joints: Mapping[str, float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def move_pose(
        self,
        pose: PoseStamped,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def move_linear(
        self,
        pose: PoseStamped,
        velocity_scale: float = 0.1,
        acceleration_scale: float = 0.1,
    ) -> MotionResult:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> MotionResult:
        raise NotImplementedError


    def move_to_named(
        self,
        name: str,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        return self.move_named(
            name=name,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )

    def move_to_joint(
        self,
        joints: Dict[str, float],
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        return self.move_joint(
            joints=joints,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )

    def move_to_pose(
        self,
        pose: PoseStamped,
        velocity_scale: float = 0.3,
        acceleration_scale: float = 0.3,
    ) -> MotionResult:
        return self.move_pose(
            pose=pose,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )
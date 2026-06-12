# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Marvin controller backed directly by the official Marvin SDK."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .marvin_robot_state import MarvinRobotState
from .marvin_sdk import (
    CARTESIAN_IMPEDANCE_CONTROL_MODE,
    JOINT_IMPEDANCE_CONTROL_MODE,
    POSITION_CONTROL_MODE,
    Marvin,
)
from .marvin_usb_gripper import MarvinUsbGripper


def _default_marvin_arm() -> str:
    return os.getenv("MARVIN_ARM_ID", "B").upper()


def _default_gripper_backend() -> str:
    return os.getenv("MARVIN_GRIPPER_BACKEND", "sdk").lower().strip()


class MarvinController(Worker):
    """Single-arm Marvin controller using the official Marvin SDK."""

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        ros_pkg: str = "serl_marvin_controllers",
        arm_id: str | None = None,
    ):
        """Launch a MarvinController on the specified worker's node."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return MarvinController.create_group(
            robot_ip, ros_pkg, arm_id or _default_marvin_arm()
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"MarvinController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        ros_pkg: str = "serl_marvin_controllers",
        arm_id: str = "B",
    ):
        """Initialize the Marvin controller."""
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._arm_id = arm_id.upper()
        self._state = MarvinRobotState()
        self._client = Marvin(robot_ip=robot_ip, log_switch=0)
        self._gripper = None
        if _default_gripper_backend() == "usb":
            self._gripper = MarvinUsbGripper.from_env(self._arm_id)
        self._active_control_mode = POSITION_CONTROL_MODE
        default_control_mode = os.getenv(
            "MARVIN_CONTROL_MODE", CARTESIAN_IMPEDANCE_CONTROL_MODE
        )
        self.start_controller(default_control_mode)

    def _update_state(self) -> MarvinRobotState:
        self._state = self._client.get_state(self._arm_id)
        return self._state

    def _wait_robot(self, sleep_time: float = 0.3) -> None:
        time.sleep(sleep_time)

    def _wait_for_joint(
        self, target_pos_rad: np.ndarray, timeout: float = 15.0
    ) -> None:
        wait_time = 0.02
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self._client.get_state(self._arm_id).arm_joint_position
            if np.allclose(target_pos_rad, current, atol=1e-2, rtol=1e-2):
                return
            time.sleep(wait_time)
        self._logger.warning("Joint position wait timeout exceeded.")

    def start_impedance(self) -> None:
        """Switch to cartesian impedance mode."""
        self._client.set_cartesian_impedance_mode(self._arm_id)
        self._active_control_mode = CARTESIAN_IMPEDANCE_CONTROL_MODE
        self._wait_robot()
        self.log_debug("Start cartesian impedance controller")

    def start_controller(self, controller_type: str) -> None:
        """Switch control mode."""
        controller_type = Marvin.normalize_control_mode(controller_type)
        if controller_type == CARTESIAN_IMPEDANCE_CONTROL_MODE:
            self._client.set_cartesian_impedance_mode(self._arm_id)
        elif controller_type == JOINT_IMPEDANCE_CONTROL_MODE:
            self._client.set_joint_impedance_mode(self._arm_id)
        elif controller_type == POSITION_CONTROL_MODE:
            self._client.set_position_mode(self._arm_id)
        else:
            raise ValueError(f"Unsupported Marvin controller type: {controller_type}")
        self._active_control_mode = controller_type
        self._wait_robot()
        self.log_debug(f"Start controller: {controller_type}")

    def stop_impedance(self) -> None:
        """Switch back to position mode."""
        self._client.set_position_mode(self._arm_id)
        self._active_control_mode = POSITION_CONTROL_MODE
        self._wait_robot()
        self.log_debug("Stop impedance controller")

    def clear_errors(self) -> None:
        """Clear robot errors."""
        self._client.clear_errors(self._arm_id)

    def reconfigure_compliance_params(self, params: dict[str, Any]) -> None:
        """Update cached TJ motion parameters and reapply when needed."""
        self._client.update_profile(self._arm_id, params)
        self.log_debug(f"Reconfigure compliance parameters: {params}")

    def reconfigure_precision_params(self, params: dict[str, Any]) -> None:
        """Update cached precision parameters and reapply when needed."""
        self._client.update_profile(self._arm_id, params)
        self.log_debug(f"Reconfigure precision parameters: {params}")

    def is_robot_up(self) -> bool:
        """Check if the target arm is streaming state data."""
        return self._client.is_robot_up(self._arm_id)

    def get_state(self) -> MarvinRobotState:
        """Get the current state of the selected arm."""
        return self._update_state()

    def reset_joint(self, reset_pos: list[float]) -> None:
        """Reset the arm to the desired joint position in radians."""
        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dimensions but got {len(reset_pos)}"
        )
        restore_mode = self._active_control_mode
        self.clear_errors()
        self._client.move_joint_positions(
            self._arm_id,
            np.asarray(reset_pos, dtype=np.float64),
            control_mode=POSITION_CONTROL_MODE,
        )
        self._wait_for_joint(np.asarray(reset_pos, dtype=np.float64))
        if restore_mode != POSITION_CONTROL_MODE:
            self.start_controller(restore_mode)

    def move_arm(self, position: np.ndarray) -> None:
        """Move the selected arm to a 7D cartesian pose [x, y, z, qx, qy, qz, qw]."""
        assert len(position) == 7, (
            f"Invalid position, expected 7 dimensions but got {len(position)}"
        )
        self._client.move_pose(
            self._arm_id,
            np.asarray(position, dtype=np.float64),
            control_mode=self._active_control_mode,
        )
        self.log_debug(f"Move arm to position: {position}")

    def move_joint_positions(self, joints_rad: np.ndarray) -> None:
        """Move the selected arm to a 7D joint position in radians."""
        assert len(joints_rad) == 7, (
            f"Invalid joint position, expected 7 dimensions but got {len(joints_rad)}"
        )
        self._client.move_joint_positions(
            self._arm_id,
            np.asarray(joints_rad, dtype=np.float64),
            control_mode=POSITION_CONTROL_MODE,
        )
        self.log_debug(f"Move arm to joint positions: {joints_rad}")

    def fk_tcp_pose(self, joints_rad: np.ndarray) -> np.ndarray:
        """Compute the selected arm TCP pose from 7D joint positions in radians."""
        assert len(joints_rad) == 7, (
            f"Invalid joint position, expected 7 dimensions but got {len(joints_rad)}"
        )
        return self._client.fk_tcp_pose(
            self._arm_id,
            np.asarray(joints_rad, dtype=np.float64),
        )

    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        """Move the configured end-effector gripper."""
        if position >= 128:
            self.open_gripper()
        else:
            self.close_gripper()

    def open_gripper(self) -> None:
        """Open the configured end-effector gripper."""
        if self._gripper is not None:
            ok = self._gripper.open()
        else:
            ok = self._client.open_gripper(self._arm_id)
        if ok:
            self._state.gripper_open = True
        self.log_debug("Open gripper")

    def close_gripper(self) -> None:
        """Close the configured end-effector gripper."""
        if self._gripper is not None:
            ok = self._gripper.close()
        else:
            ok = self._client.close_gripper(self._arm_id)
        if ok:
            self._state.gripper_open = False
        self.log_debug("Close gripper")

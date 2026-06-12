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

"""PICO/VR arm teleop source.

Wraps :class:`XrClient` + :class:`PicoArmTeleopController`. A single
``controller.step`` per tick yields the target pose and the same frame's
``trigger`` / ``grip``, which are forwarded via ``aux`` so a paired hand/gripper
teleop reuses the same device read.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.teleop.protocols import ArmReading, ArmTeleop
from rlinf.envs.realworld.common.vr import (
    PicoArmTeleopController,
    XrClient,
    limit_pose_step,
    pose7_to_matrix,
)


class VRArmTeleop(ArmTeleop):
    """Clutch-based PICO arm teleoperation producing a 6-DOF delta."""

    def __init__(
        self,
        *,
        side: str = "left",
        workspace_limits: dict[str, tuple[float, float]] | None = None,
        ema_trans: float = 0.30,
        ema_rot: float = 0.30,
        translation_scale: float = 1.0,
        xyz_scale: list[float] | None = None,
        track_rotation: bool = False,
        max_step_m: float = 0.004,
        max_rot_deg: float = 1.0,
        dummy_on_missing: bool = False,
        motion_threshold: float = 0.001,
    ) -> None:
        self._xr = XrClient(dummy_on_missing=dummy_on_missing)
        self._xr.init()
        controller_kwargs: dict[str, Any] = {
            "side": side,
            "ema_trans": ema_trans,
            "ema_rot": ema_rot,
            "translation_scale": translation_scale,
            "xyz_scale": xyz_scale,
            "track_rotation": track_rotation,
        }
        if workspace_limits is not None:
            controller_kwargs["workspace_limits"] = workspace_limits
        self._controller = PicoArmTeleopController(self._xr, **controller_kwargs)
        self._max_step_m = float(max_step_m)
        self._max_rot_deg = float(max_rot_deg)
        self._motion_threshold = float(motion_threshold)
        self._last_target_T: np.ndarray | None = None
        self._active = False

    def reset(self) -> None:
        self._last_target_T = None
        self._active = False

    def read(self, tcp_pose7: np.ndarray, action_scale: np.ndarray) -> ArmReading:
        T_tcp = pose7_to_matrix(np.asarray(tcp_pose7, dtype=np.float64))
        command = self._controller.step(T_tcp)

        T_target = command.T_target
        if command.active:
            T_last = self._last_target_T if self._last_target_T is not None else T_tcp
            T_target = limit_pose_step(
                T_target,
                T_last,
                max_step_m=self._max_step_m,
                max_rot_deg=self._max_rot_deg,
            )
            self._last_target_T = T_target.copy()
        else:
            self._last_target_T = T_tcp.copy()

        delta6 = self._target_pose_to_delta(T_tcp, T_target, action_scale)
        self._active = bool(command.active)
        active = bool(command.active and np.linalg.norm(delta6) > self._motion_threshold)
        aux: Dict[str, Any] = {
            "trigger": float(command.trigger),
            "grip": float(command.grip),
            "vr_active": bool(command.active),
        }
        return ArmReading(delta6=delta6, active=active, aux=aux)

    @staticmethod
    def _target_pose_to_delta(
        T_tcp: np.ndarray, T_target: np.ndarray, action_scale: np.ndarray
    ) -> np.ndarray:
        action_scale = np.asarray(action_scale, dtype=np.float64)
        delta_pos = (T_target[:3, 3] - T_tcp[:3, 3]) / action_scale[0]
        r_tcp = R.from_matrix(T_tcp[:3, :3])
        r_target = R.from_matrix(T_target[:3, :3])
        delta_euler = (r_target * r_tcp.inv()).as_euler("xyz") / action_scale[1]
        return np.clip(np.concatenate([delta_pos, delta_euler]), -1.0, 1.0)

    def info(self) -> Dict[str, Any]:
        return {"vr_active": self._active}

    def close(self) -> None:
        self._xr.close()

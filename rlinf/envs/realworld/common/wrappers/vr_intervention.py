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

"""PICO/VR teleoperation intervention wrapper."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.vr import (
    PicoArmTeleopController,
    XrClient,
    limit_pose_step,
    pose7_to_matrix,
)


class VRTeleopIntervention(gym.ActionWrapper):
    """Override policy actions with clutch-based PICO arm teleoperation."""

    def __init__(
        self,
        env: gym.Env,
        *,
        gripper_enabled: bool = True,
        side: str = "left",
        workspace_limits: dict[str, tuple[float, float]] | None = None,
        ema_trans: float = 0.30,
        ema_rot: float = 0.30,
        translation_scale: float = 1.0,
        xyz_scale: list[float] | None = None,
        track_rotation: bool = False,
        max_step_m: float = 0.004,
        max_rot_deg: float = 1.0,
        hold_time: float = 0.5,
        dummy_on_missing: bool = False,
        gripper_threshold: float = 0.6,
    ) -> None:
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
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
        self._hold_time = float(hold_time)
        self._gripper_threshold = float(gripper_threshold)
        self._last_intervene = 0.0
        self._last_target_T: np.ndarray | None = None
        self._last_gripper_closed: bool | None = None
        self._active = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_intervene = 0.0
        self._last_target_T = None
        self._last_gripper_closed = None
        self._active = False
        return obs, info

    def close(self) -> None:
        self._xr.close()
        return super().close()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
        T_tcp = pose7_to_matrix(tcp_pose)
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

        expert_a = self._target_pose_to_action(T_tcp, T_target)
        gripper_action = self._gripper_action(command.trigger)
        if self.gripper_enabled:
            expert_a = np.concatenate([expert_a, np.array([gripper_action])])

        arm_active = command.active and np.linalg.norm(expert_a[:6]) > 0.001
        gripper_active = abs(gripper_action) > 0.5
        self._active = command.active
        if arm_active or gripper_active:
            self._last_intervene = time.time()

        if time.time() - self._last_intervene < self._hold_time:
            return expert_a.astype(action.dtype, copy=False), True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        info["vr_active"] = self._active
        return obs, rew, done, truncated, info

    def _target_pose_to_action(self, T_tcp: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        action_scale = self.get_wrapper_attr("get_action_scale")()
        delta_pos = (T_target[:3, 3] - T_tcp[:3, 3]) / action_scale[0]
        r_tcp = R.from_matrix(T_tcp[:3, :3])
        r_target = R.from_matrix(T_target[:3, :3])
        delta_euler = (r_target * r_tcp.inv()).as_euler("xyz") / action_scale[1]
        return np.clip(np.concatenate([delta_pos, delta_euler]), -1.0, 1.0)

    def _gripper_action(self, trigger: float) -> float:
        gripper_closed = trigger >= self._gripper_threshold
        if self._last_gripper_closed is None:
            self._last_gripper_closed = gripper_closed
            return 0.0
        if gripper_closed == self._last_gripper_closed:
            return 0.0
        self._last_gripper_closed = gripper_closed
        if gripper_closed:
            return -1.0
        return 1.0

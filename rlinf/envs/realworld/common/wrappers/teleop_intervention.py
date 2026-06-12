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

"""Generic, composable teleoperation intervention wrapper.

Composes one :class:`ArmTeleop` (6-DOF arm delta) with one
:class:`EndEffectorTeleop` (end-effector command) and centralizes the
hold-time / merge logic shared by the device-specific wrappers.

The arm teleop is polled once per tick and its ``aux`` (trigger/grip/buttons)
is handed to the end-effector teleop, so both halves see the same device frame.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.teleop.protocols import ArmTeleop, EndEffectorTeleop


class TeleopInterventionWrapper(gym.ActionWrapper):
    """Override policy actions with composed arm + end-effector teleoperation.

    Args:
        env: Wrapped env. Its action space must be ``6 + ee_teleop.action_dim``.
        arm_teleop: Arm delta source.
        ee_teleop: End-effector command source.
        hold_time: Seconds to keep returning the teleop action after the last
            active frame (debounce so brief device gaps don't drop control).
        hold_ee_on_release: When teleop is released, keep the policy's arm
            action but hold the last end-effector command (avoids a dexterous
            hand snapping back to the policy value between interventions).
    """

    def __init__(
        self,
        env: gym.Env,
        arm_teleop: ArmTeleop,
        ee_teleop: EndEffectorTeleop,
        *,
        hold_time: float = 0.5,
        hold_ee_on_release: bool = True,
    ) -> None:
        super().__init__(env)
        self._arm = arm_teleop
        self._ee = ee_teleop
        self._hold_time = float(hold_time)
        self._hold_ee_on_release = bool(hold_ee_on_release)

        expected = 6 + int(ee_teleop.action_dim)
        assert self.action_space.shape == (expected,), (
            f"TeleopInterventionWrapper expects a {expected}-D action space "
            f"(6 arm + {ee_teleop.action_dim} end-effector), got "
            f"{self.action_space.shape}."
        )

        self._last_intervene = 0.0
        self._last_ee_command: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._arm.reset()
        self._ee.reset()
        self._last_intervene = 0.0
        self._last_ee_command = None
        return obs, info

    def close(self) -> None:
        self._arm.close()
        self._ee.close()
        return super().close()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
        action_scale = self.get_wrapper_attr("get_action_scale")()

        arm_reading = self._arm.read(tcp_pose, action_scale)
        ee_reading = self._ee.compute(arm_reading.aux)

        ee_command = np.asarray(ee_reading.command, dtype=np.float64).reshape(-1)
        self._last_ee_command = ee_command.copy()

        expert = np.concatenate([np.asarray(arm_reading.delta6, dtype=np.float64), ee_command])

        if arm_reading.active or ee_reading.active:
            self._last_intervene = time.time()

        if time.time() - self._last_intervene < self._hold_time:
            return expert.astype(action.dtype, copy=False), True

        # Released: keep the policy arm action, optionally holding the last EE
        # command so a dexterous hand does not snap to the policy value.
        fallback = np.asarray(action, dtype=np.float64).copy()
        if self._hold_ee_on_release and self._last_ee_command is not None:
            fallback[6:] = self._last_ee_command
        return fallback.astype(action.dtype, copy=False), False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        info.update(self._arm.info())
        info.update(self._ee.info())
        return obs, rew, done, truncated, info

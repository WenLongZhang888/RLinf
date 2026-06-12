# Copyright 2025 The RLinf Authors.
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

"""Wrappers for Marvin env, e.g. SpacemouseIntervention with configurable axis mapping."""

from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert

# Default axis remap for Marvin realworld teleop.
# Aligned with tianji_hilserl:
# x is inverted, SpaceMouse z drives robot y, and SpaceMouse y drives robot z.
SPACEMOUSE_WIRELESS_REMAP: list[tuple[int, int]] = [
    (0, -1),
    (2, 1),
    (1, 1),
    (3, 1),
    (4, 1),
    (5, 1),
]


class SpacemouseIntervention(gym.ActionWrapper):
    """Action wrapper: use Spacemouse action when non-zero, else policy action.

    Supports configurable axis_remap so that Spacemouse xyz/rpy can be mapped
    to the robot arm frame (e.g. when device axes differ from arm definition).
    """

    def __init__(
        self,
        env: gym.Env,
        action_indices: Optional[list[int]] = None,
        axis_mapping: Optional[list[int]] = None,
        axis_remap: Optional[list[tuple[int, int]]] = None,
    ):
        super().__init__(env)
        if axis_remap is None:
            axis_remap = SPACEMOUSE_WIRELESS_REMAP
        self._axis_remap = axis_remap
        self._axis_mapping = axis_mapping
        self.gripper_enabled = self.action_space.shape[0] == 7
        self.expert = SpaceMouseExpert(
            axis_mapping=axis_mapping, axis_remap=axis_remap
        )
        self.left = False
        self.right = False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (action_to_use, was_intervention)."""
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)[:2]
        intervened = False

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = action.copy()
            filtered_expert_a[self.action_indices] = expert_a[
                self.action_indices
            ]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, terminated, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, terminated, truncated, info

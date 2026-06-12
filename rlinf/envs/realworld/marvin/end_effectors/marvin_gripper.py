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

"""Marvin built-in binary gripper as an :class:`EndEffector`.

This is an adapter: it does not re-implement the gripper open/close logic.
Instead it delegates to the env-level ``gripper_fn`` (``MarvinEnv._gripper_action``)
so that all existing semantics — binary hysteresis, ``gripper_sleep`` rate
limiting, subclass overrides like ``PegInsertionEnv.fixed_gripper_closed`` and
direct task calls (``self._gripper_action(1)``) — are preserved unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from rlinf.envs.realworld.common.end_effectors.base import EndEffector


class MarvinGripper(EndEffector):
    """Binary gripper adapter over ``MarvinEnv._gripper_action``.

    Args:
        gripper_fn: Callable taking a scaled scalar position and returning
            ``True`` when the gripper state actually changed (the contract of
            ``MarvinEnv._gripper_action``).
        state_getter: Optional callable returning the current robot state
            object exposing ``gripper_position`` (and ``gripper_open``). Used
            only for :meth:`get_state`.
        action_scale: Gripper action scale (``config.action_scale[2]``). May be
            a float or a zero-arg callable returning the current scale so live
            config edits are honored.
    """

    def __init__(
        self,
        gripper_fn: Callable[[float], bool],
        state_getter: Optional[Callable[[], object]] = None,
        action_scale: float | Callable[[], float] = 1.0,
    ) -> None:
        self._gripper_fn = gripper_fn
        self._state_getter = state_getter
        self._action_scale = action_scale

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_mode(self) -> str:
        return "binary"

    @property
    def finger_names(self) -> list[str]:
        return ["gripper"]

    # ------------------------------------------------------------------
    # Lifecycle (nothing to do; the arm controller owns the hardware)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        if self._state_getter is None:
            return np.zeros(1, dtype=np.float64)
        state = self._state_getter()
        return np.array([getattr(state, "gripper_position", 0.0)], dtype=np.float64)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _scale(self) -> float:
        scale = self._action_scale
        return float(scale() if callable(scale) else scale)

    def command(self, action: np.ndarray) -> bool:
        """Apply the (scaled) binary gripper command.

        ``action`` is the end-effector slice ``action[6:]`` (length 1). The raw
        value is multiplied by the gripper action scale before being handed to
        ``gripper_fn``, matching the legacy ``action[6] * action_scale[2]`` path.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        pos = float(a[0]) * self._scale()
        return bool(self._gripper_fn(pos))

    def reset(self, target_state: np.ndarray | None = None) -> None:
        # Gripper reset is handled by the env reset flow (e.g. task-level
        # close/open calls); nothing to do here.
        return None

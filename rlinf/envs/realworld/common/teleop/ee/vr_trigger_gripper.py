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

"""Binary gripper teleop driven by the VR controller trigger."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rlinf.envs.realworld.common.teleop.protocols import EEReading, EndEffectorTeleop


class VRTriggerGripperTeleop(EndEffectorTeleop):
    """Edge-detect the VR trigger into a binary gripper command.

    Mirrors :class:`VRTeleopIntervention._gripper_action`: emits ``-1`` on a
    close edge, ``+1`` on an open edge, and ``0`` while held, where the threshold
    decides closed vs open.
    """

    def __init__(self, gripper_threshold: float = 0.6) -> None:
        self._threshold = float(gripper_threshold)
        self._last_closed: bool | None = None

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self) -> None:
        self._last_closed = None

    def _gripper_scalar(self, trigger: float) -> float:
        closed = trigger >= self._threshold
        if self._last_closed is None:
            self._last_closed = closed
            return 0.0
        if closed == self._last_closed:
            return 0.0
        self._last_closed = closed
        return -1.0 if closed else 1.0

    def compute(self, aux: Dict[str, Any]) -> EEReading:
        trigger = float(aux.get("trigger", 0.0))
        scalar = self._gripper_scalar(trigger)
        active = abs(scalar) > 0.5
        return EEReading(command=np.array([scalar], dtype=np.float64), active=active)

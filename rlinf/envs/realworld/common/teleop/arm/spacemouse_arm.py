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

"""SpaceMouse arm teleop source."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert
from rlinf.envs.realworld.common.teleop.protocols import ArmReading, ArmTeleop


class SpaceMouseArmTeleop(ArmTeleop):
    """6-DOF arm delta from a 3Dconnexion SpaceMouse.

    The raw expert axes are used directly as the arm delta (the env applies its
    own ``action_scale`` in ``step``), matching the legacy
    :class:`SpacemouseIntervention` behavior. The device buttons are forwarded
    via ``aux["buttons"]`` so a paired gripper teleop can use them.
    """

    def __init__(
        self,
        device_index: int = 0,
        axis_mapping: list[int] | None = None,
        axis_remap: list[tuple[int, int]] | None = None,
        motion_threshold: float = 0.001,
    ) -> None:
        self.expert = SpaceMouseExpert(
            device_index=device_index,
            axis_mapping=axis_mapping,
            axis_remap=axis_remap,
        )
        self._motion_threshold = float(motion_threshold)
        self._buttons: list = [False, False]

    def read(self, tcp_pose7: np.ndarray, action_scale: np.ndarray) -> ArmReading:
        expert_a, buttons = self.expert.get_action()
        expert_a = np.asarray(expert_a, dtype=np.float64)
        self._buttons = list(buttons)
        delta6 = np.clip(expert_a[:6], -1.0, 1.0)
        active = bool(np.linalg.norm(expert_a) > self._motion_threshold)
        aux: Dict[str, Any] = {"buttons": list(buttons)}
        return ArmReading(delta6=delta6, active=active, aux=aux)

    def info(self) -> Dict[str, Any]:
        left = bool(self._buttons[0]) if len(self._buttons) > 0 else False
        right = bool(self._buttons[1]) if len(self._buttons) > 1 else False
        return {"left": left, "right": right}

    def close(self) -> None:
        close = getattr(self.expert, "close", None)
        if callable(close):
            close()

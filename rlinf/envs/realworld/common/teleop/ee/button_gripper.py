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

"""Binary gripper teleop driven by SpaceMouse buttons."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rlinf.envs.realworld.common.teleop.protocols import EEReading, EndEffectorTeleop
from rlinf.envs.realworld.common.wrappers.spacemouse_intervention import (
    sample_gripper_action,
)


class ButtonGripperTeleop(EndEffectorTeleop):
    """Map ``aux['buttons']`` (left=close, right=open) to a binary gripper scalar.

    Mirrors :class:`SpacemouseIntervention`'s gripper handling: the left button
    samples a strong "close" value, the right a strong "open" value, and the
    last value is held between presses.
    """

    def __init__(self) -> None:
        self._gripper_action = sample_gripper_action(is_open=True)

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self) -> None:
        self._gripper_action = sample_gripper_action(is_open=True)

    def compute(self, aux: Dict[str, Any]) -> EEReading:
        buttons = aux.get("buttons", [False, False])
        left = bool(buttons[0]) if len(buttons) > 0 else False
        right = bool(buttons[1]) if len(buttons) > 1 else False
        active = False
        if left:  # close gripper
            self._gripper_action = sample_gripper_action(is_open=False)
            active = True
        elif right:  # open gripper
            self._gripper_action = sample_gripper_action(is_open=True)
            active = True
        return EEReading(command=np.asarray(self._gripper_action, dtype=np.float64).copy(), active=active)

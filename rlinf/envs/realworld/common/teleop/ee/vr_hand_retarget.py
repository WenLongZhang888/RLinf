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

"""Dexterous-hand (Revo2) teleop: retarget VR trigger/grip to finger targets."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rlinf.envs.realworld.common.hand.revo2_mapping import (
    HAND_MODE_GRIPPER,
    THUMB_OPPOSITION_DEFAULT,
    compute_revo2_targets,
    normalize_hand_mode,
)
from rlinf.envs.realworld.common.teleop.protocols import EEReading, EndEffectorTeleop


class VRHandRetargetTeleop(EndEffectorTeleop):
    """Map VR ``trigger`` / ``grip`` to a normalized Revo2 6-DOF target vector.

    The output order matches
    :data:`rlinf.envs.realworld.common.hand.revo2_mapping.FINGER_NAMES`
    (``[thumb_bend, thumb_opposition, index, middle, ring, pinky]``), which is
    exactly the action vector :class:`Revo2Hand` consumes.
    """

    def __init__(
        self,
        mode: str = HAND_MODE_GRIPPER,
        thumb_opposition: float = THUMB_OPPOSITION_DEFAULT,
        engage_threshold: float = 0.02,
    ) -> None:
        self._mode = normalize_hand_mode(mode)
        self._thumb_opposition = float(thumb_opposition)
        self._engage_threshold = float(engage_threshold)

    @property
    def action_dim(self) -> int:
        return 6

    def compute(self, aux: Dict[str, Any]) -> EEReading:
        trigger = float(aux.get("trigger", 0.0))
        grip = float(aux.get("grip", 0.0))
        targets = compute_revo2_targets(
            trigger=trigger,
            grip=grip,
            thumb_opposition=self._thumb_opposition,
            mode=self._mode,
        )
        command = np.asarray(targets.to_normalized_vector(), dtype=np.float64)
        active = (trigger > self._engage_threshold) or (grip > self._engage_threshold)
        return EEReading(command=command, active=active)

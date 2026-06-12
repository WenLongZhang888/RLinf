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

"""PICO analog input -> BrainCo Revo2 dexterous-hand target mapping.

Revo2 (base version) exposes 6 DoF slots (per the official
``revo2_timing_test_gui.py:FINGER_NAMES``)::

    [Thumb Flex, Thumb Aux, Index, Middle, Ring, Pinky]

- **slot 0 = Thumb Flex**: thumb flexion.
- **slot 1 = Thumb Aux**: thumb opposition / abduction.
- **slots 2..5 = Index / Middle / Ring / Pinky**: four-finger flexion,
  SDK upper bound 1000.

We split these 6 DoF into 6 normalized scalars in ``[0, 1]``.

Default mappings:

- ``gripper``: PICO ``trigger`` [0, 1] -> all five fingers synchronously
  (treat Revo2 as a 1-D gripper).
- ``two-channel``: PICO ``trigger`` drives the four fingers, ``grip``
  drives thumb flexion.
- Thumb opposition (``thumb_opposition`` -> Thumb Aux, slot 1) stays a
  caller-injected constant, default :data:`THUMB_OPPOSITION_DEFAULT` = 0.8,
  suitable for grasping tasks.

Everything is 0.1% precision over the position range ``[0, 1000]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

HAND_MODE_GRIPPER = "gripper"
HAND_MODE_TWO_CHANNEL = "two-channel"
HAND_MODES = (HAND_MODE_GRIPPER, HAND_MODE_TWO_CHANNEL)

# Default thumb opposition (normalized [0, 1]; 1.0 maps to SDK_POS_THUMB_AUX_MAX).
THUMB_OPPOSITION_DEFAULT: float = 0.8

# SDK-side 0.1% precision ranges.
SDK_POS_MAX: int = 1000
SDK_POS_THUMB_FLEX_MAX: int = 700
SDK_POS_THUMB_AUX_MAX: int = 1000

# Canonical normalized action order used as the Revo2Hand action vector.
FINGER_NAMES = (
    "thumb_bend",
    "thumb_opposition",
    "index",
    "middle",
    "ring",
    "pinky",
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


@dataclass
class Revo2FingerTargets:
    """Revo2 6-DoF normalized targets (0 fully open/extended ~ 1 fully closed).

    Fields map one-to-one to SDK slots:

    - ``thumb_bend``       -> Thumb Flex (slot 0, thumb flexion)
    - ``thumb_opposition`` -> Thumb Aux  (slot 1, thumb opposition/abduction)
    - ``index/middle/ring/pinky`` -> slots 2/3/4/5, four-finger flexion
    """

    index: float
    middle: float
    ring: float
    pinky: float
    thumb_bend: float = 0.0
    thumb_opposition: float = THUMB_OPPOSITION_DEFAULT

    def to_normalized_vector(self) -> List[float]:
        """Return the 6-D normalized action vector in :data:`FINGER_NAMES` order."""
        return [
            self.thumb_bend,
            self.thumb_opposition,
            self.index,
            self.middle,
            self.ring,
            self.pinky,
        ]

    @classmethod
    def from_normalized_vector(cls, vec) -> "Revo2FingerTargets":
        """Build targets from a 6-D vector in :data:`FINGER_NAMES` order."""
        v = [float(x) for x in vec]
        if len(v) != 6:
            raise ValueError(f"Revo2 action vector must have length 6, got {len(v)}")
        thumb_bend, thumb_opposition, index, middle, ring, pinky = v
        return cls(
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            thumb_bend=thumb_bend,
            thumb_opposition=thumb_opposition,
        )


def normalize_hand_mode(raw: str) -> str:
    mode = str(raw).strip().lower().replace("_", "-")
    if mode in {"grip", "grasp", "gripper", "one-channel", "one-d", "1d"}:
        return HAND_MODE_GRIPPER
    if mode in {"two-channel", "two", "2d", "dex", "legacy"}:
        return HAND_MODE_TWO_CHANNEL
    raise ValueError(f"Unknown hand mode: {raw}. Options: {', '.join(HAND_MODES)}")


def compute_revo2_targets(
    trigger: float,
    grip: float,
    thumb_opposition: float = THUMB_OPPOSITION_DEFAULT,
    mode: str = HAND_MODE_TWO_CHANNEL,
) -> Revo2FingerTargets:
    """Compute Revo2 targets from PICO analog inputs.

    Mapping:

    - ``gripper``: ``trigger`` -> all five fingers synchronously.
    - ``two-channel``: ``trigger`` -> four fingers, ``grip`` -> thumb flexion.
    - Thumb opposition (``thumb_opposition``) is injected by the caller.

    Args:
        trigger: Controller trigger in [0, 1]; 0 = fully open, 1 = fully closed.
        grip: Controller grip in [0, 1]; used for thumb flexion in two-channel.
        thumb_opposition: Thumb opposition amount in [0, 1].
        mode: ``gripper`` or ``two-channel``.
    """
    t = clamp(trigger, 0.0, 1.0)
    g = clamp(grip, 0.0, 1.0)
    opp = clamp(thumb_opposition, 0.0, 1.0)
    hand_mode = normalize_hand_mode(mode)
    thumb_bend = t if hand_mode == HAND_MODE_GRIPPER else g
    return Revo2FingerTargets(
        index=t,
        middle=t,
        ring=t,
        pinky=t,
        thumb_bend=thumb_bend,
        thumb_opposition=opp,
    )


def to_sdk_positions(targets: Revo2FingerTargets) -> List[int]:
    """Convert normalized targets to the Revo2 SDK 6-D position array.

    Returned order: ``[Thumb Flex, Thumb Aux, Index, Middle, Ring, Pinky]``,
    each a 0.1%-precision integer. Thumb Flex and Thumb Aux are mapped by their
    respective calibrated upper bounds.
    """
    return [
        int(clamp(targets.thumb_bend) * SDK_POS_THUMB_FLEX_MAX),  # slot 0: Thumb Flex
        int(clamp(targets.thumb_opposition) * SDK_POS_THUMB_AUX_MAX),  # slot 1: Thumb Aux
        int(clamp(targets.index) * SDK_POS_MAX),
        int(clamp(targets.middle) * SDK_POS_MAX),
        int(clamp(targets.ring) * SDK_POS_MAX),
        int(clamp(targets.pinky) * SDK_POS_MAX),
    ]


__all__ = [
    "HAND_MODE_GRIPPER",
    "HAND_MODE_TWO_CHANNEL",
    "HAND_MODES",
    "FINGER_NAMES",
    "SDK_POS_MAX",
    "SDK_POS_THUMB_AUX_MAX",
    "SDK_POS_THUMB_FLEX_MAX",
    "THUMB_OPPOSITION_DEFAULT",
    "Revo2FingerTargets",
    "clamp",
    "compute_revo2_targets",
    "normalize_hand_mode",
    "to_sdk_positions",
]

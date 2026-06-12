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

"""Composable teleoperation source protocols.

Teleoperation is split into two orthogonal, composable pieces so the arm and
the end-effector can be driven by independent devices and mixed freely:

- :class:`ArmTeleop` produces a 6-DOF arm delta (already scaled and clipped to
  ``[-1, 1]``) from the current TCP pose, plus an ``aux`` dict carrying the same
  device frame's extra signals (trigger/grip/buttons).
- :class:`EndEffectorTeleop` produces an end-effector command vector of length
  ``action_dim`` from that same ``aux`` dict, so it consumes the *same* device
  read (no second poll, no frame skew).

:class:`~rlinf.envs.realworld.common.wrappers.teleop_intervention.TeleopInterventionWrapper`
composes one of each and centralizes the hold-time / merge logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ArmReading:
    """One arm-teleop tick."""

    delta6: np.ndarray  # 6-D arm delta, scaled and clipped to [-1, 1]
    active: bool  # whether the arm is intervening this tick
    aux: Dict[str, Any] = field(default_factory=dict)  # shared device-frame extras


@dataclass
class EEReading:
    """One end-effector-teleop tick."""

    command: np.ndarray  # end-effector command, length == action_dim
    active: bool  # whether the end-effector is intervening this tick


class ArmTeleop(ABC):
    """Produces a 6-DOF arm delta from the current TCP pose."""

    @abstractmethod
    def read(self, tcp_pose7: np.ndarray, action_scale: np.ndarray) -> ArmReading:
        """Read one device frame and return the arm delta + shared aux."""

    def reset(self) -> None:
        """Reset any per-episode latched state."""

    def info(self) -> Dict[str, Any]:
        """Optional extra ``info`` keys to attach to ``step`` output."""
        return {}

    def close(self) -> None:
        """Release device resources."""


class EndEffectorTeleop(ABC):
    """Produces an end-effector command from the arm teleop's shared aux."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Length of the produced command vector (== EndEffector.action_dim)."""

    @abstractmethod
    def compute(self, aux: Dict[str, Any]) -> EEReading:
        """Compute the end-effector command from the shared device-frame aux."""

    def reset(self) -> None:
        """Reset any per-episode latched state."""

    def info(self) -> Dict[str, Any]:
        """Optional extra ``info`` keys to attach to ``step`` output."""
        return {}

    def close(self) -> None:
        """Release device resources."""


__all__ = [
    "ArmReading",
    "EEReading",
    "ArmTeleop",
    "EndEffectorTeleop",
]

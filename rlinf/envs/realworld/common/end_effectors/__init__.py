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

"""Shared end-effector abstraction and factory."""

from .base import EndEffector, EndEffectorType, normalize_end_effector_type

__all__ = [
    "EndEffector",
    "EndEffectorType",
    "create_end_effector",
    "normalize_end_effector_type",
]


def create_end_effector(
    end_effector_type: str | EndEffectorType,
    **kwargs,
) -> EndEffector:
    """Factory for end-effector instances.

    Supported types: ``"ruiyan_hand"``, ``"revo2_hand"``, ``"marvin_gripper"``.
    Extra keyword arguments are forwarded to the concrete constructor (e.g.
    ``controller=`` for ``marvin_gripper``, hand config for the hands).

    Raises:
        ValueError: If the end-effector type is not recognized.
    """
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type == EndEffectorType.RUIYAN_HAND:
        from rlinf.envs.realworld.common.hand.ruiyan_hand import RuiyanHand

        return RuiyanHand(**kwargs)
    if end_effector_type == EndEffectorType.REVO2_HAND:
        from rlinf.envs.realworld.common.hand.revo2_hand import Revo2Hand

        return Revo2Hand(**kwargs)
    if end_effector_type == EndEffectorType.MARVIN_GRIPPER:
        from rlinf.envs.realworld.marvin.end_effectors.marvin_gripper import (
            MarvinGripper,
        )

        return MarvinGripper(**kwargs)

    raise ValueError(
        f"Unsupported end-effector type: {end_effector_type}. "
        "Supported types: ['ruiyan_hand', 'revo2_hand', 'marvin_gripper']."
    )

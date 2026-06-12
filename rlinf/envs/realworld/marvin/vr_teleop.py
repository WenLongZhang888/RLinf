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

"""Backward-compatible Marvin VR teleoperation exports.

The implementation lives in ``rlinf.envs.realworld.common.vr`` so VR input stays
separate from the Tianji/Marvin SDK adapter.
"""

from rlinf.envs.realworld.common.vr import (
    ArmTeleopCommand,
    ControllerSnapshot,
    PicoArmTeleopController,
    PicoFrameProfile,
    PoseEmaFilter,
    R_ROBOT_FROM_PICO,
    XrClient,
    build_local_workspace_limits,
    clamp_workspace,
    identity_pose7,
    limit_pose_step,
    matrix_to_pose7 as transform_to_pose7,
    matrix_to_pos_rpy as transform_to_pos_rpy,
    pick_controller_inputs,
    pose7_to_matrix as pose7_to_transform,
    pose7_to_robot_pos_rpy,
    xr_pose_to_matrix as xr_pose_to_transform,
)

ArmTeleopController = PicoArmTeleopController

__all__ = [
    "ArmTeleopCommand",
    "ArmTeleopController",
    "ControllerSnapshot",
    "PicoArmTeleopController",
    "PicoFrameProfile",
    "PoseEmaFilter",
    "R_ROBOT_FROM_PICO",
    "XrClient",
    "build_local_workspace_limits",
    "clamp_workspace",
    "identity_pose7",
    "limit_pose_step",
    "pick_controller_inputs",
    "pose7_to_robot_pos_rpy",
    "pose7_to_transform",
    "transform_to_pose7",
    "transform_to_pos_rpy",
    "xr_pose_to_transform",
]

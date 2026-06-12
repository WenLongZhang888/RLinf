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

"""PICO/XR teleoperation helpers for real-world environments."""

from .pico_teleop import (
    ArmTeleopCommand,
    PicoArmTeleopController,
    PicoFrameProfile,
    PoseEmaFilter,
    R_ROBOT_FROM_PICO,
    build_local_workspace_limits,
    clamp_workspace,
    identity_pose7,
    limit_pose_step,
    matrix_to_pose7,
    matrix_to_pos_rpy,
    pick_controller_inputs,
    pose7_to_matrix,
    pose7_to_robot_pos_rpy,
    pose7_to_transform,
    transform_to_pose7,
    transform_to_pos_rpy,
    xr_pose_to_matrix,
    xr_pose_to_transform,
)
from .xr_client import ControllerSnapshot, XrClient, identity_pose7

__all__ = [
    "ArmTeleopCommand",
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
    "matrix_to_pose7",
    "matrix_to_pos_rpy",
    "pick_controller_inputs",
    "pose7_to_matrix",
    "pose7_to_robot_pos_rpy",
    "pose7_to_transform",
    "transform_to_pose7",
    "transform_to_pos_rpy",
    "xr_pose_to_matrix",
    "xr_pose_to_transform",
]

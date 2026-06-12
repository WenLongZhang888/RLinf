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

"""PICO controller pose mapping and clutch-based arm teleoperation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .xr_client import ControllerSnapshot, XrClient

logger = logging.getLogger(__name__)

WORKSPACE_LIMITS = {
    "x": (-1.50, 1.50),
    "y": (-1.50, 1.50),
    "z": (-1.50, 1.50),
}
EMA_ALPHA_TRANSLATION = 0.30
EMA_ALPHA_ROTATION = 0.30

# PICO(OpenXR): +X right, +Y up, +Z backward.
# Marvin measured base frame: +X front, +Y up, +Z operator-left.
# Therefore: X_p -> -Z_r, Y_p -> +Y_r, Z_p -> -X_r.
R_ROBOT_FROM_PICO = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

LEFT_ARM_CONTROL_BASIS_FROM_LEGACY = np.diag([1.0, -1.0, 1.0])
RIGHT_ARM_CONTROL_BASIS_FROM_LEGACY = np.eye(3, dtype=np.float64)


def identity_pose7() -> np.ndarray:
    """Return identity pose ``[x, y, z, qx, qy, qz, qw]``."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class PicoFrameProfile:
    """PICO pose profile for a controller/arm side."""

    side: str
    position_basis: np.ndarray
    rotation_basis: np.ndarray
    description: str


PICO_FRAME_PROFILES = {
    "right": PicoFrameProfile(
        side="right",
        position_basis=RIGHT_ARM_CONTROL_BASIS_FROM_LEGACY,
        rotation_basis=RIGHT_ARM_CONTROL_BASIS_FROM_LEGACY,
        description="right-controller/right-arm frame",
    ),
    "left": PicoFrameProfile(
        side="left",
        position_basis=LEFT_ARM_CONTROL_BASIS_FROM_LEGACY,
        rotation_basis=LEFT_ARM_CONTROL_BASIS_FROM_LEGACY,
        description="left-controller/A-arm frame",
    ),
}


def _frame_profile(side: str | PicoFrameProfile) -> PicoFrameProfile:
    if isinstance(side, PicoFrameProfile):
        return side
    try:
        return PICO_FRAME_PROFILES[side]
    except KeyError as exc:
        raise ValueError("side must be 'left' or 'right'.") from exc


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` to a homogeneous matrix."""
    pose = np.asarray(pose7, dtype=np.float64).reshape(7)
    quat = pose[3:].copy()
    quat_norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat_norm) or quat_norm < 1e-6:
        quat = identity_pose7()[3:]
    else:
        quat /= quat_norm
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def pose7_to_transform(pose7: np.ndarray) -> np.ndarray:
    """Compatibility alias for :func:`pose7_to_matrix`."""
    return pose7_to_matrix(pose7)


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert a homogeneous matrix to ``[x, y, z, qx, qy, qz, qw]``."""
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()])


def transform_to_pose7(T: np.ndarray) -> np.ndarray:
    """Compatibility alias for :func:`matrix_to_pose7`."""
    return matrix_to_pose7(T)


def _apply_frame_profile(T: np.ndarray, profile: PicoFrameProfile) -> np.ndarray:
    T_out = T.copy()
    T_out[:3, 3] = profile.position_basis @ T[:3, 3]
    T_out[:3, :3] = profile.rotation_basis @ T[:3, :3] @ profile.rotation_basis.T
    return T_out


def _extract_yaw_rotation(R_mat: np.ndarray) -> np.ndarray:
    """Extract yaw around the robot +Y axis for Marvin's Y-up base frame."""
    forward_r = R_mat[:, 0]
    yaw = float(np.arctan2(-forward_r[2], forward_r[0]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )


def xr_pose_to_matrix(
    headset_pose7: np.ndarray,
    controller_pose7: np.ndarray,
    *,
    side: str | PicoFrameProfile = "right",
) -> np.ndarray:
    """Map raw PICO headset/controller poses into a robot control frame."""
    profile = _frame_profile(side)
    T_head_p = pose7_to_matrix(headset_pose7)
    T_ctrl_p = pose7_to_matrix(controller_pose7)

    def transform_basis(T_p: np.ndarray) -> np.ndarray:
        T_r = np.eye(4, dtype=np.float64)
        T_r[:3, :3] = R_ROBOT_FROM_PICO @ T_p[:3, :3] @ R_ROBOT_FROM_PICO.T
        T_r[:3, 3] = R_ROBOT_FROM_PICO @ T_p[:3, 3]
        return T_r

    T_head_r = transform_basis(T_head_p)
    T_ctrl_r = transform_basis(T_ctrl_p)

    p_rel = T_ctrl_r[:3, 3] - T_head_r[:3, 3]
    R_inv_yaw = _extract_yaw_rotation(T_head_r[:3, :3]).T

    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = R_inv_yaw @ T_ctrl_r[:3, :3]
    T_out[:3, 3] = R_inv_yaw @ p_rel
    return _apply_frame_profile(T_out, profile)


def xr_pose_to_transform(
    headset_pose7: np.ndarray,
    controller_pose7: np.ndarray,
    *,
    side: str | PicoFrameProfile = "right",
) -> np.ndarray:
    """Compatibility alias for :func:`xr_pose_to_matrix`."""
    return xr_pose_to_matrix(headset_pose7, controller_pose7, side=side)


def pose7_to_robot_pos_rpy(
    pose7: np.ndarray,
    *,
    side: str | PicoFrameProfile = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagnostic ``(xyz_m, rpy_deg)`` in the chosen control frame."""
    profile = _frame_profile(side)
    T_p = pose7_to_matrix(pose7)
    T_r = np.eye(4, dtype=np.float64)
    T_r[:3, :3] = R_ROBOT_FROM_PICO @ T_p[:3, :3] @ R_ROBOT_FROM_PICO.T
    T_r[:3, 3] = R_ROBOT_FROM_PICO @ T_p[:3, 3]
    T_ctrl = _apply_frame_profile(T_r, profile)
    return T_ctrl[:3, 3].copy(), R.from_matrix(T_ctrl[:3, :3]).as_euler(
        "xyz", degrees=True
    )


def matrix_to_pos_rpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz_m, rpy_deg)`` from a 4x4 transform."""
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)


def transform_to_pos_rpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for :func:`matrix_to_pos_rpy`."""
    return matrix_to_pos_rpy(T)


def build_local_workspace_limits(
    center_pose7_or_T: np.ndarray,
    *,
    xy_range: float = 0.15,
    z_range_low: float = 0.10,
    z_range_high: float = 0.10,
    global_limits: dict[str, tuple[float, float]] = WORKSPACE_LIMITS,
) -> dict[str, tuple[float, float]]:
    """Build a local workspace box around a current TCP pose."""
    arr = np.asarray(center_pose7_or_T, dtype=np.float64)
    if arr.shape == (4, 4):
        xyz = arr[:3, 3]
    elif arr.size == 7:
        xyz = arr.reshape(7)[:3]
    else:
        raise ValueError("center must be a 4x4 transform or a 7D pose.")

    ranges = {
        "x": (float(xyz[0] - xy_range), float(xyz[0] + xy_range)),
        "y": (float(xyz[1] - xy_range), float(xyz[1] + xy_range)),
        "z": (float(xyz[2] - z_range_low), float(xyz[2] + z_range_high)),
    }
    return {
        axis: (max(global_limits[axis][0], lo), min(global_limits[axis][1], hi))
        for axis, (lo, hi) in ranges.items()
    }


def clamp_workspace(
    T: np.ndarray,
    limits: dict[str, tuple[float, float]] = WORKSPACE_LIMITS,
) -> np.ndarray:
    """Clamp pose translation inside an axis-aligned workspace box."""
    T_out = T.copy()
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = limits[axis]
        T_out[i, 3] = float(np.clip(T_out[i, 3], lo, hi))
    return T_out


class PoseEmaFilter:
    """EMA filter for matrix poses, using Slerp for orientation."""

    def __init__(
        self,
        alpha_trans: float = EMA_ALPHA_TRANSLATION,
        alpha_rot: float = EMA_ALPHA_ROTATION,
    ) -> None:
        self.alpha_t = float(np.clip(alpha_trans, 0.0, 1.0))
        self.alpha_r = float(np.clip(alpha_rot, 0.0, 1.0))
        self._p: np.ndarray | None = None
        self._q: np.ndarray | None = None

    def reset(self, T_init: np.ndarray | None = None) -> None:
        if T_init is None:
            self._p = None
            self._q = None
            return
        self._p = T_init[:3, 3].copy()
        self._q = R.from_matrix(T_init[:3, :3]).as_quat()

    def apply(self, T_new: np.ndarray) -> np.ndarray:
        p_new = T_new[:3, 3]
        q_new = R.from_matrix(T_new[:3, :3]).as_quat()

        if self._p is None or self._q is None:
            self._p = p_new.copy()
            self._q = q_new.copy()
            return T_new.copy()

        self._p = self.alpha_t * p_new + (1.0 - self.alpha_t) * self._p
        key_rots = R.from_quat(np.stack([self._q, q_new], axis=0))
        q_interp = Slerp([0.0, 1.0], key_rots)([self.alpha_r]).as_quat()[0]
        self._q = q_interp

        T_out = np.eye(4, dtype=np.float64)
        T_out[:3, :3] = R.from_quat(self._q).as_matrix()
        T_out[:3, 3] = self._p
        return T_out


@dataclass
class ArmTeleopCommand:
    """One arm-only teleoperation command."""

    active: bool
    T_target: np.ndarray
    trigger: float = 0.0
    grip: float = 0.0
    vr_delta_m: np.ndarray | None = None


@dataclass
class _ClutchState:
    active: bool = False
    last_button: bool = False
    T_vr_init: np.ndarray | None = None
    T_ee_init: np.ndarray | None = None


def pick_controller_inputs(
    snap: ControllerSnapshot, side: str
) -> tuple[np.ndarray, float, float, bool]:
    """Return pose, trigger, grip, and clutch button state for one side."""
    if side == "right":
        return (
            snap.right_pose,
            float(snap.right_trigger),
            float(snap.right_grip),
            bool(snap.button_a or snap.button_menu),
        )
    if side == "left":
        return (
            snap.left_pose,
            float(snap.left_trigger),
            float(snap.left_grip),
            bool(snap.button_x or snap.left_button_menu),
        )
    raise ValueError("side must be 'left' or 'right'.")


class PicoArmTeleopController:
    """PICO controller to target TCP pose state machine.

    The clutch button toggles relative teleoperation. When enabled, current
    PICO pose and current end-effector pose are latched; subsequent commands
    apply PICO deltas to that latched end-effector pose.
    """

    def __init__(
        self,
        xr: XrClient,
        *,
        side: str = "left",
        workspace_limits: dict[str, tuple[float, float]] = WORKSPACE_LIMITS,
        ema_trans: float = EMA_ALPHA_TRANSLATION,
        ema_rot: float = EMA_ALPHA_ROTATION,
        translation_scale: float = 1.0,
        xyz_scale: np.ndarray | list[float] | None = None,
        track_rotation: bool = False,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        self._xr = xr
        self._side = side
        self._workspace_limits = workspace_limits
        self._filter = PoseEmaFilter(alpha_trans=ema_trans, alpha_rot=ema_rot)
        self._clutch = _ClutchState()
        self._translation_scale = float(translation_scale)
        self._xyz_scale = (
            np.ones(3, dtype=np.float64)
            if xyz_scale is None
            else np.asarray(xyz_scale, dtype=np.float64).reshape(3)
        )
        self._track_rotation = bool(track_rotation)
        self._last_snap: ControllerSnapshot | None = None

    @property
    def last_snap(self) -> ControllerSnapshot | None:
        return self._last_snap

    @property
    def side(self) -> str:
        return self._side

    @property
    def active(self) -> bool:
        return self._clutch.active

    def is_active(self) -> bool:
        """Return whether clutch control is currently active."""
        return self._clutch.active

    def reset_clutch(self) -> None:
        """Deactivate clutch control and clear latched references."""
        self._clutch = _ClutchState()
        self._filter.reset()

    def set_workspace_limits(
        self,
        workspace_limits: dict[str, tuple[float, float]],
    ) -> None:
        """Replace the workspace limits used for clamping commands."""
        self._workspace_limits = workspace_limits

    def step(self, T_ee_now: np.ndarray) -> ArmTeleopCommand:
        """Read one PICO frame and return a target pose command."""
        try:
            return self._step_inner(T_ee_now)
        except Exception as exc:
            logger.warning("PICO teleop frame degraded: %s", exc)
            return ArmTeleopCommand(
                active=False,
                T_target=T_ee_now.copy(),
                vr_delta_m=np.zeros(3, dtype=np.float64),
            )

    def _step_inner(self, T_ee_now: np.ndarray) -> ArmTeleopCommand:
        snap = self._xr.snapshot()
        self._last_snap = snap

        ctrl_pose, trigger, grip, button_pressed = pick_controller_inputs(
            snap, self._side
        )
        T_vr_now = xr_pose_to_matrix(snap.headset_pose, ctrl_pose, side=self._side)
        self._update_clutch(button_pressed, T_vr_now, T_ee_now)

        vr_delta = np.zeros(3, dtype=np.float64)
        if (
            self._clutch.active
            and self._clutch.T_vr_init is not None
            and self._clutch.T_ee_init is not None
        ):
            T_cmd_raw = self._clutch.T_ee_init.copy()
            vr_delta = T_vr_now[:3, 3] - self._clutch.T_vr_init[:3, 3]
            cmd_delta = vr_delta * self._xyz_scale * self._translation_scale
            T_cmd_raw[:3, 3] = self._clutch.T_ee_init[:3, 3] + cmd_delta

            if self._track_rotation:
                R_delta = T_vr_now[:3, :3] @ self._clutch.T_vr_init[:3, :3].T
                T_cmd_raw[:3, :3] = R_delta @ self._clutch.T_ee_init[:3, :3]

            T_cmd = clamp_workspace(
                self._filter.apply(T_cmd_raw),
                limits=self._workspace_limits,
            )
        else:
            T_cmd = T_ee_now.copy()

        return ArmTeleopCommand(
            active=self._clutch.active,
            T_target=T_cmd,
            trigger=trigger,
            grip=grip,
            vr_delta_m=vr_delta,
        )

    def _update_clutch(
        self,
        button_pressed: bool,
        T_vr_now: np.ndarray,
        T_ee_now: np.ndarray,
    ) -> None:
        rising_edge = button_pressed and not self._clutch.last_button
        self._clutch.last_button = button_pressed
        if not rising_edge:
            return

        self._clutch.active = not self._clutch.active
        if self._clutch.active:
            self._clutch.T_vr_init = T_vr_now.copy()
            self._clutch.T_ee_init = T_ee_now.copy()
            self._filter.reset(T_ee_now)
        else:
            self._clutch.T_vr_init = None
            self._clutch.T_ee_init = None
            self._filter.reset()


def limit_pose_step(
    T_target: np.ndarray,
    T_last: np.ndarray,
    *,
    max_step_m: float,
    max_rot_deg: float,
) -> np.ndarray:
    """Limit per-step pose jumps before converting to an env action."""
    T_out = T_target.copy()
    delta_p = T_target[:3, 3] - T_last[:3, 3]
    dist = float(np.linalg.norm(delta_p))
    if max_step_m > 0.0 and dist > max_step_m:
        T_out[:3, 3] = T_last[:3, 3] + delta_p * (max_step_m / dist)

    if max_rot_deg <= 0.0:
        return T_out

    r_last = R.from_matrix(T_last[:3, :3])
    r_target = R.from_matrix(T_target[:3, :3])
    r_delta = r_last.inv() * r_target
    rotvec = r_delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    max_angle = float(np.deg2rad(max_rot_deg))
    if angle > max_angle and angle > 1e-9:
        r_limited = r_last * R.from_rotvec(rotvec * (max_angle / angle))
        T_out[:3, :3] = r_limited.as_matrix()
    return T_out

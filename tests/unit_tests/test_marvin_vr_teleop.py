"""Unit tests for Marvin PICO VR teleoperation math."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_vr_teleop_module():
    package_name = "marvin_vr_test_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(REPO_ROOT / "rlinf" / "envs" / "realworld" / "common" / "vr")
    ]
    sys.modules[package_name] = package

    _load_module(
        f"{package_name}.xr_client",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "common" / "vr" / "xr_client.py",
    )
    return _load_module(
        f"{package_name}.pico_teleop",
        REPO_ROOT
        / "rlinf"
        / "envs"
        / "realworld"
        / "common"
        / "vr"
        / "pico_teleop.py",
    )


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


vr_teleop = _load_vr_teleop_module()
xr_client = sys.modules["marvin_vr_test_pkg.xr_client"]
ArmTeleopController = vr_teleop.PicoArmTeleopController
ControllerSnapshot = xr_client.ControllerSnapshot
R_ROBOT_FROM_PICO = vr_teleop.R_ROBOT_FROM_PICO
identity_pose7 = xr_client.identity_pose7
limit_pose_step = vr_teleop.limit_pose_step
pose7_to_robot_pos_rpy = vr_teleop.pose7_to_robot_pos_rpy
pose7_to_transform = vr_teleop.pose7_to_transform
transform_to_pose7 = vr_teleop.transform_to_pose7
xr_pose_to_transform = vr_teleop.xr_pose_to_transform


class FakeXr:
    def __init__(self, snapshots: list[ControllerSnapshot]):
        self._snapshots = list(snapshots)
        self._last = self._snapshots[-1]

    def snapshot(self) -> ControllerSnapshot:
        if self._snapshots:
            self._last = self._snapshots.pop(0)
        return self._last


def _pose(xyz, quat=None):
    out = identity_pose7()
    out[:3] = np.asarray(xyz, dtype=np.float64)
    if quat is not None:
        out[3:] = np.asarray(quat, dtype=np.float64)
    return out


def _snap(
    *,
    headset=None,
    left=None,
    right=None,
    left_button=False,
    right_button=False,
):
    return ControllerSnapshot(
        headset_pose=identity_pose7() if headset is None else headset,
        left_pose=identity_pose7() if left is None else left,
        right_pose=identity_pose7() if right is None else right,
        right_trigger=0.0,
        right_grip=0.0,
        button_a=right_button,
        button_menu=False,
        button_x=left_button,
        left_button_menu=False,
        timestamp=0.0,
    )


def test_zero_quaternion_falls_back_to_identity_rotation():
    xyz, rpy = pose7_to_robot_pos_rpy(np.zeros(7), side="right")

    np.testing.assert_allclose(xyz, np.zeros(3))
    np.testing.assert_allclose(rpy, np.zeros(3), atol=1e-8)


def test_pico_basis_maps_forward_up_right_to_marvin_axes():
    T_forward = xr_pose_to_transform(
        identity_pose7(),
        _pose([0.0, 0.0, -1.0]),
        side="right",
    )
    T_up = xr_pose_to_transform(
        identity_pose7(),
        _pose([0.0, 1.0, 0.0]),
        side="right",
    )
    T_right = xr_pose_to_transform(
        identity_pose7(),
        _pose([1.0, 0.0, 0.0]),
        side="right",
    )

    np.testing.assert_allclose(T_forward[:3, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(T_up[:3, 3], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(T_right[:3, 3], [0.0, 0.0, -1.0])
    assert np.isclose(np.linalg.det(R_ROBOT_FROM_PICO), -1.0)


def test_left_frame_mirrors_marvin_y_axis_from_legacy_frame():
    headset = identity_pose7()
    controller = _pose([0.0, 1.0, 0.0])

    right_T = xr_pose_to_transform(headset, controller, side="right")
    left_T = xr_pose_to_transform(headset, controller, side="left")

    np.testing.assert_allclose(right_T[:3, 3], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(left_T[:3, 3], [0.0, -1.0, 0.0])


def test_arm_teleop_uses_world_frame_translation_delta_with_clutch():
    T_ee = np.eye(4, dtype=np.float64)
    T_ee[:3, 3] = [0.50, 0.10, 0.20]
    headset = identity_pose7()
    left_start = _pose([0.0, 0.0, 0.0])
    left_now = _pose([0.0, 0.0, -0.10])

    xr = FakeXr(
        [
            _snap(headset=headset, left=left_start, left_button=True),
            _snap(headset=headset, left=left_now, left_button=False),
        ]
    )
    teleop = ArmTeleopController(
        xr,
        side="left",
        ema_trans=1.0,
        ema_rot=1.0,
        workspace_limits={"x": (-10, 10), "y": (-10, 10), "z": (-10, 10)},
    )

    first = teleop.step(T_ee)
    second = teleop.step(T_ee)

    assert first.active
    assert second.active
    np.testing.assert_allclose(second.vr_delta_m, [0.10, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(second.T_target[:3, 3], [0.60, 0.10, 0.20])


def test_arm_teleop_tracks_relative_rotation_when_enabled():
    T_ee = np.eye(4, dtype=np.float64)
    start_quat = R.from_euler("z", 0.0).as_quat()
    now_quat = R.from_euler("z", 30.0, degrees=True).as_quat()

    xr = FakeXr(
        [
            _snap(left=_pose([0.0, 0.0, 0.0], start_quat), left_button=True),
            _snap(left=_pose([0.0, 0.0, 0.0], now_quat), left_button=False),
        ]
    )
    teleop = ArmTeleopController(
        xr,
        side="left",
        ema_trans=1.0,
        ema_rot=1.0,
        track_rotation=True,
        workspace_limits={"x": (-10, 10), "y": (-10, 10), "z": (-10, 10)},
    )

    teleop.step(T_ee)
    command = teleop.step(T_ee)
    delta = R.from_matrix(command.T_target[:3, :3]).magnitude()

    assert np.isclose(delta, np.deg2rad(30.0), atol=1e-8)


def test_limit_pose_step_caps_translation_and_rotation():
    last = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [1.0, 0.0, 0.0]
    target[:3, :3] = R.from_euler("z", 90.0, degrees=True).as_matrix()

    limited = limit_pose_step(
        target,
        last,
        max_step_m=0.1,
        max_rot_deg=10.0,
    )

    np.testing.assert_allclose(limited[:3, 3], [0.1, 0.0, 0.0])
    assert np.isclose(R.from_matrix(limited[:3, :3]).magnitude(), np.deg2rad(10.0))


def test_transform_pose_roundtrip():
    pose = _pose(
        [0.1, -0.2, 0.3],
        R.from_euler("xyz", [10.0, 20.0, -30.0], degrees=True).as_quat(),
    )
    roundtrip = transform_to_pose7(pose7_to_transform(pose))

    np.testing.assert_allclose(roundtrip[:3], pose[:3])
    np.testing.assert_allclose(roundtrip[3:], pose[3:])

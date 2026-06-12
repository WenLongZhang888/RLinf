"""Unit tests for Marvin impedance mode handling."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_marvin_modules():
    _ensure_package("rlinf", REPO_ROOT / "rlinf")
    _ensure_package("rlinf.envs", REPO_ROOT / "rlinf" / "envs")
    _ensure_package("rlinf.envs.realworld", REPO_ROOT / "rlinf" / "envs" / "realworld")
    _ensure_package(
        "rlinf.envs.realworld.marvin",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin",
    )
    _ensure_package("rlinf.utils", REPO_ROOT / "rlinf" / "utils")

    logging_module = types.ModuleType("rlinf.utils.logging")
    logger = logging.getLogger("marvin-test")
    logging_module.get_logger = lambda: logger
    sys.modules["rlinf.utils.logging"] = logging_module

    scheduler_module = types.ModuleType("rlinf.scheduler")

    class DummyWorker:
        def __init__(self):
            pass

        @classmethod
        def create_group(cls, *args, **kwargs):
            raise NotImplementedError

        def log_debug(self, *args, **kwargs):
            pass

    class DummyCluster:
        pass

    class DummyNodePlacementStrategy:
        def __init__(self, node_ranks):
            self.node_ranks = node_ranks

    scheduler_module.Cluster = DummyCluster
    scheduler_module.NodePlacementStrategy = DummyNodePlacementStrategy
    DummyWorker.logger = logger
    scheduler_module.Worker = DummyWorker
    sys.modules["rlinf.scheduler"] = scheduler_module

    _load_module(
        "rlinf.envs.realworld.marvin.marvin_robot_state",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin" / "marvin_robot_state.py",
    )
    marvin_sdk = _load_module(
        "rlinf.envs.realworld.marvin.marvin_sdk",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin" / "marvin_sdk.py",
    )
    marvin_controller = _load_module(
        "rlinf.envs.realworld.marvin.marvin_controller",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin" / "marvin_controller.py",
    )
    return marvin_sdk, marvin_controller


class MarvinImpedanceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marvin_sdk, cls.marvin_controller = _load_marvin_modules()

    def _make_marvin(self):
        marvin = object.__new__(self.marvin_sdk.Marvin)
        marvin._logger = logging.getLogger("marvin-test")
        marvin._robot = mock.Mock()
        marvin._dcss = mock.Mock()
        marvin._profile_by_arm = {
            "B": {
                "vel_ratio": 100,
                "acc_ratio": 100,
                "cart_k": self.marvin_sdk.DEFAULT_CART_K.copy(),
                "cart_d": self.marvin_sdk.DEFAULT_CART_D.copy(),
                "joint_k": self.marvin_sdk.DEFAULT_JOINT_K.copy(),
                "joint_d": self.marvin_sdk.DEFAULT_JOINT_D.copy(),
            }
        }
        marvin._mode_by_arm = {"B": self.marvin_sdk.CARTESIAN_IMPEDANCE_CONTROL_MODE}
        return marvin

    @staticmethod
    def _subscribe_payload(cur_state):
        return {
            "states": [
                {"cur_state": 0, "cmd_state": 0, "err_code": 0},
                {"cur_state": cur_state, "cmd_state": -1, "err_code": 0},
            ],
            "outputs": [
                {"frame_serial": 1},
                {"frame_serial": 1},
            ],
        }

    def _make_controller(self):
        controller = object.__new__(self.marvin_controller.MarvinController)
        controller._logger = logging.getLogger("marvin-test")
        controller._arm_id = "B"
        controller._client = mock.Mock()
        controller._state = mock.Mock(gripper_open=False)
        controller._active_control_mode = (
            self.marvin_controller.CARTESIAN_IMPEDANCE_CONTROL_MODE
        )
        controller._wait_robot = mock.Mock()
        controller._wait_for_joint = mock.Mock()
        controller.log_debug = mock.Mock()
        return controller

    def test_joint_impedance_mode_uses_joint_sdk_api(self):
        marvin = self._make_marvin()
        marvin._robot.set_imp_joint_state.return_value = True

        with mock.patch.object(self.marvin_sdk.time, "sleep"):
            marvin.set_joint_impedance_mode(
                "B",
                vel_ratio=60,
                acc_ratio=70,
                joint_k=[3.0] * 7,
                joint_d=[0.4] * 7,
            )

        marvin._robot.set_imp_joint_state.assert_called_once_with(
            arm="B",
            velRatio=60,
            AccRatio=70,
            K=[3.0] * 7,
            D=[0.4] * 7,
        )
        self.assertEqual(
            marvin._mode_by_arm["B"],
            self.marvin_sdk.JOINT_IMPEDANCE_CONTROL_MODE,
        )
        self.assertEqual(marvin._profile_by_arm["B"]["joint_k"], [3.0] * 7)
        self.assertEqual(marvin._profile_by_arm["B"]["joint_d"], [0.4] * 7)
        self.assertEqual(
            marvin._profile_by_arm["B"]["cart_k"],
            self.marvin_sdk.DEFAULT_CART_K,
        )

    def test_legacy_sdk_cartesian_impedance_uses_torque_state_and_kd(self):
        marvin = self._make_marvin()
        marvin._robot = mock.Mock(spec=[
            "clear_set",
            "send_cmd",
            "set_vel_acc",
            "set_state",
            "set_impedance_type",
            "set_cart_kd_params",
            "subscribe",
        ])
        marvin._robot.clear_set.return_value = True
        marvin._robot.send_cmd.return_value = True
        marvin._robot.set_vel_acc.return_value = True
        marvin._robot.set_state.return_value = True
        marvin._robot.set_impedance_type.return_value = True
        marvin._robot.set_cart_kd_params.return_value = True
        marvin._robot.subscribe.side_effect = [
            self._subscribe_payload(self.marvin_sdk.ARM_STATE_POSITION),
            self._subscribe_payload(self.marvin_sdk.ARM_STATE_TORQ),
        ]

        with mock.patch.object(self.marvin_sdk.time, "sleep"):
            marvin.set_cartesian_impedance_mode("B")

        self.assertEqual(marvin._robot.clear_set.call_count, 3)
        self.assertEqual(marvin._robot.send_cmd.call_count, 3)
        marvin._robot.set_vel_acc.assert_called_once_with(
            arm="B",
            velRatio=100,
            AccRatio=100,
        )
        marvin._robot.set_state.assert_called_once_with(
            arm="B",
            state=self.marvin_sdk.ARM_STATE_TORQ,
        )
        marvin._robot.set_impedance_type.assert_called_once_with(
            arm="B",
            type=self.marvin_sdk.IMPEDANCE_TYPE_CARTESIAN,
        )
        marvin._robot.set_cart_kd_params.assert_called_once_with(
            arm="B",
            K=self.marvin_sdk.DEFAULT_CART_K,
            D=self.marvin_sdk.DEFAULT_CART_D,
            type=self.marvin_sdk.IMPEDANCE_TYPE_CARTESIAN,
        )

    def test_legacy_joint_command_uses_clear_set_and_send(self):
        marvin = self._make_marvin()
        marvin._robot = mock.Mock(spec=[
            "clear_set",
            "set_joint_cmd_pose",
            "send_cmd",
        ])
        marvin._robot.clear_set.return_value = True
        marvin._robot.set_joint_cmd_pose.return_value = True
        marvin._robot.send_cmd.return_value = True

        ok = marvin._send_joint_command("B", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        self.assertTrue(ok)
        marvin._robot.clear_set.assert_called_once_with()
        marvin._robot.set_joint_cmd_pose.assert_called_once_with(
            arm="B",
            joints=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        marvin._robot.send_cmd.assert_called_once_with()

    def test_update_profile_maps_service_style_stiffness_to_joint_impedance(self):
        marvin = self._make_marvin()
        marvin._mode_by_arm["B"] = self.marvin_sdk.JOINT_IMPEDANCE_CONTROL_MODE
        marvin.set_joint_impedance_mode = mock.Mock()

        marvin.update_profile(
            "B",
            {
                "impedance_type": 1,
                "stiffness": [4.0] * 7,
                "damping": [0.5] * 7,
            },
        )

        self.assertEqual(marvin._profile_by_arm["B"]["joint_k"], [4.0] * 7)
        self.assertEqual(marvin._profile_by_arm["B"]["joint_d"], [0.5] * 7)
        self.assertEqual(
            marvin._profile_by_arm["B"]["cart_k"],
            self.marvin_sdk.DEFAULT_CART_K,
        )
        marvin.set_joint_impedance_mode.assert_called_once_with("B")

    def test_update_profile_can_switch_from_cartesian_to_joint_impedance(self):
        marvin = self._make_marvin()
        marvin.set_joint_impedance_mode = mock.Mock()
        marvin.set_cartesian_impedance_mode = mock.Mock()

        marvin.update_profile("B", {"impedance_type": 1})

        marvin.set_joint_impedance_mode.assert_called_once_with("B")
        marvin.set_cartesian_impedance_mode.assert_not_called()

    def test_controller_keeps_joint_impedance_for_motion_and_reset(self):
        controller = self._make_controller()
        controller._client.set_joint_impedance_mode.return_value = None
        controller._client.move_pose.return_value = np.zeros(7)

        controller.start_controller("joint_impedance")
        self.assertEqual(
            controller._active_control_mode,
            self.marvin_controller.JOINT_IMPEDANCE_CONTROL_MODE,
        )
        controller._client.set_joint_impedance_mode.assert_called_once_with("B")

        target_pose = np.zeros(7, dtype=np.float64)
        controller.move_arm(target_pose)
        controller._client.move_pose.assert_called_once()
        _, move_kwargs = controller._client.move_pose.call_args
        self.assertEqual(
            move_kwargs["control_mode"],
            self.marvin_controller.JOINT_IMPEDANCE_CONTROL_MODE,
        )

        controller.reset_joint([0.0] * 7)
        controller._client.move_joint_positions.assert_called_once()
        _, joint_kwargs = controller._client.move_joint_positions.call_args
        self.assertEqual(
            joint_kwargs["control_mode"],
            self.marvin_controller.POSITION_CONTROL_MODE,
        )
        self.assertEqual(controller._client.set_joint_impedance_mode.call_count, 2)


if __name__ == "__main__":
    unittest.main()

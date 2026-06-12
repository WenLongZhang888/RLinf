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

"""Direct helpers for the official Marvin SDK."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.logging import get_logger

from .marvin_robot_state import MarvinRobotState

ARM_INDEX = {"A": 0, "B": 1}
ARM_ORDER = ("A", "B")
DEFAULT_KINE_CONFIG_NAMES = (
    "ccs.MvKDCfg",
    "ccs_m6_31.MvKDCfg",
    "ccs_m6_40.MvKDCfg",
    "ccs_m3.MvKDCfg",
    "srs.MvKDCfg",
)
POSITION_CONTROL_MODE = "position"
CARTESIAN_IMPEDANCE_CONTROL_MODE = "impedance_cartesian"
JOINT_IMPEDANCE_CONTROL_MODE = "impedance_joint"
# DEFAULT_CART_K = [800.0, 800.0, 800.0, 120.0, 120.0, 120.0, 20.0]
# DEFAULT_CART_D = [0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.3]
DEFAULT_CART_K = [2000.0, 2000.0, 2000.0, 40.0, 40.0, 40.0, 20.0]
DEFAULT_CART_D = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0]
DEFAULT_JOINT_K = [2.0, 2.0, 2.0, 1.6, 1.0, 1.0, 1.0]
DEFAULT_JOINT_D = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
DEFAULT_VEL_RATIO = 10
DEFAULT_ACC_RATIO = 10
ARM_STATE_POSITION = 1
ARM_STATE_TORQ = 3
ARM_STATE_ERROR = 100
ARM_STATE_TRANS_TO_POSITION = 101
ARM_STATE_TRANS_TO_TORQ = 103
IMPEDANCE_TYPE_JOINT = 1
IMPEDANCE_TYPE_CARTESIAN = 2


def _arm_env_value(prefix: str, arm: str) -> str | None:
    return os.getenv(f"{prefix}_{arm}") or os.getenv(prefix)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_hex_bytes(raw: str | None) -> bytes | None:
    if not raw:
        return None
    normalized = raw.replace("0x", " ").replace(",", " ")
    tokens = normalized.split()
    if not tokens:
        return None
    try:
        return bytes.fromhex(" ".join(tokens))
    except ValueError as exc:
        raise ValueError(f"Invalid hex payload: {raw}") from exc


def resolve_marvin_sdk_root() -> Path:
    """Resolve the installed Marvin SDK root."""
    candidates: list[Path] = []
    env_root = os.getenv("MARVIN_SDK_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    repo_local_root = Path(__file__).resolve().parents[4] / "third_party" / "marvin_sdk"
    candidates.append(repo_local_root)
    candidates.append(Path(sys.prefix) / "opt" / "TJ_FX_ROBOT_CONTRL_SDK")
    candidates.append(Path(sys.prefix) / "marvin_sdk")

    for module_name in ("fx_robot", "SDK_PYTHON.fx_robot"):
        try:
            spec = importlib_util.find_spec(module_name)
        except (ModuleNotFoundError, ImportError):
            spec = None
        if spec and spec.origin:
            module_path = Path(spec.origin).resolve()
            if module_path.parent.name == "SDK_PYTHON":
                candidates.append(module_path.parent.parent)

    for candidate in candidates:
        sdk_python = candidate / "SDK_PYTHON" / "fx_robot.py"
        if sdk_python.exists():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Marvin SDK not found in the current environment. Checked: "
        f"{checked}. Install it via requirements/install.sh or set "
        "MARVIN_SDK_PATH to the installed SDK root."
    )


def resolve_marvin_kine_config_path(sdk_root: Path) -> Path:
    """Resolve the Marvin kinematics config file."""
    env_cfg = os.getenv("MARVIN_KINE_CONFIG")
    if env_cfg:
        candidate = Path(env_cfg).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"MARVIN_KINE_CONFIG points to a missing file: {candidate}"
        )

    demo_dir = sdk_root / "DEMO_PYTHON"
    for filename in DEFAULT_KINE_CONFIG_NAMES:
        candidate = demo_dir / filename
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No TJ kinematics config found under "
        f"{demo_dir}. Set MARVIN_KINE_CONFIG explicitly."
    )


def _ensure_sdk_python_path(sdk_root: Path) -> None:
    sdk_python = str((sdk_root / "SDK_PYTHON").resolve())
    if sdk_python not in sys.path:
        sys.path.insert(0, sdk_python)


def load_marvin_sdk_symbols() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Import official Marvin SDK Python symbols lazily."""
    sdk_root = resolve_marvin_sdk_root()
    _ensure_sdk_python_path(sdk_root)

    import fx_kine  # type: ignore[import-not-found]
    import fx_robot  # type: ignore[import-not-found]

    robot_cls = getattr(
        fx_robot,
        "Concise_Marvin_Robot",
        getattr(fx_robot, "Marvin_Robot"),
    )
    return (
        robot_cls,
        fx_robot.DCSS,
        fx_kine.Marvin_Kine,
        fx_kine.FX_InvKineSolvePara,
    )


@dataclass
class MarvinGripperConfig:
    channel: int
    close_command: bytes | None
    open_command: bytes | None
    settle_time: float
    initial_open: bool

    @classmethod
    def from_env(cls, arm: str) -> "MarvinGripperConfig":
        channel = _parse_int_env(f"MARVIN_GRIPPER_CHANNEL_{arm}", -1)
        if channel < 0:
            channel = _parse_int_env("MARVIN_GRIPPER_CHANNEL", 2)
        return cls(
            channel=channel,
            close_command=_parse_hex_bytes(
                _arm_env_value("MARVIN_GRIPPER_CLOSE_HEX", arm)
            ),
            open_command=_parse_hex_bytes(
                _arm_env_value("MARVIN_GRIPPER_OPEN_HEX", arm)
            ),
            settle_time=_parse_float_env("MARVIN_GRIPPER_SETTLE_TIME", 0.3),
            initial_open=_parse_bool_env("MARVIN_GRIPPER_INITIAL_OPEN", False),
        )


class Marvin:
    """Thin adapter around the official Marvin control + kinematics SDK."""

    @staticmethod
    def normalize_control_mode(control_mode: str) -> str:
        """Normalize user-facing control mode aliases to Marvin SDK modes."""
        normalized = control_mode.lower().strip()
        if normalized in {"position", "joint", POSITION_CONTROL_MODE}:
            return POSITION_CONTROL_MODE
        if normalized in {
            "impedance",
            "impedance_cartesian",
            "cartesian",
            CARTESIAN_IMPEDANCE_CONTROL_MODE,
        }:
            return CARTESIAN_IMPEDANCE_CONTROL_MODE
        if normalized in {
            "impedance_joint",
            "joint_impedance",
            JOINT_IMPEDANCE_CONTROL_MODE,
        }:
            return JOINT_IMPEDANCE_CONTROL_MODE
        raise ValueError(f"Unsupported Marvin control mode: {control_mode}")

    def __init__(
        self,
        robot_ip: str,
        kine_config_path: str | None = None,
        log_switch: int = 0,
    ) -> None:
        self._logger = get_logger()
        self._sdk_root = resolve_marvin_sdk_root()
        self._kine_config_path = (
            Path(kine_config_path).expanduser().resolve()
            if kine_config_path
            else resolve_marvin_kine_config_path(self._sdk_root)
        )
        (
            concise_robot_cls,
            dcss_cls,
            marvin_kine_cls,
            fx_inv_kine_cls,
        ) = load_marvin_sdk_symbols()

        try:
            self._robot = concise_robot_cls()
        except OSError as exc:
            raise RuntimeError(
                "Failed to load Marvin SDK shared libraries. "
                "Reinstall the Marvin env via requirements/install.sh so the SDK "
                "can be compiled locally for this machine."
            ) from exc
        self._dcss = dcss_cls()
        self._marvin_kine_cls = marvin_kine_cls
        self._fx_inv_kine_cls = fx_inv_kine_cls
        self._mode_by_arm: dict[str, str | None] = dict.fromkeys(ARM_ORDER, None)
        self._profile_by_arm = {
            arm: {
                "vel_ratio": DEFAULT_VEL_RATIO,
                "acc_ratio": DEFAULT_ACC_RATIO,
                "cart_k": DEFAULT_CART_K.copy(),
                "cart_d": DEFAULT_CART_D.copy(),
                "joint_k": DEFAULT_JOINT_K.copy(),
                "joint_d": DEFAULT_JOINT_D.copy(),
            }
            for arm in ARM_ORDER
        }
        self._gripper_cfg = {
            arm: MarvinGripperConfig.from_env(arm) for arm in ARM_ORDER
        }
        self._gripper_open = {
            arm: self._gripper_cfg[arm].initial_open for arm in ARM_ORDER
        }
        self._missing_gripper_warning: set[str] = set()
        self._kine: dict[str, Any] = {}

        self._set_sdk_log_switch(log_switch)
        if not self._connect_robot(robot_ip):
            raise RuntimeError(f"Failed to connect to Marvin robot at {robot_ip}")

        self._wait_for_udp_frames()
        self._init_kine()
        self._apply_saved_tool_info()

    @property
    def kine_config_path(self) -> Path:
        return self._kine_config_path

    @property
    def sdk_root(self) -> Path:
        return self._sdk_root

    def close(self) -> None:
        """Release robot connection."""
        try:
            self._robot.release_robot()
        except Exception:
            pass

    def _set_sdk_log_switch(self, log_switch: int) -> None:
        """Best-effort SDK log switch across Marvin Python SDK variants."""
        if not _parse_bool_env("MARVIN_ENABLE_SDK_LOG_SWITCH", False):
            return
        if not hasattr(self._robot, "log_switch"):
            return
        try:
            self._robot.log_switch(str(int(log_switch)))
        except TypeError:
            self._robot.log_switch(int(log_switch))

    def _connect_robot(self, robot_ip: str) -> bool:
        """Connect using either the concise or legacy Marvin Python SDK API."""
        try:
            return bool(self._robot.connect(robot_ip=robot_ip, log_switch=0))
        except TypeError:
            return bool(self._robot.connect(robot_ip=robot_ip))

    def _wait_for_udp_frames(self, timeout_s: float = 5.0) -> None:
        end_time = time.time() + timeout_s
        last_frame = None
        while time.time() < end_time:
            sub_data = self._robot.subscribe(self._dcss)
            if sub_data is None:
                time.sleep(0.05)
                continue
            frame = sub_data["outputs"][0]["frame_serial"]
            if frame != 0 and frame != last_frame:
                return
            last_frame = frame
            time.sleep(0.02)
        raise RuntimeError(
            "Marvin SDK connected but no UDP state frames were received. "
            "Check controller networking and firewall settings."
        )

    def _init_kine(self) -> None:
        for arm, arm_idx in ARM_INDEX.items():
            kine = self._marvin_kine_cls()
            kine.log_switch(0)
            ini_result = kine.load_config(arm_type=arm_idx, config_path=str(self._kine_config_path))
            if not ini_result:
                raise RuntimeError(
                    f"Failed to load TJ kinematics config for arm {arm}: "
                    f"{self._kine_config_path}"
                )
            ok = kine.initial_kine(
                robot_type=ini_result["TYPE"][arm_idx],
                dh=ini_result["DH"][arm_idx],
                pnva=ini_result["PNVA"][arm_idx],
                j67=ini_result["BD"][arm_idx],
            )
            if not ok:
                raise RuntimeError(f"Failed to initialize TJ kinematics for arm {arm}")
            self._kine[arm] = kine

    def _apply_saved_tool_info(self) -> None:
        try:
            tool_result = self._robot.get_tool_info()
        except Exception as exc:
            self._logger.warning("Failed to read TJ tool info: %s", exc)
            return

        tool_by_arm: dict[str, list[float]] = {}
        if tool_result in (0, None):
            return
        if isinstance(tool_result, tuple) and len(tool_result) == 2:
            tool_by_arm["A" if tool_result[0] == "line1" else "B"] = tool_result[1]
        elif (
            isinstance(tool_result, list)
            and len(tool_result) == 2
            and all(isinstance(item, list) for item in tool_result)
        ):
            tool_by_arm["A"] = tool_result[0]
            tool_by_arm["B"] = tool_result[1]
        else:
            return

        for arm, values in tool_by_arm.items():
            if len(values) != 16:
                continue
            dyn_para = values[:10]
            kine_para = values[10:]
            try:
                self._set_tool(arm=arm, kine_para=kine_para, dyn_para=dyn_para)
                tool_mat = self._kine[arm].xyzabc_to_mat4x4(kine_para)
                self._kine[arm].set_tool_kine(tool_mat)
            except Exception as exc:
                self._logger.warning(
                    "Failed to apply saved TJ tool info for arm %s: %s", arm, exc
                )

    def subscribe(self) -> dict[str, Any]:
        """Return the latest TJ subscription payload."""
        sub_data = self._robot.subscribe(self._dcss)
        if sub_data is None:
            raise RuntimeError("Marvin subscribe() returned no data.")
        return sub_data

    def is_robot_up(self, arm: str = "B") -> bool:
        """Check whether the requested arm is streaming state data."""
        arm = arm.upper()
        sub_data = self._robot.subscribe(self._dcss)
        if sub_data is None:
            return False
        idx = ARM_INDEX[arm]
        return sub_data["outputs"][idx]["frame_serial"] != 0

    def clear_errors(self, arm: str) -> None:
        """Clear arm errors."""
        arm = arm.upper()
        self._robot.clear_error(arm)
        time.sleep(0.05)

    def _clear_set(self) -> bool:
        if not hasattr(self._robot, "clear_set"):
            return True
        return bool(self._robot.clear_set())

    def _send_cmd(self) -> bool:
        if not hasattr(self._robot, "send_cmd"):
            return True
        return bool(self._robot.send_cmd())

    def _current_arm_state(self, arm: str) -> int | None:
        if not hasattr(self._robot, "subscribe"):
            return None
        try:
            sub_data = self.subscribe()
        except Exception:
            return None
        return int(sub_data["states"][ARM_INDEX[arm]]["cur_state"])

    @staticmethod
    def _transition_state_for(target_state: int) -> int | None:
        if target_state == ARM_STATE_POSITION:
            return ARM_STATE_TRANS_TO_POSITION
        if target_state == ARM_STATE_TORQ:
            return ARM_STATE_TRANS_TO_TORQ
        return None

    def _wait_for_arm_state(
        self,
        arm: str,
        target_state: int,
        timeout_s: float = 8.0,
    ) -> bool:
        if not hasattr(self._robot, "subscribe"):
            return True
        end_time = time.time() + timeout_s
        while time.time() < end_time:
            current_state = self._current_arm_state(arm)
            if current_state == target_state:
                return True
            if current_state == ARM_STATE_ERROR:
                return False
            time.sleep(0.05)
        return False

    def _send_velocity_limits(self, arm: str, vel: int, acc: int) -> bool:
        if not self._clear_set():
            return False
        ok = self._robot.set_vel_acc(arm=arm, velRatio=vel, AccRatio=acc)
        return bool(ok) and self._send_cmd()

    def _send_legacy_state_command(
        self,
        arm: str,
        target_state: int,
        impedance_type: int | None = None,
    ) -> bool:
        current_state = self._current_arm_state(arm)
        if current_state == target_state and impedance_type is None:
            return True
        if (
            current_state == self._transition_state_for(target_state)
            and impedance_type is None
        ):
            return self._wait_for_arm_state(arm, target_state)
        if not self._clear_set():
            return False
        ok = self._robot.set_state(arm=arm, state=target_state)
        if impedance_type is not None:
            ok = bool(ok) and bool(
                self._robot.set_impedance_type(arm=arm, type=impedance_type)
            )
        ok = bool(ok) and self._send_cmd()
        return bool(ok) and self._wait_for_arm_state(arm, target_state)

    def _set_tool(
        self,
        arm: str,
        kine_para: list[float],
        dyn_para: list[float],
    ) -> Any:
        """Set tool parameters across Marvin Python SDK variants."""
        try:
            return self._robot.set_tool(
                arm=arm,
                kine_para=kine_para,
                dyn_para=dyn_para,
            )
        except TypeError:
            return self._robot.set_tool(
                arm=arm,
                kineParams=kine_para,
                dynamicParams=dyn_para,
            )

    def set_position_mode(
        self,
        arm: str,
        vel_ratio: int | None = None,
        acc_ratio: int | None = None,
    ) -> None:
        """Switch an arm to position mode."""
        arm = arm.upper()
        profile = self._profile_by_arm[arm]
        vel = int(profile["vel_ratio"] if vel_ratio is None else vel_ratio)
        acc = int(profile["acc_ratio"] if acc_ratio is None else acc_ratio)
        if hasattr(self._robot, "set_position_state"):
            ok = self._robot.set_position_state(arm=arm, velRatio=vel, AccRatio=acc)
        else:
            ok = self._send_velocity_limits(arm, vel, acc)
            ok = bool(ok) and self._send_legacy_state_command(
                arm,
                ARM_STATE_POSITION,
            )
        if not ok:
            raise RuntimeError(f"Failed to switch Marvin arm {arm} to position mode.")
        profile["vel_ratio"] = vel
        profile["acc_ratio"] = acc
        self._mode_by_arm[arm] = POSITION_CONTROL_MODE
        time.sleep(0.05)

    def set_cartesian_impedance_mode(
        self,
        arm: str,
        vel_ratio: int | None = None,
        acc_ratio: int | None = None,
        cart_k: list[float] | None = None,
        cart_d: list[float] | None = None,
    ) -> None:
        """Switch an arm to cartesian impedance mode."""
        arm = arm.upper()
        profile = self._profile_by_arm[arm]
        vel = int(profile["vel_ratio"] if vel_ratio is None else vel_ratio)
        acc = int(profile["acc_ratio"] if acc_ratio is None else acc_ratio)
        cart_k = list(profile["cart_k"] if cart_k is None else cart_k)
        cart_d = list(profile["cart_d"] if cart_d is None else cart_d)
        if hasattr(self._robot, "set_imp_cart_state"):
            ok = self._robot.set_imp_cart_state(
                arm=arm,
                velRatio=vel,
                AccRatio=acc,
                K=cart_k,
                D=cart_d,
                rot_type=0,
                cart_ctrl_para=[0.0] * 7,
            )
        else:
            ok = self._send_velocity_limits(arm, vel, acc)
            ok = bool(ok) and self._send_legacy_state_command(
                arm,
                ARM_STATE_TORQ,
                impedance_type=IMPEDANCE_TYPE_CARTESIAN,
            )
            if ok:
                ok = self._clear_set()
            ok = bool(ok) and bool(
                self._robot.set_cart_kd_params(
                    arm=arm,
                    K=cart_k,
                    D=cart_d,
                    type=IMPEDANCE_TYPE_CARTESIAN,
                )
            )
            ok = bool(ok) and self._send_cmd()
        if not ok:
            raise RuntimeError(
                f"Failed to switch Marvin arm {arm} to cartesian impedance mode."
            )
        profile["vel_ratio"] = vel
        profile["acc_ratio"] = acc
        profile["cart_k"] = cart_k
        profile["cart_d"] = cart_d
        self._mode_by_arm[arm] = CARTESIAN_IMPEDANCE_CONTROL_MODE
        time.sleep(0.05)

    def set_joint_impedance_mode(
        self,
        arm: str,
        vel_ratio: int | None = None,
        acc_ratio: int | None = None,
        joint_k: list[float] | None = None,
        joint_d: list[float] | None = None,
    ) -> None:
        """Switch an arm to joint impedance mode."""
        arm = arm.upper()
        profile = self._profile_by_arm[arm]
        vel = int(profile["vel_ratio"] if vel_ratio is None else vel_ratio)
        acc = int(profile["acc_ratio"] if acc_ratio is None else acc_ratio)
        joint_k = list(profile["joint_k"] if joint_k is None else joint_k)
        joint_d = list(profile["joint_d"] if joint_d is None else joint_d)
        if hasattr(self._robot, "set_imp_joint_state"):
            ok = self._robot.set_imp_joint_state(
                arm=arm,
                velRatio=vel,
                AccRatio=acc,
                K=joint_k,
                D=joint_d,
            )
        else:
            ok = self._send_velocity_limits(arm, vel, acc)
            ok = bool(ok) and self._send_legacy_state_command(
                arm,
                ARM_STATE_TORQ,
                impedance_type=IMPEDANCE_TYPE_JOINT,
            )
            if ok:
                ok = self._clear_set()
            ok = bool(ok) and bool(
                self._robot.set_joint_kd_params(arm=arm, K=joint_k, D=joint_d)
            )
            ok = bool(ok) and self._send_cmd()
        if not ok:
            raise RuntimeError(
                f"Failed to switch Marvin arm {arm} to joint impedance mode."
            )
        profile["vel_ratio"] = vel
        profile["acc_ratio"] = acc
        profile["joint_k"] = joint_k
        profile["joint_d"] = joint_d
        self._mode_by_arm[arm] = JOINT_IMPEDANCE_CONTROL_MODE
        time.sleep(0.05)

    def update_profile(self, arm: str, params: dict[str, Any]) -> None:
        """Update cached motion profile parameters for an arm."""
        arm = arm.upper()
        profile = self._profile_by_arm[arm]
        if "vel_ratio" in params:
            profile["vel_ratio"] = int(params["vel_ratio"])
        if "acc_ratio" in params:
            profile["acc_ratio"] = int(params["acc_ratio"])
        if "joint_k" in params and len(params["joint_k"]) == 7:
            profile["joint_k"] = [float(value) for value in params["joint_k"]]
        if "joint_d" in params and len(params["joint_d"]) == 7:
            profile["joint_d"] = [float(value) for value in params["joint_d"]]
        if "cart_k" in params and len(params["cart_k"]) == 7:
            profile["cart_k"] = [float(value) for value in params["cart_k"]]
        if "cart_d" in params and len(params["cart_d"]) == 7:
            profile["cart_d"] = [float(value) for value in params["cart_d"]]

        target_mode = self._mode_by_arm[arm]
        raw_impedance_type = params.get("impedance_type")
        if raw_impedance_type is not None:
            try:
                impedance_type = int(raw_impedance_type)
            except (TypeError, ValueError):
                target_mode = self.normalize_control_mode(str(raw_impedance_type))
            else:
                if impedance_type == 1:
                    target_mode = JOINT_IMPEDANCE_CONTROL_MODE
                elif impedance_type == 2:
                    target_mode = CARTESIAN_IMPEDANCE_CONTROL_MODE
        elif "controller_type" in params:
            target_mode = self.normalize_control_mode(str(params["controller_type"]))
        elif "control_mode" in params:
            target_mode = self.normalize_control_mode(str(params["control_mode"]))

        if "stiffness" in params and len(params["stiffness"]) == 7:
            stiffness = [float(value) for value in params["stiffness"]]
            if target_mode == JOINT_IMPEDANCE_CONTROL_MODE:
                profile["joint_k"] = stiffness
            elif target_mode == CARTESIAN_IMPEDANCE_CONTROL_MODE:
                profile["cart_k"] = stiffness
        if "damping" in params and len(params["damping"]) == 7:
            damping = [float(value) for value in params["damping"]]
            if target_mode == JOINT_IMPEDANCE_CONTROL_MODE:
                profile["joint_d"] = damping
            elif target_mode == CARTESIAN_IMPEDANCE_CONTROL_MODE:
                profile["cart_d"] = damping

        if target_mode == CARTESIAN_IMPEDANCE_CONTROL_MODE:
            self.set_cartesian_impedance_mode(arm)
        elif target_mode == JOINT_IMPEDANCE_CONTROL_MODE:
            self.set_joint_impedance_mode(arm)
        elif target_mode == POSITION_CONTROL_MODE:
            self.set_position_mode(arm)

    def _ensure_mode(self, arm: str, control_mode: str) -> None:
        control_mode = self.normalize_control_mode(control_mode)
        if control_mode == POSITION_CONTROL_MODE:
            if self._mode_by_arm[arm] != POSITION_CONTROL_MODE:
                self.set_position_mode(arm)
            return
        if control_mode == JOINT_IMPEDANCE_CONTROL_MODE:
            if self._mode_by_arm[arm] != JOINT_IMPEDANCE_CONTROL_MODE:
                self.set_joint_impedance_mode(arm)
            return
        if self._mode_by_arm[arm] != CARTESIAN_IMPEDANCE_CONTROL_MODE:
            self.set_cartesian_impedance_mode(arm)

    def _fk_tcp_pose(self, arm: str, joint_deg: np.ndarray) -> np.ndarray:
        fk_mat = self._kine[arm].fk(joints=joint_deg.tolist())
        if not fk_mat:
            raise RuntimeError(f"TJ FK failed for arm {arm}.")
        fk_mat_np = np.asarray(fk_mat, dtype=np.float64)
        position = fk_mat_np[:3, 3] / 1000.0
        quat = R.from_matrix(fk_mat_np[:3, :3]).as_quat()
        return np.concatenate([position, quat], axis=0)

    def fk_tcp_pose(self, arm: str, joints_rad: np.ndarray | list[float]) -> np.ndarray:
        """Compute TCP pose from joint positions in radians."""
        arm = arm.upper()
        joint_deg = np.rad2deg(np.asarray(joints_rad, dtype=np.float64))
        return self._fk_tcp_pose(arm, joint_deg)

    def _jacobian_si(self, arm: str, joint_deg: np.ndarray) -> np.ndarray:
        jacobian = self._kine[arm].joints2JacobMatrix(joint_deg.tolist())
        if not jacobian:
            return np.zeros((6, 7), dtype=np.float64)
        jacobian_np = np.asarray(jacobian, dtype=np.float64)
        jacobian_np[:3] *= (180.0 / np.pi) / 1000.0
        return jacobian_np

    def get_joint_positions_deg(self, arm: str) -> np.ndarray:
        """Read current joint positions in degrees."""
        arm = arm.upper()
        sub_data = self.subscribe()
        idx = ARM_INDEX[arm]
        return np.asarray(sub_data["outputs"][idx]["fb_joint_pos"], dtype=np.float64)

    def get_state(self, arm: str) -> MarvinRobotState:
        """Build a MarvinRobotState for one arm."""
        arm = arm.upper()
        idx = ARM_INDEX[arm]
        sub_data = self.subscribe()
        output = sub_data["outputs"][idx]
        joint_deg = np.asarray(output["fb_joint_pos"], dtype=np.float64)
        joint_vel_deg = np.asarray(output["fb_joint_vel"], dtype=np.float64)
        tcp_pose = self._fk_tcp_pose(arm, joint_deg)
        arm_jacobian = self._jacobian_si(arm, joint_deg)
        tcp_vel = arm_jacobian @ np.deg2rad(joint_vel_deg)
        est_cart_force = np.asarray(output["est_cart_fn"], dtype=np.float64)

        gripper_open = self._gripper_open[arm]
        gripper_position = 255 if gripper_open else 0
        return MarvinRobotState(
            tcp_pose=tcp_pose,
            tcp_vel=tcp_vel,
            arm_joint_position=np.deg2rad(joint_deg),
            arm_joint_velocity=np.deg2rad(joint_vel_deg),
            tcp_force=est_cart_force[:3],
            tcp_torque=est_cart_force[3:],
            arm_jacobian=arm_jacobian,
            joints_torque=np.asarray(output["fb_joint_sToq"], dtype=np.float64),
            gripper_position=gripper_position,
            gripper_open=gripper_open,
        )

    def solve_ik(
        self,
        arm: str,
        pose: np.ndarray,
        reference_joints_deg: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        """Solve IK for a 7D pose [x, y, z, qx, qy, qz, qw]."""
        arm = arm.upper()
        pose = np.asarray(pose, dtype=np.float64)
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = R.from_quat(pose[3:]).as_matrix()
        target[:3, 3] = pose[:3] * 1000.0

        if reference_joints_deg is None:
            reference_joints = self.get_joint_positions_deg(arm)
        else:
            reference_joints = np.asarray(reference_joints_deg, dtype=np.float64)
        if abs(reference_joints[3]) < 1e-3:
            reference_joints = reference_joints.copy()
            reference_joints[3] = 0.1

        target_flat = target.reshape(-1).tolist()

        def _build_request(
            zsp_type: int,
            zsp_para: list[float] | None = None,
        ):
            solve_para = self._fx_inv_kine_cls()
            solve_para.set_input_ik_target_tcp(target_flat)
            solve_para.set_input_ik_ref_joint(reference_joints.tolist())
            solve_para.set_input_ik_zsp_type(zsp_type)
            if zsp_para is not None:
                solve_para.set_input_ik_zsp_para(zsp_para)
            return solve_para

        # First try the simplest "closest-to-reference-joints" solve.
        result = self._kine[arm].ik(_build_request(zsp_type=0))
        if result:
            return np.asarray(result.m_Output_RetJoint.to_list(), dtype=np.float64)

        # If that fails, fall back to the current arm-angle plane guidance.
        fk_nsp = self._kine[arm].fk_nsp(reference_joints.tolist())
        if fk_nsp:
            _, nsp_mat = fk_nsp
            zsp_para = [nsp_mat[0][0], nsp_mat[1][0], nsp_mat[2][0], 0.0, 0.0, 0.0]
            result = self._kine[arm].ik(_build_request(zsp_type=1, zsp_para=zsp_para))
            if result:
                return np.asarray(result.m_Output_RetJoint.to_list(), dtype=np.float64)

        raise RuntimeError(f"Marvin IK failed for arm {arm}.")

    def _send_joint_command(self, arm: str, joints_deg: list[float]) -> bool:
        """Send a joint position command across Marvin SDK variants."""
        if hasattr(self._robot, "set_joint_position_cmd"):
            return bool(self._robot.set_joint_position_cmd(arm=arm, joint=joints_deg))

        if not self._clear_set():
            return False
        ok = self._robot.set_joint_cmd_pose(arm=arm, joints=joints_deg)
        return bool(ok) and self._send_cmd()

    def move_joint_positions(
        self,
        arm: str,
        joints_rad: np.ndarray | list[float],
        control_mode: str = POSITION_CONTROL_MODE,
    ) -> np.ndarray:
        """Move an arm by joint command in radians."""
        arm = arm.upper()
        joints_deg = np.rad2deg(np.asarray(joints_rad, dtype=np.float64))
        self._ensure_mode(arm, control_mode)
        if not self._send_joint_command(arm=arm, joints_deg=joints_deg.tolist()):
            raise RuntimeError(f"Failed to send joint command to Marvin arm {arm}.")
        return joints_deg

    def move_pose(
        self,
        arm: str,
        pose: np.ndarray,
        control_mode: str = CARTESIAN_IMPEDANCE_CONTROL_MODE,
    ) -> np.ndarray:
        """Move an arm to a cartesian pose by IK + joint command."""
        arm = arm.upper()
        self._ensure_mode(arm, control_mode)
        target_joints_deg = self.solve_ik(arm=arm, pose=np.asarray(pose))
        if not self._send_joint_command(
            arm=arm,
            joints_deg=target_joints_deg.tolist(),
        ):
            raise RuntimeError(
                f"Failed to send cartesian IK joint command to Marvin arm {arm}."
            )
        return target_joints_deg

    def _send_tool_command(self, arm: str, payload: bytes | None) -> bool:
        arm = arm.upper()
        if payload is None:
            if arm not in self._missing_gripper_warning:
                self._logger.warning(
                    "No TJ gripper hex command configured for arm %s. "
                    "Set MARVIN_GRIPPER_OPEN_HEX / MARVIN_GRIPPER_CLOSE_HEX "
                    "(optionally suffixed with _A or _B) to enable gripper control.",
                    arm,
                )
                self._missing_gripper_warning.add(arm)
            return False

        config = self._gripper_cfg[arm]
        self._clear_tool_channel(arm)
        sent = self._set_tool_channel(
            arm=arm,
            data=payload,
            channel=config.channel,
        )
        if sent <= 0:
            return False
        time.sleep(config.settle_time)
        return True

    def _clear_tool_channel(self, arm: str) -> None:
        """Clear the Marvin end-effector communication channel."""
        if hasattr(self._robot, "clear_ch_data"):
            self._robot.clear_ch_data(arm)
            return
        self._robot.clear_485_cache(arm)

    def _set_tool_channel(self, arm: str, data: bytes, channel: int) -> int:
        """Send bytes to the Marvin end-effector communication channel."""
        if hasattr(self._robot, "set_ch_data"):
            return int(
                self._robot.set_ch_data(
                    arm=arm,
                    data=data,
                    size_int=len(data),
                    set_ch=channel,
                )
            )
        ok, sent = self._robot.set_485_data(
            arm=arm,
            data=data,
            size_int=len(data),
            com=channel,
        )
        if not ok:
            return 0
        return int(sent)

    def open_gripper(self, arm: str) -> bool:
        """Open a configured end-effector gripper."""
        arm = arm.upper()
        ok = self._send_tool_command(arm, self._gripper_cfg[arm].open_command)
        if ok:
            self._gripper_open[arm] = True
        return ok

    def close_gripper(self, arm: str) -> bool:
        """Close a configured end-effector gripper."""
        arm = arm.upper()
        ok = self._send_tool_command(arm, self._gripper_cfg[arm].close_command)
        if ok:
            self._gripper_open[arm] = False
        return ok

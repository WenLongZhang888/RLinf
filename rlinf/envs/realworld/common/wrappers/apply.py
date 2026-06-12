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

"""Wrapper-stack builders shared by realworld task factories."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym

from rlinf.envs.realworld.common.wrappers.dual_euler_obs import (
    DualQuat2EulerWrapper,
)
from rlinf.envs.realworld.common.wrappers.dual_gello_intervention import (
    DualGelloIntervention,
)
from rlinf.envs.realworld.common.wrappers.dual_relative_frame import (
    DualRelativeFrame,
)
from rlinf.envs.realworld.common.wrappers.dual_spacemouse_intervention import (
    DualSpacemouseIntervention,
)
from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.gello_intervention import (
    GelloIntervention,
)
from rlinf.envs.realworld.common.wrappers.gripper_close import GripperCloseEnv
from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame
from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
)
from rlinf.envs.realworld.common.wrappers.spacemouse_intervention import (
    SpacemouseIntervention,
)
from rlinf.envs.realworld.common.wrappers.teleop_intervention import (
    TeleopInterventionWrapper,
)
from rlinf.envs.realworld.common.wrappers.vr_intervention import (
    VRTeleopIntervention,
)

# Keyword arguments accepted by ``VRArmTeleop`` — used to filter the shared
# ``vr_config`` block (which may also carry gripper-only keys like
# ``gripper_threshold`` and the wrapper-level ``hold_time``).
_VR_ARM_TELEOP_KEYS = frozenset(
    {
        "side",
        "workspace_limits",
        "ema_trans",
        "ema_rot",
        "translation_scale",
        "xyz_scale",
        "track_rotation",
        "max_step_m",
        "max_rot_deg",
        "dummy_on_missing",
        "motion_threshold",
    }
)


def _load_dexhand_intervention():
    """Import DexHandIntervention only when dex-hand teleop is requested."""
    try:
        from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
            DexHandIntervention,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "rlinf_dexhand":
            raise ModuleNotFoundError(
                "DexHandIntervention requires optional dependency "
                "'rlinf_dexhand'. Install it before enabling "
                "dexterous-hand teleoperation."
            ) from exc
        raise
    return DexHandIntervention


def _teleop_mode(cfg: Mapping[str, Any], *, default_spacemouse: bool = True) -> str:
    mode = cfg.get("teleop_mode", None)
    if mode is not None:
        return str(mode).lower()
    legacy_keys = ("use_spacemouse", "use_vr", "use_gello")
    explicit_enabled = [
        name
        for name, key in (
            ("spacemouse", "use_spacemouse"),
            ("vr", "use_vr"),
            ("gello", "use_gello"),
        )
        if key in cfg and bool(cfg.get(key))
    ]
    if any(key in cfg for key in legacy_keys):
        enabled = explicit_enabled
    elif default_spacemouse:
        enabled = ["spacemouse"]
    else:
        enabled = []
    if len(enabled) > 1:
        raise ValueError(
            "Only one teleop input can be active at a time. "
            f"Got enabled inputs: {enabled}. Prefer teleop_mode."
        )
    if "vr" in enabled:
        return "vr"
    if "gello" in enabled:
        return "gello"
    if "spacemouse" in enabled:
        return "spacemouse"
    return "none"


def _validate_teleop_mode(mode: str) -> None:
    if mode not in {"none", "spacemouse", "vr", "gello"}:
        raise ValueError(
            "teleop_mode must be one of 'none', 'spacemouse', 'vr', or 'gello'. "
            f"Got {mode!r}."
        )


def _apply_keyboard_reward(env: gym.Env, mode: Optional[str]) -> gym.Env:
    if env.config.is_dummy or not mode:
        return env
    if mode == "multi_stage":
        return KeyboardRewardDoneMultiStageWrapper(env)
    if mode == "single_stage":
        return KeyboardRewardDoneWrapper(env)
    return env


def apply_single_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for single-arm realworld envs (franka single, xsquare)."""
    end_effector_type = str(
        getattr(getattr(env, "config", None), "end_effector_type", "franka_gripper")
    )
    is_dex_hand = end_effector_type.endswith("hand")
    is_revo2_hand = end_effector_type == "revo2_hand"

    no_gripper = cfg.get("no_gripper", True)
    if no_gripper and not is_dex_hand:
        env = GripperCloseEnv(env)

    mode = _teleop_mode(cfg)
    _validate_teleop_mode(mode)

    gripper_enabled = not no_gripper

    if not env.config.is_dummy and mode == "spacemouse":
        if is_dex_hand:
            glove_cfg = cfg.get("glove_config", {})
            DexHandIntervention = _load_dexhand_intervention()
            env = DexHandIntervention(
                env,
                left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
                right_port=glove_cfg.get("right_port", None),
                glove_frequency=glove_cfg.get("frequency", 60),
                glove_config_file=glove_cfg.get("config_file", None),
            )
        else:
            spacemouse_cfg = cfg.get("spacemouse_config", {})
            env = SpacemouseIntervention(
                env,
                gripper_enabled=gripper_enabled,
                **spacemouse_cfg,
            )

    if not env.config.is_dummy and mode == "vr":
        vr_cfg = dict(cfg.get("vr_config", {}))
        if is_revo2_hand:
            # VR arm teleop composed with Revo2 hand retargeting through the
            # generic wrapper. ``vr_config`` drives the arm; ``hand_teleop_config``
            # (mode, thumb_opposition, engage_threshold) drives the hand.
            from rlinf.envs.realworld.common.teleop import (
                VRArmTeleop,
                VRHandRetargetTeleop,
            )

            hold_time = float(vr_cfg.get("hold_time", 0.5))
            arm_kwargs = {k: v for k, v in vr_cfg.items() if k in _VR_ARM_TELEOP_KEYS}
            hand_cfg = dict(cfg.get("hand_teleop_config", {}))
            env = TeleopInterventionWrapper(
                env,
                VRArmTeleop(**arm_kwargs),
                VRHandRetargetTeleop(**hand_cfg),
                hold_time=hold_time,
            )
        elif is_dex_hand:
            raise ValueError(
                "teleop_mode='vr' is only supported for gripper or "
                "'revo2_hand' end-effectors."
            )
        else:
            env = VRTeleopIntervention(
                env,
                gripper_enabled=gripper_enabled,
                **vr_cfg,
            )

    if not env.config.is_dummy and mode == "gello":
        if is_dex_hand:
            raise ValueError("use_gello=True is not supported for ruiyan_hand.")
        gello_port = cfg.get("gello_port", None)
        if gello_port is None:
            raise ValueError(
                "use_gello=True requires 'gello_port' in the env config "
                "(e.g. env.eval.gello_port)."
            )
        env = GelloIntervention(env, port=gello_port, gripper_enabled=gripper_enabled)

    env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    return env


def apply_dual_arm_wrappers(env: gym.Env, cfg: Mapping[str, Any]) -> gym.Env:
    """Wrapper stack for dual-arm realworld envs (dual-franka today)."""
    if cfg.get("no_gripper", True):
        # No DualGripperCloseEnv yet, so a 12D action would blow up as reshape(2,7).
        raise NotImplementedError(
            "no_gripper=True is not yet supported for dual-arm envs: "
            "DualGripperCloseEnv is not implemented. "
            "Set env.eval.no_gripper=False (or env.train.no_gripper=False)."
        )

    mode = _teleop_mode(cfg)
    _validate_teleop_mode(mode)

    gripper_enabled = True

    if not env.config.is_dummy and mode == "spacemouse":
        env = DualSpacemouseIntervention(env, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and mode == "vr":
        raise ValueError("teleop_mode='vr' is not implemented for dual-arm envs.")

    if not env.config.is_dummy and mode == "gello":
        left_port = cfg.get("left_gello_port", None)
        right_port = cfg.get("right_gello_port", None)
        if left_port is None or right_port is None:
            raise ValueError(
                "use_gello=True on a dual-arm env requires both "
                "'left_gello_port' and 'right_gello_port' in the env config."
            )
        env = DualGelloIntervention(
            env,
            left_port=left_port,
            right_port=right_port,
            gripper_enabled=gripper_enabled,
        )

    env = _apply_keyboard_reward(env, cfg.get("keyboard_reward_wrapper", None))

    if cfg.get("use_relative_frame", True):
        env = DualRelativeFrame(env)
    env = DualQuat2EulerWrapper(env)
    return env

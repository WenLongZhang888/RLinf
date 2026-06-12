# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers import (
    Quat2EulerWrapper,
    RelativeFrame,
    SpacemouseIntervention,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.marvin.tasks.bottle import BottleEnv as BottleEnv
from rlinf.envs.realworld.marvin.tasks.marvin_bin_relocation import (
    MarvinBinRelocationEnv as MarvinBinRelocationEnv,
)
from rlinf.envs.realworld.marvin.tasks.material_grab_and_place import (
    MaterialBinaryRewardClassifierWrapper,
    MaterialTcpTranslationCommandRotateWrapper,
)
from rlinf.envs.realworld.marvin.tasks.material_grab_and_place import (
    MaterialGrabAndPlaceEnv as MaterialGrabAndPlaceEnv,
)
from rlinf.envs.realworld.marvin.tasks.peg_insertion_env import (
    PegInsertionEnv as PegInsertionEnv,
)
from rlinf.envs.realworld.marvin.wrappers import SPACEMOUSE_WIRELESS_REMAP


def _with_marvin_teleop_defaults(env_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(env_cfg)
    spacemouse_cfg = dict(cfg.get("spacemouse_config", {}))
    spacemouse_cfg.setdefault("axis_remap", SPACEMOUSE_WIRELESS_REMAP)
    cfg["spacemouse_config"] = spacemouse_cfg
    return cfg


def create_marvin_peg_insertion_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = PegInsertionEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, _with_marvin_teleop_defaults(env_cfg))


def create_marvin_bin_relocation_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = MarvinBinRelocationEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, _with_marvin_teleop_defaults(env_cfg))


def create_marvin_bottle_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = BottleEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, _with_marvin_teleop_defaults(env_cfg))


def create_marvin_material_grab_and_place_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = MaterialGrabAndPlaceEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    env_cfg = _with_marvin_teleop_defaults(env_cfg)
    if env_cfg.get("use_relative_frame", True):
        env = RelativeFrame(env, include_relative_pose=False)
    env = MaterialTcpTranslationCommandRotateWrapper(
        env,
        rotation_matrix=env.unwrapped.config.tcp_translation_rot,
    )
    mode = str(env_cfg.get("teleop_mode", "spacemouse")).lower()
    if (
        not env.unwrapped.config.is_dummy
        and mode == "spacemouse"
        and bool(env_cfg.get("use_spacemouse", True))
    ):
        spacemouse_cfg = dict(env_cfg.get("spacemouse_config", {}))
        env = SpacemouseIntervention(
            env,
            gripper_enabled=not env_cfg.get("no_gripper", True),
            **spacemouse_cfg,
        )
    env = Quat2EulerWrapper(env)
    classifier_cfg = getattr(env.unwrapped.config, "classifier", None)
    if classifier_cfg is not None and getattr(classifier_cfg, "enabled", False):
        env = MaterialBinaryRewardClassifierWrapper(
            env,
            checkpoint_path=classifier_cfg.checkpoint_path,
            image_keys=classifier_cfg.image_keys,
            threshold=classifier_cfg.threshold,
            debug=classifier_cfg.debug,
        )
    return env


register(
    id="MarvinPegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.marvin.tasks:create_marvin_peg_insertion_env",
)

register(
    id="MarvinBinRelocationEnv-v1",
    entry_point="rlinf.envs.realworld.marvin.tasks:create_marvin_bin_relocation_env",
)

register(
    id="MarvinBottleEnv-v1",
    entry_point="rlinf.envs.realworld.marvin.tasks:create_marvin_bottle_env",
)

register(
    id="MarvinMaterialGrabAndPlaceEnv-v1",
    entry_point="rlinf.envs.realworld.marvin.tasks:create_marvin_material_grab_and_place_env",
)

__all__ = [
    "BottleEnv",
    "MarvinBinRelocationEnv",
    "MaterialGrabAndPlaceEnv",
    "PegInsertionEnv",
    "create_marvin_bottle_env",
    "create_marvin_bin_relocation_env",
    "create_marvin_material_grab_and_place_env",
    "create_marvin_peg_insertion_env",
]

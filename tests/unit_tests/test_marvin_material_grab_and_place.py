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

import gymnasium as gym
import numpy as np

import rlinf.envs.realworld.marvin.tasks  # noqa: F401
from rlinf.envs.realworld.marvin.tasks.material_grab_and_place import (
    MaterialGrabAndPlaceConfig,
)


def test_material_config_defaults():
    cfg = MaterialGrabAndPlaceConfig(
        is_dummy=True,
        camera_serials={
            "wrist_1": "230322273834",
            "wrist_2": "230322274885",
            "side_policy": "335122271065",
            "side_classifier": "335122271065",
        },
    )

    assert cfg.task_description == "material grab and place"
    np.testing.assert_allclose(cfg.action_scale, np.array([0.001, 0.0, 1.0]))
    assert cfg.classifier.enabled is True
    assert cfg.camera_serials["side_policy"] == "335122271065"


def test_material_env_registers_and_resets_dummy_without_classifier():
    env = gym.make(
        "MarvinMaterialGrabAndPlaceEnv-v1",
        disable_env_checker=True,
        override_cfg={
            "is_dummy": True,
            "camera_serials": {
                "wrist_1": "230322273834",
                "wrist_2": "230322274885",
                "side_policy": "335122271065",
                "side_classifier": "335122271065",
            },
            "classifier": {"enabled": False},
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={
            "teleop_mode": "none",
            "use_spacemouse": False,
            "no_gripper": False,
            "use_relative_frame": True,
        },
    )
    try:
        obs, _ = env.reset()
        assert env.action_space.shape == (7,)
        assert set(obs["frames"]) == {
            "wrist_1",
            "wrist_2",
            "side_policy",
            "side_classifier",
        }
        assert obs["state"]["tcp_pose"].shape == (6,)
        assert obs["state"]["joint_pos"].shape == (7,)
    finally:
        env.close()

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

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..marvin_env import MarvinEnv, MarvinRobotConfig

# Default target pose (reference TARGET_POSE / RESET_POSE from usb_pickup_insertion)
_DEFAULT_TARGET_POSE = np.array(
    [0.48537828, -0.17391559, 0.17181447, -1.5444949, -0.0084113264, -3.1288216]
)


@dataclass
class PegInsertionConfig(MarvinRobotConfig):
    """Config for peg insertion task aligned with usb_pickup_insertion config/wrapper."""

    target_ee_pose: np.ndarray = field(default_factory=lambda: _DEFAULT_TARGET_POSE.copy())
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.02
    random_z_range: float = 0.02
    random_rz_range: float = 0.0
    enable_random_reset: bool = True
    spacemouse_open_loop: bool = True
    fixed_gripper_closed: bool = True
    update_controller_params: bool = False
    lock_orientation: bool = True
    reset_move_wait: float = 0.5
    reset_final_wait: float = 1.0
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        # Compliance / precision params from reference COMPLIANCE_PARAM / PRECISION_PARAM
        self.compliance_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0,
            "translational_clip_x": 0.006,
            "translational_clip_y": 0.0059,
            "translational_clip_z": 0.0035,
            "translational_clip_neg_x": 0.005,
            "translational_clip_neg_y": 0.005,
            "translational_clip_neg_z": 0.0035,
            "rotational_clip_x": 0.02,
            "rotational_clip_y": 0.02,
            "rotational_clip_z": 0.015,
            "rotational_clip_neg_x": 0.02,
            "rotational_clip_neg_y": 0.02,
            "rotational_clip_neg_z": 0.015,
            "rotational_Ki": 0,
        }
        self.precision_param = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
            "translational_Ki": 0.0,
            "translational_clip_x": 0.01,
            "translational_clip_y": 0.01,
            "translational_clip_z": 0.01,
            "translational_clip_neg_x": 0.01,
            "translational_clip_neg_y": 0.01,
            "translational_clip_neg_z": 0.01,
            "rotational_clip_x": 0.03,
            "rotational_clip_y": 0.03,
            "rotational_clip_z": 0.03,
            "rotational_clip_neg_x": 0.03,
            "rotational_clip_neg_y": 0.03,
            "rotational_clip_neg_z": 0.03,
            "rotational_Ki": 0.0,
        }
        self.target_ee_pose = np.array(self.target_ee_pose, dtype=np.float64)
        self.reset_ee_pose = np.array(self.target_ee_pose, dtype=np.float64)
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array([0.004, 0.1, 1])
        # ee_pose_limit from reference ABS_POSE_LIMIT: x/z ±0.03, y: -0.075 / +0.03
        t = self.target_ee_pose
        pi = np.pi
        self.ee_pose_limit_min = np.array(
            [t[0] - 0.03, t[1] - 0.075, t[2] - 0.03, -pi, -pi, -pi]
        )
        self.ee_pose_limit_max = np.array(
            [t[0] + 0.03, t[1] + 0.03, t[2] + 0.03, pi, pi, pi]
        )


class PegInsertionEnv(MarvinEnv):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = PegInsertionConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "peg and insertion"

    def _gripper_action(self, position: float, is_binary: bool = True):
        """If fixed_gripper_closed, do not send gripper commands (align with reference)."""
        if getattr(self.config, "fixed_gripper_closed", False):
            return False
        return super()._gripper_action(position, is_binary)

    def reset(self, joint_reset=False, seed=None, options=None):
        """Reset flow aligned with reference USBEnv.reset: close gripper → precision → above target → target → go_to_rest → final wait."""
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0
        if not self._controller_type_applied:
            self._controller.start_controller(self.config.controller_type).wait()
            self._controller_type_applied = True

        if self._recording_frames:
            self._save_video_recording()
            self._recording_frames.clear()

        joint_reset_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_reset_cycle == 0
        if joint_reset:
            self._logger.info(
                "Number of resets reached %s, resetting joints to initial position.",
                self.config.joint_reset_cycle,
            )

        if getattr(self.config, "fixed_gripper_closed", False):
            self._controller.close_gripper().wait()
            time.sleep(0.2)

        if self.config.update_controller_params and self.config.precision_param:
            self._controller.reconfigure_precision_params(
                self.config.precision_param
            ).wait()

        self._marvin_state = self._controller.get_state().wait()[0]
        move_wait = getattr(self.config, "reset_move_wait", 0.5)
        target_6d = self.config.target_ee_pose
        target_7d = np.concatenate(
            [
                target_6d[:3],
                R.from_euler("xyz", target_6d[3:].copy()).as_quat(),
            ]
        ).astype(np.float64)
        above_pose = self._marvin_state.tcp_pose.copy()
        above_pose[1] = float(target_6d[1]) + 0.04
        self._interpolate_move(above_pose, timeout=move_wait)
        time.sleep(move_wait)
        self._interpolate_move(target_7d, timeout=move_wait)
        time.sleep(move_wait)

        self.go_to_rest(joint_reset)

        self._clear_error()
        self._num_steps = 0
        self._marvin_state = self._controller.get_state().wait()[0]
        if self.config.lock_orientation:
            self._locked_quat = self._marvin_state.tcp_pose[3:].copy()
        if self.config.spacemouse_open_loop:
            self._cmd_pose = self._marvin_state.tcp_pose.copy()

        final_wait = getattr(self.config, "reset_final_wait", 1.0)
        time.sleep(final_wait)
        return self._get_observation(), {}

    def go_to_rest(self, joint_reset=False):
        """Reset pose logic aligned with reference go_to_reset: joint reset → random/fixed reset_pose → interpolate → compliance."""
        if not self.config.is_dummy:
            if joint_reset:
                self._controller.reset_joint(self.config.joint_reset_qpos).wait()
                time.sleep(0.5)

            if self.config.enable_random_reset:
                reset_pose = self._reset_pose.copy()
                reset_pose[:2] += np.random.uniform(
                    -self.config.random_xy_range,
                    self.config.random_xy_range,
                    (2,),
                )
                if self.config.random_z_range != 0:
                    reset_pose[2] += np.random.uniform(
                        -self.config.random_z_range,
                        self.config.random_z_range,
                    )
                euler_random = self.config.reset_ee_pose[3:].copy()
                if self.config.random_rz_range != 0:
                    euler_random[2] += np.random.uniform(
                        -self.config.random_rz_range,
                        self.config.random_rz_range,
                    )
                reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
            else:
                reset_pose = self._reset_pose.copy()

            self._interpolate_move(reset_pose, timeout=getattr(self.config, "reset_move_wait", 1.0))

            if self.config.update_controller_params:
                self._controller.reconfigure_compliance_params(
                    self.config.compliance_param
                ).wait()

            self._marvin_state = self._controller.get_state().wait()[0]
            cnt = 0
            while not np.allclose(
                self._marvin_state.tcp_pose[:3], reset_pose[:3], atol=0.02
            ):
                cnt += 1
                self._interpolate_move(
                    reset_pose,
                    timeout=getattr(self.config, "reset_move_wait", 1.0),
                )
                self._marvin_state = self._controller.get_state().wait()[0]
                if cnt > 2:
                    break

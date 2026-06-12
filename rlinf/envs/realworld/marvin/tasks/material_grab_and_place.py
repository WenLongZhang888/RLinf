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

"""Material grab-and-place task for the Marvin real-world setup."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import CameraInfo, create_camera

from ..marvin_env import MarvinEnv, MarvinRobotConfig

_DEFAULT_TARGET_POSE = np.array(
    [0.48537828, -0.17391559, 0.17181447, -1.5444949, -0.0084113264, -3.1288216],
    dtype=np.float64,
)
_FIXED_RESET_QUAT = R.from_euler("xyz", np.deg2rad([-90.0, 0.0, -180.0])).as_quat()


@dataclass
class MaterialGrabAndPlaceClassifierConfig:
    """Binary reward classifier config for material grab-and-place."""

    enabled: bool = True
    checkpoint_path: str | None = None
    image_keys: list[str] = field(default_factory=lambda: ["side_classifier"])
    threshold: float = 0.8
    debug: bool = True


@dataclass
class MaterialGrabAndPlaceConfig(MarvinRobotConfig):
    """Config aligned with tianji_hilserl material_grab_and_place."""

    target_ee_pose: np.ndarray = field(
        default_factory=lambda: _DEFAULT_TARGET_POSE.copy()
    )
    reset_ee_pose: np.ndarray = field(
        default_factory=lambda: _DEFAULT_TARGET_POSE.copy()
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.02
    random_z_range: float = 0.02
    random_rz_range: float = 0.1
    enable_random_reset: bool = False
    reset_pallet_approach_sleep_s: float = 2.0
    reset_pre_lift_y: float = 0.06
    reset_pre_lift_duration_s: float = 1.0
    reset_jointreset_retries: int = 1
    preserve_reference_reset: bool = True
    lock_tcp_orientation: bool = True
    classifier: MaterialGrabAndPlaceClassifierConfig | dict[str, Any] = field(
        default_factory=MaterialGrabAndPlaceClassifierConfig
    )
    tcp_translation_rot: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    get_high_joint_target: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.868563, -1.718057, -1.480224, -1.752637, 0.225669, -0.678226, 0.096122],
            dtype=np.float64,
        )
    )
    get_low_joint_target: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.547826, -1.712943, -1.486736, -1.655913, 0.190307, -0.443616, 0.086038],
            dtype=np.float64,
        )
    )
    put_high_joint_target: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.834177, -0.957887, -1.443988, -1.722619, -0.40249, -0.758867, 0.277139],
            dtype=np.float64,
        )
    )
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [
            0.55201099,
            -1.57590316,
            -1.56055124,
            -1.3581873,
            0.00168599,
            -0.75184421,
            0.03362202,
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.task_description = "material grab and place"
        self.compliance_param = {
            "impedance_type": 1,
            "stiffness": [2.0, 2.0, 2.0, 1.6, 1.0, 1.0, 1.0],
            "damping": [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2],
        }
        self.precision_param = {
            "impedance_type": 2,
            "stiffness": [2000.0, 2000.0, 2000.0, 40.0, 40.0, 40.0, 20.0],
            "damping": [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1.0],
        }
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64)
        self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64)
        self.reward_threshold = np.asarray(self.reward_threshold, dtype=np.float64)
        self.action_scale = np.array([0.001, 0.0, 1.0], dtype=np.float64)
        self.ee_pose_limit_min = np.array(
            [
                self.target_ee_pose[0] - 0.18,
                self.target_ee_pose[1] - 0.086,
                self.target_ee_pose[2] - 0.15,
                -np.pi,
                -np.pi,
                -np.pi,
            ],
            dtype=np.float64,
        )
        self.ee_pose_limit_max = np.array(
            [
                self.target_ee_pose[0] + 0.08,
                self.target_ee_pose[1] + 0.08,
                self.target_ee_pose[2] + 0.15,
                np.pi,
                np.pi,
                np.pi,
            ],
            dtype=np.float64,
        )
        self.tcp_translation_rot = np.asarray(
            self.tcp_translation_rot, dtype=np.float64
        ).reshape(3, 3)
        self.get_high_joint_target = np.asarray(
            self.get_high_joint_target, dtype=np.float64
        )
        self.get_low_joint_target = np.asarray(
            self.get_low_joint_target, dtype=np.float64
        )
        self.put_high_joint_target = np.asarray(
            self.put_high_joint_target, dtype=np.float64
        )
        if isinstance(self.classifier, dict):
            self.classifier = MaterialGrabAndPlaceClassifierConfig(**self.classifier)


class MaterialGrabAndPlaceEnv(MarvinEnv):
    """Material grab-and-place task with the reference reset trajectory."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = MaterialGrabAndPlaceConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return "material grab and place"

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

    def _init_action_obs_spaces(self):
        super()._init_action_obs_spaces()
        self.observation_space["state"]["joint_pos"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(7,)
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        name = getattr(self, "_active_camera_name", None)
        if name == "side_policy":
            cropped = np.rot90(frame[110:-110, 100:-100], k=1)
            return cropped, cv2.resize(cropped, reshape_size)
        if name == "side_classifier":
            h, w = frame.shape[:2]
            cropped = np.rot90(
                frame[(2 * h) // 7 : h - (2 * h) // 7, (2 * w) // 7 : w - (2 * w) // 7],
                k=1,
            )
            return cropped, cv2.resize(cropped, reshape_size)
        return super()._crop_frame(frame, reshape_size)

    def _read_one_camera(self, camera):
        self._active_camera_name = camera._camera_info.name
        try:
            return super()._read_one_camera(camera)
        finally:
            self._active_camera_name = None

    def _open_cameras(self):
        self._cameras = []
        self._camera_aliases: dict[str, Any] = {}
        if self.config.camera_serials is None:
            return
        by_serial = {}
        for name, serial in self._get_camera_name_serial_pairs():
            if serial not in by_serial:
                camera = create_camera(CameraInfo(name=name, serial_number=serial))
                if not self.config.is_dummy:
                    camera.open()
                by_serial[serial] = camera
                self._cameras.append(camera)
            self._camera_aliases[name] = by_serial[serial]

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        display_frames = {}
        full_res_for_recording: dict[str, np.ndarray] = {}
        serial_frames = {}

        try:
            for camera in self._cameras:
                serial_frames[camera._camera_info.serial_number] = camera.get_frame()
            for name, camera in self._camera_aliases.items():
                frame = serial_frames[camera._camera_info.serial_number]
                reshape_size = self.observation_space["frames"][name].shape[:2][::-1]
                self._active_camera_name = name
                try:
                    cropped, resized = self._crop_frame(frame, reshape_size)
                finally:
                    self._active_camera_name = None
                if resized.shape[2] == 3:
                    for_obs = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                else:
                    for_obs = resized[..., :3].copy()
                frames[name] = for_obs
                display_frames[name] = resized
                display_frames[f"{name}_full"] = cropped
                full_res_for_recording[name] = cropped.copy()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "A material task camera is not producing frames (%s). "
                "Wait 5 seconds and try again.",
                exc,
            )
            time.sleep(5)
            self._close_cameras()
            self._open_cameras()
            return self._get_camera_frames()

        if self.config.save_video_path and full_res_for_recording:
            self._recording_frames.append(full_res_for_recording)
        if hasattr(self, "camera_player") and self.camera_player is not None:
            self.camera_player.put_frame(display_frames)
        return frames

    def _get_observation(self) -> dict:
        obs = super()._get_observation()
        if not self.config.is_dummy:
            obs["state"]["joint_pos"] = self._marvin_state.arm_joint_position
        else:
            obs["state"]["joint_pos"] = np.zeros(7, dtype=np.float64)
        return obs

    def _interpolate_line_xyz_fixed_quat(
        self, xyz_goal: np.ndarray, duration_s: float
    ) -> None:
        self._marvin_state = self._controller.get_state().wait()[0]
        start = self._marvin_state.tcp_pose[:3].copy()
        goal = np.asarray(xyz_goal, dtype=np.float64).reshape(3)
        quat = self._normalize_quat(_FIXED_RESET_QUAT)
        steps = max(2, int(float(duration_s) * self.config.step_frequency))
        for i in range(1, steps + 1):
            alpha = i / steps
            pos = (1.0 - alpha) * start + alpha * goal
            self._move_action(np.concatenate([pos, quat]))
            time.sleep(1.0 / self.config.step_frequency)
        self._marvin_state = self._controller.get_state().wait()[0]
        self._cmd_pose = np.concatenate([goal, quat])

    def _move_joints_and_sync(self, joints_rad: np.ndarray, wait_s: float) -> None:
        joints_rad = np.asarray(joints_rad, dtype=np.float64).reshape(7)
        self._controller.move_joint_positions(joints_rad).wait()
        time.sleep(float(wait_s))
        self._marvin_state = self._controller.get_state().wait()[0]
        self._cmd_pose = self._marvin_state.tcp_pose.copy()

    def _fk_tcp_pose_from_joints(self, joints_rad: np.ndarray) -> np.ndarray | None:
        try:
            return self._controller.fk_tcp_pose(
                np.asarray(joints_rad, dtype=np.float64).reshape(7)
            ).wait()[0]
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Material reset FK failed: %s", exc)
            return None

    def _run_reference_material_reset(self) -> None:
        self._controller.open_gripper().wait()
        time.sleep(0.5)

        self._marvin_state = self._controller.get_state().wait()[0]
        lift_goal = self._marvin_state.tcp_pose[:3].copy()
        lift_goal[1] += float(self.config.reset_pre_lift_y)
        self._interpolate_line_xyz_fixed_quat(
            lift_goal, self.config.reset_pre_lift_duration_s
        )

        line_dur = float(self.config.reset_pallet_approach_sleep_s)
        q_high = self.config.get_high_joint_target
        q_low = self.config.get_low_joint_target
        q_put = self.config.put_high_joint_target
        pose_high = self._fk_tcp_pose_from_joints(q_high)
        pose_low = self._fk_tcp_pose_from_joints(q_low)
        pose_put = self._fk_tcp_pose_from_joints(q_put)

        if pose_high is not None and pose_low is not None and pose_put is not None:
            self._interpolate_line_xyz_fixed_quat(pose_high[:3], line_dur)
            time.sleep(1.0)
            self._move_joints_and_sync(q_high, 1.0)
            self._interpolate_line_xyz_fixed_quat(pose_low[:3], line_dur)
            time.sleep(1.0)
            self._controller.close_gripper().wait()
            time.sleep(0.5)
            self._interpolate_line_xyz_fixed_quat(pose_high[:3], line_dur)
            time.sleep(1.0)
            self._interpolate_line_xyz_fixed_quat(pose_put[:3], line_dur)
            time.sleep(1.0)
            self._move_joints_and_sync(q_put, 1.0)
            return

        self._move_joints_and_sync(q_high, line_dur)
        self._move_joints_and_sync(q_low, line_dur)
        self._controller.close_gripper().wait()
        time.sleep(2.5)
        self._move_joints_and_sync(q_high, line_dur)

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            return self._get_observation(), {}

        self._success_hold_counter = 0
        if not self._controller_type_applied:
            self._controller.start_controller(self.config.controller_type).wait()
            self._controller_type_applied = True
        if self.config.update_controller_params and self.config.precision_param:
            self._controller.reconfigure_precision_params(
                self.config.precision_param
            ).wait()
        if self.config.preserve_reference_reset:
            self._run_reference_material_reset()

        obs, info = super().reset(joint_reset=joint_reset, seed=seed, options=options)
        if self.config.lock_tcp_orientation:
            self._marvin_state = self._controller.get_state().wait()[0]
            self._locked_quat = self._normalize_quat(self._marvin_state.tcp_pose[3:])
        if self.config.spacemouse_open_loop:
            self._cmd_pose = self._marvin_state.tcp_pose.copy()
        return obs, info

    def go_to_rest(self, joint_reset=False):
        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)
            return


class MaterialBinaryRewardClassifierWrapper(gym.Wrapper):
    """Apply the reference binary reward classifier to material observations."""

    def __init__(
        self,
        env: gym.Env,
        checkpoint_path: str | None,
        image_keys: list[str],
        threshold: float = 0.8,
        debug: bool = True,
    ):
        super().__init__(env)
        try:
            import jax
            import jax.numpy as jnp
            from serl_launcher.networks.reward_classifier import load_classifier_func
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Material reward classifier requires optional tianji_hilserl "
                "dependencies (jax and serl_launcher). Disable it with "
                "env.eval.override_cfg.classifier.enabled=False or install the "
                "reference classifier stack."
            ) from exc

        if checkpoint_path is None:
            checkpoint_path = str(
                Path(
                    "/home/standard/workspaces/gitlab/tianji_hilserl/src/"
                    "marvin_rl_package/examples/classifier_ckpt"
                )
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Material classifier checkpoint not found: {checkpoint_path}"
            )

        self._jnp = jnp
        self._threshold = float(threshold)
        self._debug = bool(debug)
        self._step = 0
        self._classifier_image_keys = list(image_keys)
        sample = self._classifier_obs(env.observation_space.sample())
        self._classifier = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=sample,
            image_keys=image_keys,
            checkpoint_path=checkpoint_path,
        )

    @staticmethod
    def _state_vector(obs: dict[str, Any]) -> np.ndarray:
        state = obs["state"]
        parts = []
        for key in ("tcp_pose", "tcp_vel", "joint_pos"):
            if key in state:
                parts.append(np.asarray(state[key], dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros((1, 0), dtype=np.float32)
        return np.concatenate(parts, axis=0)[None]

    def _classifier_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self._state_vector(obs)}
        frames = obs.get("frames", {})
        for key in self._classifier_image_keys:
            if key not in frames:
                raise KeyError(
                    f"Material classifier image key {key!r} not in frames "
                    f"{list(frames)}."
                )
            out[key] = np.asarray(frames[key], dtype=np.uint8)[None]
        return out

    def _reward(self, obs: dict[str, Any]) -> int:
        def sigmoid(x):
            return 1 / (1 + self._jnp.exp(-x))

        score = float(
            self._jnp.squeeze(sigmoid(self._classifier(self._classifier_obs(obs))))
        )
        succeed = int(score > self._threshold)
        obs["_reward_classifier_info"] = {
            "score": score,
            "threshold": self._threshold,
        }
        self._step += 1
        if self._debug:
            print(
                f"[material_reward_cls] step={self._step} "
                f"score={score:.3f} succeed={succeed}"
            )
        return succeed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._reward(obs)
        terminated = bool(reward)
        info["succeed"] = bool(reward)
        return obs, reward, terminated, truncated, info


class MaterialTcpTranslationCommandRotateWrapper(gym.ActionWrapper):
    """Rotate TCP-frame translation before the common RelativeFrame wrapper."""

    def __init__(self, env: gym.Env, rotation_matrix: np.ndarray):
        super().__init__(env)
        self._rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(
            3, 3
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        out = np.asarray(action, dtype=np.float64).copy()
        out[:3] = self._rotation_matrix @ out[:3]
        return out

    def step(self, action):
        return self.env.step(self.action(action))

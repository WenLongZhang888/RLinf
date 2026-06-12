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

import copy
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import cycle
from typing import Any, Callable, Optional, Union

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import (
    MarvinHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from .marvin_robot_state import MarvinRobotState
from .utils import construct_adjoint_matrix, construct_homogeneous_matrix, quat_slerp


@dataclass
class MarvinRobotConfig:
    robot_ip: Optional[str] = None
    camera_serials: Optional[Union[list[str], dict[str, str]]] = None
    enable_camera_player: bool = True
    task_description: str = ""
    end_effector_type: str = "marvin_gripper"

    is_dummy: bool = False
    lock_orientation: bool = False
    spacemouse_open_loop: bool = False
    controller_type: str = "impedance"
    update_controller_params: bool = True
    gripper_sleep: float = 0.6
    random_z_range: float = 0.0
    random_rx_range: float = 0.0
    random_ry_range: float = 0.0
    load_param: Optional[dict] = None
    image_crop: Optional[dict[str, Callable[[np.ndarray], np.ndarray]]] = None
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Max number of steps per second

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.9, -0, 2, 0]
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False

    random_xy_range: float = 0.0
    random_rz_range: float = 0.0  # np.pi / 6

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # For Marvin, prefer 7D arrays:
    # - joint impedance: {"impedance_type": 1, "stiffness": [...], "damping": [...]}
    # - cartesian impedance: {"impedance_type": 2, "stiffness": [...], "damping": [...]}
    # `joint_k` / `joint_d` and `cart_k` / `cart_d` are also accepted aliases.
    compliance_param: dict[str, Any] = field(default_factory=dict)
    precision_param: dict[str, Any] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000  # Number of resets before resetting joints
    success_hold_steps: int = (
        1  # Default to 1 to maintain backward compatibility (immediate success)
    )
    # Connection/behavior config for a dexterous-hand end-effector (e.g. Revo2).
    # Keys mirror ``Revo2Hand.__init__`` (side, port, baudrate, slave_id, speed,
    # mode, thumb_opposition, ...). Ignored for the default ``marvin_gripper``.
    revo2_hand_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize Hydra/YAML list values to runtime-friendly types."""
        if isinstance(self.camera_serials, dict):
            self.camera_serials = {
                str(name): str(serial)
                for name, serial in self.camera_serials.items()
            }
        elif self.camera_serials is not None:
            self.camera_serials = [str(serial) for serial in self.camera_serials]
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64)
        self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64)
        self.reward_threshold = np.asarray(self.reward_threshold, dtype=np.float64)
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)
        self.ee_pose_limit_min = np.asarray(
            self.ee_pose_limit_min, dtype=np.float64
        )
        self.ee_pose_limit_max = np.asarray(
            self.ee_pose_limit_max, dtype=np.float64
        )


class MarvinEnv(gym.Env):
    """Marvin robot arm environment."""

    def __init__(
        self,
        config: MarvinRobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[MarvinHWInfo],
        env_idx: int,
    ):
        self._logger = get_logger()
        self.config = config
        self._task_description = config.task_description
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._marvin_state = MarvinRobotState()
        if not self.config.is_dummy:
            self._reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:].copy()).as_quat(),
                ]
            ).copy()
        else:
            self._reset_pose = np.zeros(7)
        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)  # Initialize the cycle

        self._success_hold_counter = 0  # Initialize the success hold counter
        self._locked_quat = None  # For lock_orientation: fixed orientation in step
        self._cmd_pose = None  # For spacemouse_open_loop: integrated command pose
        self._controller_type_applied = False  # First reset applies controller_type
        self._recording_frames: list[dict[str, np.ndarray]] = []  # Video recording

        if not self.config.is_dummy:
            self._setup_hardware()

        # Build the end-effector (gripper or dexterous hand) before sizing the
        # action space, which depends on its action_dim.
        self._end_effector = self._build_end_effector()
        if not self.config.is_dummy:
            self._end_effector.initialize()

        # Init action and observation spaces
        cam_serials = self.config.camera_serials
        if cam_serials is None:
            self.config.camera_serials = []
            cam_serials = self.config.camera_serials
        assert self.config.is_dummy or len(cam_serials) > 0, (
            "At least one camera serial must be provided for MarvinEnv."
        )
        if len(cam_serials) == 0:
            self._logger.info(
                "No camera serials configured for dummy MarvinEnv. "
                "Observations will contain an empty frames dict."
            )
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the robot to be ready
        start_time = time.time()
        while not self._controller.is_robot_up().wait()[0]:
            time.sleep(0.5)
            if time.time() - start_time > 30:
                self._logger.warning(
                    f"Waited {time.time() - start_time} seconds for Marvin robot to be ready."
                )

        self._interpolate_move(self._reset_pose)
        time.sleep(1.0)
        self._marvin_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._open_cameras()
        # Video player for displaying camera frames
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    def _setup_hardware(self):
        from .marvin_controller import MarvinController

        assert self.env_idx >= 0, "env_idx must be set for MarvinEnv."

        # Setup Marvin IP and camera serials
        assert isinstance(self.hardware_info, MarvinHWInfo), (
            f"hardware_info must be MarvinHWInfo, but got {type(self.hardware_info)}."
        )
        # Only set robot_ip and camera_serials if they are not provided in config
        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials

        # Launch Marvin controller
        self._controller = MarvinController.launch_controller(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
        )

    # Keys accepted by ``Revo2Hand.__init__`` (driver-scoped). Teleop-only keys
    # such as ``mode`` / ``thumb_opposition`` live in the teleop config instead.
    _REVO2_HAND_DRIVER_KEYS = frozenset(
        {
            "side",
            "port",
            "baudrate",
            "slave_id",
            "speed",
            "release_on_close",
            "default_state",
            "command_timeout",
            "effective_eps",
        }
    )

    def _build_end_effector(self):
        """Construct the configured end-effector.

        Construction must not touch hardware (so it works in dummy mode and
        before ``initialize()``); ``action_dim`` is always statically known.
        """
        from rlinf.envs.realworld.common.end_effectors.base import EndEffectorType

        ee_type = EndEffectorType(str(self.config.end_effector_type))

        if ee_type == EndEffectorType.MARVIN_GRIPPER:
            from .end_effectors.marvin_gripper import MarvinGripper

            return MarvinGripper(
                gripper_fn=self._gripper_action,
                state_getter=lambda: self._marvin_state,
                action_scale=lambda: float(self.config.action_scale[2]),
            )

        if ee_type == EndEffectorType.REVO2_HAND:
            from rlinf.envs.realworld.common.hand.revo2_hand import Revo2Hand

            hand_cfg = {
                k: v
                for k, v in dict(self.config.revo2_hand_config).items()
                if k in self._REVO2_HAND_DRIVER_KEYS
            }
            return Revo2Hand(**hand_cfg)

        if ee_type == EndEffectorType.RUIYAN_HAND:
            from rlinf.envs.realworld.common.hand.ruiyan_hand import RuiyanHand

            ruiyan_cfg = dict(getattr(self.config, "ruiyan_hand_config", {}) or {})
            return RuiyanHand(**ruiyan_cfg)

        raise ValueError(
            f"Unsupported end_effector_type for Marvin: "
            f"{self.config.end_effector_type!r}"
        )

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        action (np.ndarray): The action to take, which is a 7D vector representing the desired end-effector position and orientation,
        as well as the gripper action. The first 3 elements correspond to the delta in x, y, z position, the next 3 elements correspond to the delta in rx, ry, rz orientation (in euler angles), and the last element corresponds to the gripper action.
        [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]
        """
        start_time = time.time()

        # if self.use_rel_frame:
        #     action = self.transform_action_ee_to_base(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        if (
            self.config.spacemouse_open_loop
            and self._cmd_pose is not None
            and not self.config.is_dummy
        ):
            self.next_position = self._cmd_pose.copy()
        else:
            self.next_position = self._marvin_state.tcp_pose.copy()
        self.next_position[:3] = (
            self.next_position[:3] + xyz_delta * self.config.action_scale[0]
        )

        if not self.config.is_dummy:
            if self.config.lock_orientation and self._locked_quat is not None:
                self.next_position[3:] = self._locked_quat
            else:
                self.next_position[3:] = (
                    R.from_euler(
                        "xyz", action[3:6] * self.config.action_scale[1]
                    )
                    * R.from_quat(self._marvin_state.tcp_pose[3:].copy())
                ).as_quat()

            # Route the end-effector portion of the action (action[6:]) to the
            # active end-effector. For the binary gripper this delegates to
            # ``_gripper_action`` (scaling applied inside MarvinGripper); for a
            # dexterous hand it sends the normalized finger targets.
            is_gripper_action_effective = self._end_effector.command(action[6:])
            sent_pose = self._clip_position_to_safety_box(self.next_position)
            self._move_action(sent_pose)
            if self.config.spacemouse_open_loop:
                self._cmd_pose = sent_pose.copy()
        else:
            is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._marvin_state = self._controller.get_state().wait()[0]
        else:
            self._marvin_state = self._marvin_state
        observation = self._get_observation()

        # Calculate reward and update the internal hold counter
        reward = self._calc_step_reward(observation, is_gripper_action_effective)

        # Logic to determine termination
        # The episode is done only if the robot has reached the target (reward == 1.0)
        # AND has held the position for the required number of steps.
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )

        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def get_tcp_pose(self) -> np.ndarray:
        """Return the current TCP pose ``[x, y, z, qx, qy, qz, qw]``."""
        if not self.config.is_dummy:
            self._marvin_state = self._controller.get_state().wait()[0]
        return self._marvin_state.tcp_pose

    def get_action_scale(self) -> np.ndarray:
        """Return the action scale ``[pos_scale, ori_scale, gripper_scale]``."""
        return self.config.action_scale

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | MarvinRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
            is_gripper_action_effective (bool): Whether the gripper action was effective (i.e., the gripper state changed).
        """
        if not self.config.is_dummy:
            # Convert orientation to euler angles
            euler_angles = np.abs(
                R.from_quat(self._marvin_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._marvin_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            # Check if current state meets the success threshold
            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                # Increment hold counter if in target zone
                self._success_hold_counter += 1
                reward = 1.0
            else:
                # Reset counter if robot leaves the target zone
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._success_hold_counter = 0  # Reset hold counter at the start of the episode

        # Apply controller_type on first reset
        if not self._controller_type_applied:
            self._controller.start_controller(self.config.controller_type).wait()
            self._controller_type_applied = True

        if self.config.update_controller_params:
            self._controller.reconfigure_compliance_params(
                self.config.compliance_param
            ).wait()

        # Save video from previous episode if recording was enabled
        if self._recording_frames:
            self._save_video_recording()
            self._recording_frames.clear()

        # Reset joint
        joint_reset_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_reset_cycle == 0
        if joint_reset:
            self._logger.info(
                f"Number of resets reached {self.config.joint_reset_cycle}, resetting joints to initial position."
            )

        self.go_to_rest(joint_reset)

        self._clear_error()
        self._num_steps = 0
        self._marvin_state = self._controller.get_state().wait()[0]

        if self.config.lock_orientation:
            self._locked_quat = self._marvin_state.tcp_pose[3:].copy()
        if self.config.spacemouse_open_loop:
            self._cmd_pose = self._marvin_state.tcp_pose.copy()

        observation = self._get_observation()

        return observation, {}

    def go_to_rest(self, joint_reset=False):
        if not self.config.is_dummy:
            self._marvin_state = self._controller.get_state().wait()[0]
            self._move_action(self._marvin_state.tcp_pose.copy())
            time.sleep(0.3)
            if self.config.update_controller_params and self.config.precision_param:
                self._controller.reconfigure_precision_params(
                    self.config.precision_param
                ).wait()
            time.sleep(0.5)

        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)

        # Reset arm
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            if self.config.random_z_range != 0:
                reset_pose[2] += np.random.uniform(
                    -self.config.random_z_range, self.config.random_z_range
                )
            euler_random = self.config.reset_ee_pose[3:].copy()
            if self.config.random_rx_range != 0:
                euler_random[0] += np.random.uniform(
                    -self.config.random_rx_range, self.config.random_rx_range
                )
            if self.config.random_ry_range != 0:
                euler_random[1] += np.random.uniform(
                    -self.config.random_ry_range, self.config.random_ry_range
                )
            euler_random[2] += np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range
            )
            reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
        else:
            reset_pose = self._reset_pose.copy()

        if not self.config.is_dummy:
            if self.config.update_controller_params:
                self._controller.reconfigure_compliance_params(
                    self.config.compliance_param
                ).wait()

        self._marvin_state = self._controller.get_state().wait()[0]
        cnt = 0
        while not np.allclose(self._marvin_state.tcp_pose[:3], reset_pose[:3], 0.02):
            cnt += 1
            self._interpolate_move(reset_pose)
            self._marvin_state = self._controller.get_state().wait()[0]
            if cnt > 2:
                break

    def _get_camera_names(self) -> list[str]:
        """Return camera names (keys if camera_serials is dict, else wrist_1, wrist_2, ...)."""
        if isinstance(self.config.camera_serials, dict):
            return list(self.config.camera_serials.keys())
        return [
            f"wrist_{i + 1}"
            for i in range(len(self.config.camera_serials or []))
        ]

    def _get_camera_name_serial_pairs(self) -> list[tuple[str, str]]:
        """Return list of (camera_name, serial_number) for opening cameras."""
        if isinstance(self.config.camera_serials, dict):
            return list(self.config.camera_serials.items())
        return [
            (f"wrist_{i + 1}", serial)
            for i, serial in enumerate(self.config.camera_serials or [])
        ]

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )
        # 6 arm DOFs (xyz delta + rpy delta) + end-effector action dims.
        action_dim = 6 + self._end_effector.action_dim
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )

        camera_names = self._get_camera_names()
        obs_tcp_pose_dim = 7
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(obs_tcp_pose_dim,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "joints_torque": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for name in camera_names
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        if self.config.camera_serials is None:
            return
        camera_infos = [
            CameraInfo(name=name, serial_number=serial)
            for name, serial in self._get_camera_name_serial_pairs()
        ]
        for info in camera_infos:
            camera = create_camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop the frame to the desired resolution. Returns (cropped, resized)."""
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _read_one_camera(
        self, camera: BaseCamera
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Read one camera: (name, cropped, resized, for_obs). Used for parallel get."""
        name = camera._camera_info.name
        reshape_size = self.observation_space["frames"][name].shape[:2][::-1]
        frame = camera.get_frame()
        if self.config.image_crop and name in self.config.image_crop:
            cropped_frame = self.config.image_crop[name](frame)
            resized = cv2.resize(cropped_frame, reshape_size)
        else:
            cropped_frame, resized = self._crop_frame(frame, reshape_size)
        # Observation: RGB (Camera may return BGR from RealSense)
        if resized.shape[2] == 3:
            for_obs = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            for_obs = resized[..., :3].copy()
        return name, cropped_frame.copy(), resized, for_obs

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get frames from all cameras (parallel read). Optionally record for video."""
        frames = {}
        display_frames = {}
        full_res_for_recording: dict[str, np.ndarray] = {}

        def read_one(cam: BaseCamera):
            try:
                return self._read_one_camera(cam)
            except queue.Empty:
                return None, None, None, None

        try:
            with ThreadPoolExecutor(max_workers=len(self._cameras)) as ex:
                futures = [ex.submit(read_one, c) for c in self._cameras]
                for fut in as_completed(futures):
                    name, cropped_frame, resized, for_obs = fut.result()
                    if name is None:
                        raise queue.Empty
                    frames[name] = for_obs
                    display_frames[name] = resized
                    display_frames[f"{name}_full"] = cropped_frame
                    full_res_for_recording[name] = cropped_frame.copy()
        except queue.Empty:
            self._logger.warning(
                "A camera is not producing frames. Wait 5 seconds and try again."
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

    def _save_video_recording(self) -> None:
        """Write recorded frames to mp4 files per camera (source save_video_recording)."""
        if not self._recording_frames:
            return
        out_dir = self.config.save_video_path if self.config.save_video_path else "./videos"
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            self._logger.warning("Could not create video output dir %s", out_dir)
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for camera_key in self._recording_frames[0].keys():
            first_frame = self._recording_frames[0][camera_key]
            height, width = first_frame.shape[:2]
            video_path = f"{out_dir}/{camera_key}_{timestamp}.mp4"
            try:
                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )
                for frame_dict in self._recording_frames:
                    video_writer.write(frame_dict[camera_key])
                video_writer.release()
                self._logger.info("Saved video for camera %s at %s", camera_key, video_path)
            except Exception as e:
                self._logger.warning("Failed to save video for %s: %s", camera_key, e)

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self._rpy_safe_space.low[0],
                self._rpy_safe_space.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self._rpy_safe_space.low[1:], self._rpy_safe_space.high[1:]
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        sleep_time = self.config.gripper_sleep
        if is_binary:
            if (
                position <= -self.config.binary_gripper_threshold
                and self._marvin_state.gripper_open
            ):
                # Close gripper
                self._controller.close_gripper().wait()
                time.sleep(sleep_time)
                return True
            elif (
                position >= self.config.binary_gripper_threshold
                and not self._marvin_state.gripper_open
            ):
                # Open gripper
                self._controller.open_gripper().wait()
                time.sleep(sleep_time)
                return True
            else:  # No change
                return False
        else:
            raise NotImplementedError("Non-binary gripper action not implemented.")

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5):
        num_steps = int(timeout * self.config.step_frequency)
        self._marvin_state: MarvinRobotState = self._controller.get_state().wait()[0]
        pos_path = np.linspace(
            self._marvin_state.tcp_pose[:3], pose[:3], int(num_steps) + 1
        )
        quat_path = quat_slerp(
            self._marvin_state.tcp_pose[3:], pose[3:], int(num_steps) + 1
        )

        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            pose = np.concatenate([pos, quat])
            self._move_action(pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)

        self._marvin_state: MarvinRobotState = self._controller.get_state().wait()[0]

    def _move_action(self, position: np.ndarray):
        if not self.config.is_dummy:
            self._clear_error()
            self._controller.move_arm(position.astype(np.float32)).wait()
        else:
            print(f"Executing dummy action towards {position=}.")

    def _get_observation(self) -> dict:
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "tcp_pose": self._marvin_state.tcp_pose,
                "tcp_vel": self._marvin_state.tcp_vel,
                "gripper_position": np.array(
                    [
                        self._marvin_state.gripper_position,
                    ]
                ),
                "tcp_force": self._marvin_state.tcp_force,
                "tcp_torque": self._marvin_state.tcp_torque,
                "joints_torque": self._marvin_state.joints_torque,
            }
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            obs = copy.deepcopy(self._base_observation_space.sample())
            obs["state"]["joints_torque"] = np.zeros(7, dtype=np.float64)
            return obs

    def transform_obs_base_to_ee(self, state):
        self.adjoint_matrix = construct_adjoint_matrix(self._marvin_state.tcp_pose)
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)

        state["tcp_vel"] = adjoint_inv @ state["tcp_vel"]

        T_b_o = construct_homogeneous_matrix(self._marvin_state.tcp_pose)
        T_r_o = self.T_b_r_inv @ T_b_o

        p_r_o = T_r_o[:3, 3]
        quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
        state["tcp_pose"] = np.concatenate([p_r_o, quat_r_o], axis=0)

        return state

    @property
    def target_ee_pose(self):
        tgt = np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat(),
            ]
        ).copy()
        return tgt

    @property
    def task_description(self):
        return self._task_description

    def close(self):
        """Release the end-effector and cameras."""
        ee = getattr(self, "_end_effector", None)
        if ee is not None:
            try:
                ee.shutdown()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("End-effector shutdown failed: %s", exc)
        if getattr(self, "_cameras", None):
            try:
                self._close_cameras()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Camera shutdown failed: %s", exc)
        return super().close()

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

"""PICO VR teleoperation smoke check for a single Marvin arm.

Example:
    export MARVIN_ROBOT_IP=192.168.71.190
    python toolkits/realworld_check/marvin/test_marvin_vr.py \
        --arm A --controller left --hz 30 --workspace-margin-m 0.08

The controller uses clutch/latch teleoperation:
    left controller: X or left menu toggles clutch
    right controller: A or right menu toggles clutch
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

from rlinf.envs.realworld.marvin import Marvin, XrClient
from rlinf.envs.realworld.marvin.vr_teleop import (
    ArmTeleopController,
    build_local_workspace_limits,
    limit_pose_step,
    pose7_to_transform,
    transform_to_pose7,
    transform_to_pos_rpy,
)


def _is_data() -> bool:
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def _parse_vec3(raw: str) -> np.ndarray:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return np.asarray(vals, dtype=np.float64)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marvin single-arm PICO VR check.")
    parser.add_argument(
        "--ip",
        default=os.environ.get("MARVIN_ROBOT_IP"),
        help="Marvin controller IP, or MARVIN_ROBOT_IP.",
    )
    parser.add_argument(
        "--arm",
        choices=("A", "B"),
        default=os.getenv("MARVIN_ARM_ID", "A").upper(),
    )
    parser.add_argument(
        "--controller",
        choices=("left", "right"),
        default="left",
        help="PICO controller side. A arm usually uses left.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=float(os.getenv("MARVIN_VR_HZ", "30.0")),
    )
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument(
        "--xyz-scale",
        type=_parse_vec3,
        default=np.ones(3, dtype=np.float64),
        help="Per-axis translation scale/mirror, e.g. 1,1,1.",
    )
    parser.add_argument(
        "--track-rotation",
        action="store_true",
        help="Follow controller relative rotation. Default keeps initial TCP rotation.",
    )
    parser.add_argument(
        "--workspace-margin-m",
        type=float,
        default=0.08,
        help="Local workspace half-width around current TCP pose. Use 0 for global limits.",
    )
    parser.add_argument(
        "--max-step-mm",
        type=float,
        default=8.0,
        help="Per-tick target translation limit. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-rot-deg",
        type=float,
        default=6.0,
        help="Per-tick target rotation limit. Use 0 to disable.",
    )
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.6,
        help="Trigger threshold for close/open gripper toggle.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if not args.ip:
        raise ValueError("Set --ip or MARVIN_ROBOT_IP.")
    if args.hz <= 0:
        raise ValueError("--hz must be positive.")

    print(f"Connecting to Marvin at {args.ip} (arm {args.arm})...")
    robot = Marvin(robot_ip=args.ip)
    try:
        start_time = time.time()
        while not robot.is_robot_up(args.arm):
            time.sleep(0.5)
            if time.time() - start_time > 30:
                print(
                    f"Waited {time.time() - start_time:.1f}s for Marvin arm {args.arm}."
                )

        xr = XrClient()
        xr.init()

        home_pose = robot.get_state(args.arm).tcp_pose.copy()
        home_T = robot_pose_to_transform(home_pose)
        workspace_limits = build_local_workspace_limits(
            home_T,
            xy_range=args.workspace_margin_m,
            z_range_low=args.workspace_margin_m,
            z_range_high=args.workspace_margin_m,
        )
        teleop = ArmTeleopController(
            xr,
            side=args.controller,
            workspace_limits=workspace_limits,
            translation_scale=args.translation_scale,
            xyz_scale=args.xyz_scale,
            track_rotation=args.track_rotation,
        )
        last_target_T = home_T.copy()
        last_gripper_closed = False

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        print("\n--- Marvin PICO VR Control ---")
        print(f"Arm {args.arm} controlled by {args.controller} controller.")
        print("Clutch: left X/menu or right A/menu toggles active control.")
        print("Trigger: close gripper above threshold, open below threshold.")
        print("Press h to recenter workspace; q to quit.")
        print("--------------------------------\n")

        try:
            while True:
                loop_start = time.time()

                if _is_data():
                    key = sys.stdin.read(1)
                    if key == "q":
                        print("\nQuitting...")
                        break
                    if key == "h":
                        home_pose = robot.get_state(args.arm).tcp_pose.copy()
                        home_T = robot_pose_to_transform(home_pose)
                        workspace_limits = build_local_workspace_limits(
                            home_T,
                            xy_range=args.workspace_margin_m,
                            z_range_low=args.workspace_margin_m,
                            z_range_high=args.workspace_margin_m,
                        )
                        teleop.set_workspace_limits(workspace_limits)
                        teleop.reset_clutch()
                        last_target_T = home_T.copy()
                        print("Recentered workspace around current TCP pose.")

                state = robot.get_state(args.arm)
                current_T = robot_pose_to_transform(state.tcp_pose)
                cmd = teleop.step(current_T)
                target_T = limit_pose_step(
                    cmd.T_target,
                    last_target_T,
                    max_step_m=max(0.0, args.max_step_mm) * 0.001,
                    max_rot_deg=args.max_rot_deg,
                )

                if cmd.active:
                    try:
                        robot.move_pose(args.arm, transform_to_pose7(target_T))
                        last_target_T = target_T
                    except RuntimeError as exc:
                        print(f"[{args.arm}] IK/move failed: {exc}")

                gripper_closed = cmd.trigger >= args.trigger_threshold
                if gripper_closed != last_gripper_closed:
                    if gripper_closed:
                        robot.close_gripper(args.arm)
                    else:
                        robot.open_gripper(args.arm)
                    last_gripper_closed = gripper_closed

                xyz, rpy = transform_to_pos_rpy(last_target_T)
                print(
                    "\r"
                    f"active={cmd.active} xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
                    f"rpy=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}) "
                    f"trigger={cmd.trigger:.2f}",
                    end="",
                    flush=True,
                )

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, (1.0 / args.hz) - elapsed))
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            xr.close()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.close()
        print("\nDone.")


def robot_pose_to_transform(pose7: np.ndarray) -> np.ndarray:
    """Convert Marvin robot pose to a homogeneous transform."""
    return pose7_to_transform(pose7)


if __name__ == "__main__":
    main()

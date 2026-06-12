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

"""BrainCo Revo2 dexterous-hand smoke check.

Two modes:

  # Per-finger sweep (no VR): open -> close -> open each finger in turn.
  python toolkits/realworld_check/marvin/test_marvin_revo2.py sweep \
      --side right --port /dev/ttyUSB0

  # VR retarget (requires xrobotoolkit_sdk + a PICO controller): stream the
  # controller trigger/grip into Revo2 finger targets.
  python toolkits/realworld_check/marvin/test_marvin_revo2.py vr \
      --side right --controller right --mode gripper --hz 30

``--port`` may be omitted to let the SDK auto-detect the hand.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from rlinf.envs.realworld.common.hand.revo2_driver import (
    Revo2HandConfig,
    Revo2HandDriver,
)
from rlinf.envs.realworld.common.hand.revo2_mapping import (
    FINGER_NAMES,
    HAND_MODES,
    Revo2FingerTargets,
    compute_revo2_targets,
    to_sdk_positions,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Marvin Revo2 hand check.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--side", choices=("left", "right"), default="right")
    common.add_argument("--port", default=None, help="Serial port; omit to auto-detect.")
    common.add_argument("--baudrate", type=int, default=460800)
    common.add_argument("--slave-id", type=lambda x: int(x, 0), default=None)
    common.add_argument("--speed", type=int, default=1000)

    p_sweep = sub.add_parser("sweep", parents=[common], help="Per-finger sweep.")
    p_sweep.add_argument("--dwell", type=float, default=0.6, help="Seconds per pose.")

    p_vr = sub.add_parser("vr", parents=[common], help="VR retarget stream.")
    p_vr.add_argument("--controller", choices=("left", "right"), default="right")
    p_vr.add_argument("--mode", choices=HAND_MODES, default="gripper")
    p_vr.add_argument("--thumb-opposition", type=float, default=0.8)
    p_vr.add_argument("--hz", type=float, default=30.0)
    return parser


def _make_driver(args) -> Revo2HandDriver:
    return Revo2HandDriver(
        Revo2HandConfig(
            side=args.side,
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            speed=args.speed,
        )
    )


async def run_sweep(args) -> None:
    driver = _make_driver(args)
    await driver.connect()
    try:
        print(f"Connected: {driver.label}. Sweeping fingers {FINGER_NAMES}...")
        # Start fully open.
        await driver.set_sdk_positions([0, 0, 0, 0, 0, 0])
        time.sleep(args.dwell)
        for i, name in enumerate(FINGER_NAMES):
            vec = [0.0] * 6
            vec[i] = 1.0
            targets = Revo2FingerTargets.from_normalized_vector(vec)
            pos = to_sdk_positions(targets)
            print(f"  close {name:16s} -> sdk={pos}")
            await driver.set_sdk_positions(pos)
            time.sleep(args.dwell)
            await driver.set_sdk_positions([0, 0, 0, 0, 0, 0])
            time.sleep(args.dwell)
        print("Sweep done.")
    finally:
        await driver.close()


async def run_vr(args) -> None:
    from rlinf.envs.realworld.common.vr import XrClient, pick_controller_inputs

    driver = _make_driver(args)
    xr = XrClient()
    xr.init()
    await driver.connect()
    print(
        f"Connected: {driver.label}. Streaming {args.controller} controller "
        f"(mode={args.mode}). Ctrl-C to stop."
    )
    try:
        while True:
            loop_start = time.time()
            snap = xr.snapshot()
            _, trigger, grip, _ = pick_controller_inputs(snap, args.controller)
            targets = compute_revo2_targets(
                trigger=trigger,
                grip=grip,
                thumb_opposition=args.thumb_opposition,
                mode=args.mode,
            )
            pos = await driver.set_normalized_targets(targets)
            print(
                f"\rtrigger={trigger:.2f} grip={grip:.2f} "
                f"sdk={np.array(pos)}",
                end="",
                flush=True,
            )
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, (1.0 / args.hz) - elapsed))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        await driver.close()
        xr.close()
        print("\nDone.")


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.cmd == "sweep":
        asyncio.run(run_sweep(args))
    elif args.cmd == "vr":
        asyncio.run(run_vr(args))


if __name__ == "__main__":
    main()

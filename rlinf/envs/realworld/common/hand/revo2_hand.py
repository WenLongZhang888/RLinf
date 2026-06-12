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

"""BrainCo Revo2 dexterous hand as a continuous :class:`EndEffector`.

The underlying :class:`Revo2HandDriver` is asynchronous, so this wrapper owns a
dedicated asyncio event-loop thread and submits coroutines with
``run_coroutine_threadsafe`` (latest-wins). The public :meth:`command` interface
stays synchronous so it fits the env step loop.

The 6-D action / state vector is the normalized
``[thumb_bend, thumb_opposition, index, middle, ring, pinky]`` order defined by
:data:`rlinf.envs.realworld.common.hand.revo2_mapping.FINGER_NAMES`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import numpy as np

from rlinf.envs.realworld.common.end_effectors.base import EndEffector
from rlinf.utils.logging import get_logger

from .revo2_driver import Revo2HandConfig, Revo2HandDriver
from .revo2_mapping import FINGER_NAMES, Revo2FingerTargets


class Revo2Hand(EndEffector):
    """BrainCo Revo2 hand — continuous 6-DOF normalized end-effector.

    Args:
        side: ``"left"`` or ``"right"``.
        port: Serial device path. ``None`` triggers SDK auto-detection.
        baudrate: Modbus baudrate (default 460800).
        slave_id: Modbus slave id. ``None`` picks the side default
            (left=0x7E, right=0x7F).
        speed: Per-finger SDK speed (default 1000).
        release_on_close: Open all fingers on :meth:`shutdown`.
        default_state: Optional 6-D normalized reset target (defaults to open).
        command_timeout: Seconds to wait for a submitted command to land. ``0``
            or ``None`` means fire-and-forget (latest-wins, non-blocking).
        effective_eps: Min change in the target vector to count a command as a
            meaningful state change (used by the reward path).
    """

    _NUM_DOFS = 6

    def __init__(
        self,
        side: str = "right",
        port: Optional[str] = None,
        baudrate: int = 460800,
        slave_id: Optional[int] = None,
        speed: int = 1000,
        release_on_close: bool = False,
        default_state: Optional[list[float]] = None,
        command_timeout: Optional[float] = None,
        effective_eps: float = 1e-3,
    ) -> None:
        self._config = Revo2HandConfig(
            side=side,
            port=port,
            baudrate=baudrate,
            slave_id=slave_id,
            speed=speed,
            release_on_close=release_on_close,
        )
        self._driver = Revo2HandDriver(self._config)
        self._logger = get_logger()

        self._default_state = (
            np.zeros(self._NUM_DOFS, dtype=np.float64)
            if default_state is None
            else np.asarray(default_state, dtype=np.float64).reshape(-1)
        )
        if self._default_state.shape[0] != self._NUM_DOFS:
            raise ValueError(
                f"Revo2 default_state must have length {self._NUM_DOFS}, "
                f"got {self._default_state.shape[0]}"
            )
        self._command_timeout = command_timeout
        self._effective_eps = float(effective_eps)

        self._last_target = self._default_state.copy()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def state_dim(self) -> int:
        return self._NUM_DOFS

    @property
    def control_mode(self) -> str:
        return "continuous"

    @property
    def finger_names(self) -> list[str]:
        return list(FINGER_NAMES)

    # ------------------------------------------------------------------
    # Event-loop helpers
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="revo2-hand-loop",
            daemon=True,
        )
        self._loop_thread.start()

    def _submit(self, coro, *, wait: bool):
        """Schedule a coroutine on the loop thread.

        With ``wait=True`` block on the result (used for connect/shutdown).
        With ``wait=False`` return immediately (latest-wins command path).
        """
        if self._loop is None:
            raise RuntimeError("Revo2Hand event loop is not running; call initialize().")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if wait:
            return future.result()
        return future

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Start the event loop and connect to the hand hardware."""
        self._start_loop()
        self._submit(self._driver.connect(), wait=True)
        self._connected = True
        # Move to the default (open) state on connect.
        self._submit(
            self._driver.set_normalized_targets(
                Revo2FingerTargets.from_normalized_vector(self._default_state)
            ),
            wait=True,
        )
        self._last_target = self._default_state.copy()

    def shutdown(self) -> None:
        """Disconnect the hand and stop the event loop."""
        if self._loop is None:
            return
        try:
            if self._connected:
                self._submit(self._driver.close(), wait=True)
        finally:
            self._connected = False
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
            self._loop.close()
            self._loop = None
            self._loop_thread = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return the last commanded normalized target (no hardware readback)."""
        return self._last_target.copy()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def command(self, action: np.ndarray) -> bool:
        """Send a 6-D normalized finger target; return whether it changed."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self._NUM_DOFS:
            raise ValueError(
                f"Revo2 action must have length {self._NUM_DOFS}, got {a.shape[0]}"
            )
        a = np.clip(a, 0.0, 1.0)
        changed = bool(np.linalg.norm(a - self._last_target) > self._effective_eps)
        self._last_target = a.copy()

        if self._loop is None:
            # Not initialized (e.g. dummy mode): record intent only.
            return changed

        targets = Revo2FingerTargets.from_normalized_vector(a)
        future = self._submit(self._driver.set_normalized_targets(targets), wait=False)
        if self._command_timeout:
            try:
                future.result(timeout=self._command_timeout)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Revo2 command timed out/failed: %s", exc)
        return changed

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Move the hand to the default (open) or a specified normalized target."""
        target = (
            self._default_state
            if target_state is None
            else np.asarray(target_state, dtype=np.float64).reshape(-1)
        )
        self.command(target)

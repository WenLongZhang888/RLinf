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

"""Safe wrapper around XRoboToolkit's PICO Python SDK."""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

PoseArray = np.ndarray
logger = logging.getLogger(__name__)


def _identity_pose() -> PoseArray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def identity_pose7() -> PoseArray:
    """Return an identity 7D pose ``[x, y, z, qx, qy, qz, qw]``."""
    return _identity_pose()


@dataclass
class ControllerSnapshot:
    """One PICO headset/controller sample.

    Poses use ``[x, y, z, qx, qy, qz, qw]`` in the raw SDK frame.
    """

    headset_pose: PoseArray
    right_pose: PoseArray
    right_trigger: float
    right_grip: float
    button_a: bool
    button_menu: bool
    timestamp: float
    left_pose: PoseArray = field(default_factory=_identity_pose)
    left_trigger: float = 0.0
    left_grip: float = 0.0
    button_x: bool = False
    left_button_menu: bool = False


class XrClient:
    """Small, optional-dependency client for ``xrobotoolkit_sdk``."""

    def __init__(
        self,
        sdk_module: Any | None = None,
        *,
        dummy_on_missing: bool = False,
    ) -> None:
        self._dummy_on_missing = dummy_on_missing
        self._initialized = False
        self._lock = threading.Lock()
        self._xrt = sdk_module

    def init(self) -> None:
        """Initialize XRoboToolkit once."""
        with self._lock:
            if self._initialized:
                return
            if self._xrt is None:
                try:
                    import xrobotoolkit_sdk as xrt  # type: ignore[import-not-found]
                except ImportError as exc:
                    if self._dummy_on_missing:
                        logger.warning(
                            "xrobotoolkit_sdk is not installed; XrClient runs in dummy mode."
                        )
                        self._initialized = True
                        return
                    raise ImportError(
                        "VR teleoperation requires optional dependency "
                        "'xrobotoolkit_sdk'. Install the Marvin VR target or add the "
                        "XRoboToolkit PC Service Python binding to the environment."
                    ) from exc
                self._xrt = xrt

            init_fn = getattr(self._xrt, "init", None)
            if callable(init_fn):
                init_fn()
            self._initialized = True

    def close(self) -> None:
        """Release SDK resources when supported."""
        with self._lock:
            if not self._initialized or self._xrt is None:
                self._initialized = False
                return
            close_fn = getattr(self._xrt, "close", None) or getattr(
                self._xrt, "shutdown", None
            )
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.warning("Failed to close xrobotoolkit_sdk: %s", exc)
            self._initialized = False

    def _safe_call(self, getter: Callable[[], Any] | None, default: Any) -> Any:
        if not self._initialized:
            self.init()
        if self._xrt is None or getter is None:
            return default
        try:
            value = getter()
        except Exception as exc:
            logger.debug("XR SDK call failed: %s", exc)
            return default
        return default if value is None else value

    def _getter(self, *names: str) -> Callable[[], Any] | None:
        if self._xrt is None:
            return None
        for name in names:
            getter = getattr(self._xrt, name, None)
            if callable(getter):
                return getter
        return None

    def _safe_pose(self, *names: str) -> PoseArray:
        raw = self._safe_call(self._getter(*names), None)
        if raw is None:
            return _identity_pose()
        try:
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        except Exception:
            return _identity_pose()
        if arr.size != 7:
            return _identity_pose()
        q_norm = float(np.linalg.norm(arr[3:7]))
        if not np.isfinite(q_norm) or q_norm < 1e-6:
            safe = arr.copy()
            safe[3:7] = _identity_pose()[3:7]
            return safe
        safe = arr.copy()
        safe[3:7] /= q_norm
        return safe

    def _safe_float(self, *names: str) -> float:
        value = self._safe_call(self._getter(*names), 0.0)
        try:
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            return 0.0

    def _safe_bool(self, *names: str) -> bool:
        return bool(self._safe_call(self._getter(*names), False))

    def snapshot(self) -> ControllerSnapshot:
        """Read headset, both controllers, triggers, grips, and clutch buttons."""
        return ControllerSnapshot(
            headset_pose=self._safe_pose("get_headset_pose"),
            right_pose=self._safe_pose("get_right_controller_pose"),
            right_trigger=self._safe_float("get_right_trigger"),
            right_grip=self._safe_float("get_right_grip"),
            button_a=self._safe_bool(
                "get_A_button", "get_right_button_a", "get_right_a_button"
            ),
            button_menu=self._safe_bool("get_right_menu_button", "get_menu_button"),
            timestamp=time.time(),
            left_pose=self._safe_pose("get_left_controller_pose"),
            left_trigger=self._safe_float("get_left_trigger"),
            left_grip=self._safe_float("get_left_grip"),
            button_x=self._safe_bool(
                "get_X_button", "get_left_button_x", "get_left_x_button"
            ),
            left_button_menu=self._safe_bool("get_left_menu_button"),
        )

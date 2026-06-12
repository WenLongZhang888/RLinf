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

import threading
from typing import Optional

import numpy as np


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(
        self,
        device_index: int = 0,
        axis_mapping: Optional[list[int]] = None,
        axis_remap: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        try:
            import pyspacemouse
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SpaceMouse teleoperation requires optional dependency "
                "'pyspacemouse'. Install the matching real-world teleop extra "
                "or run requirements/install.sh with a spacemouse target."
            ) from exc

        self._device = pyspacemouse.open(device_index=device_index)
        if self._device is None:
            raise RuntimeError(
                f"Failed to open SpaceMouse device at index {device_index}. "
                "Check that the device is connected and readable by this user."
            )
        self._axis_mapping = axis_mapping
        self._axis_remap = axis_remap

        self.state_lock = threading.Lock()
        self.latest_data: dict = {"action": np.zeros(6), "buttons": [0, 0]}
        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _remap_action(self, action: np.ndarray) -> np.ndarray:
        if self._axis_mapping is not None:
            action = action[np.asarray(self._axis_mapping, dtype=np.int64)]
        if self._axis_remap is not None:
            action = np.asarray(
                [sign * action[src_index] for src_index, sign in self._axis_remap],
                dtype=np.float64,
            )
        return action

    def _read_spacemouse(self) -> None:
        while True:
            state = self._device.read()
            with self.state_lock:
                action = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["action"] = self._remap_action(action)
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


if __name__ == "__main__":
    import time

    def test_spacemouse():
        """Test the SpaceMouseExpert class.

        This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
        The user is expected to move the spacemouse and press its buttons while the test is running.
        It keeps running until the user stops it.

        """
        spacemouse = SpaceMouseExpert()
        with np.printoptions(precision=3, suppress=True):
            while True:
                action, buttons = spacemouse.get_action()
                print(f"Spacemouse action: {action}, buttons: {buttons}")
                time.sleep(0.1)

    test_spacemouse()

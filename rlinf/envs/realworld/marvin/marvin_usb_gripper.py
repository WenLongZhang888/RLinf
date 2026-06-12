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

"""Direct USB serial gripper control for Marvin setups."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from rlinf.utils.logging import get_logger


def _arm_env_value(prefix: str, arm: str) -> str | None:
    return os.getenv(f"{prefix}_{arm}") or os.getenv(prefix)


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_hex_bytes(raw: str | None) -> bytes | None:
    if not raw:
        return None
    normalized = raw.replace("0x", " ").replace(",", " ")
    tokens = normalized.split()
    if not tokens:
        return None
    try:
        return bytes.fromhex(" ".join(tokens))
    except ValueError as exc:
        raise ValueError(f"Invalid hex payload: {raw}") from exc


@dataclass
class MarvinUsbGripperConfig:
    """Configuration for a gripper connected through a local USB serial port."""

    port: str
    baudrate: int
    open_command: bytes
    close_command: bytes
    init_command: bytes | None
    timeout: float
    settle_time: float

    @classmethod
    def from_env(cls, arm: str) -> "MarvinUsbGripperConfig":
        port = _arm_env_value("MARVIN_GRIPPER_SERIAL_PORT", arm)
        if not port:
            raise ValueError(
                "MARVIN_GRIPPER_SERIAL_PORT or "
                f"MARVIN_GRIPPER_SERIAL_PORT_{arm} must be set for USB gripper "
                "control."
            )

        open_command = _parse_hex_bytes(
            _arm_env_value("MARVIN_GRIPPER_OPEN_HEX", arm)
        )
        close_command = _parse_hex_bytes(
            _arm_env_value("MARVIN_GRIPPER_CLOSE_HEX", arm)
        )
        if open_command is None or close_command is None:
            raise ValueError(
                "MARVIN_GRIPPER_OPEN_HEX and MARVIN_GRIPPER_CLOSE_HEX "
                f"(optionally suffixed with _{arm}) must be set."
            )

        baudrate_raw = _arm_env_value("MARVIN_GRIPPER_BAUDRATE", arm) or "115200"
        try:
            baudrate = int(baudrate_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid MARVIN_GRIPPER_BAUDRATE: {baudrate_raw}") from exc

        return cls(
            port=port,
            baudrate=baudrate,
            open_command=open_command,
            close_command=close_command,
            init_command=_parse_hex_bytes(
                _arm_env_value("MARVIN_GRIPPER_INIT_HEX", arm)
            ),
            timeout=_parse_float_env("MARVIN_GRIPPER_SERIAL_TIMEOUT", 1.0),
            settle_time=_parse_float_env("MARVIN_GRIPPER_SETTLE_TIME", 0.3),
        )


class MarvinUsbGripper:
    """Send complete Modbus RTU frames to a USB-RS485 gripper adapter."""

    def __init__(self, config: MarvinUsbGripperConfig):
        self._logger = get_logger()
        self._config = config
        self._serial = self._open_serial(config)
        self._is_open = False
        if config.init_command is not None:
            self._send(config.init_command)

    @classmethod
    def from_env(cls, arm: str) -> "MarvinUsbGripper":
        """Create a USB gripper using MARVIN_GRIPPER_* environment variables."""
        return cls(MarvinUsbGripperConfig.from_env(arm))

    @staticmethod
    def _open_serial(config: MarvinUsbGripperConfig) -> Any:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for MARVIN_GRIPPER_BACKEND=usb. "
                "Install the Marvin extra or install pyserial in this environment."
            ) from exc

        return serial.Serial(
            port=config.port,
            baudrate=config.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=config.timeout,
            write_timeout=config.timeout,
        )

    def _send(self, payload: bytes) -> bool:
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        written = self._serial.write(payload)
        self._serial.flush()
        if written != len(payload):
            self._logger.warning(
                "Short write to Marvin USB gripper on %s: wrote %s/%s bytes.",
                self._config.port,
                written,
                len(payload),
            )
            return False
        time.sleep(self._config.settle_time)
        return True

    def open(self) -> bool:
        """Open the gripper."""
        ok = self._send(self._config.open_command)
        if ok:
            self._is_open = True
        return ok

    def close(self) -> bool:
        """Close the gripper."""
        ok = self._send(self._config.close_command)
        if ok:
            self._is_open = False
        return ok

    @property
    def is_open(self) -> bool:
        """Return the last commanded open/close state."""
        return self._is_open

    def cleanup(self) -> None:
        """Close the serial port."""
        self._serial.close()

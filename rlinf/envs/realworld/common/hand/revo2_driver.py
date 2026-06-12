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

"""BrainCo Revo2 hand driver primitives.

This module keeps the Revo2 hardware details out of higher-level code:
auto-detect/open Modbus, switch to normalized finger units, send finger
targets, and close the serial connection. The optional ``bc_stark_sdk``
dependency is imported lazily so importing this module never requires the
hardware SDK to be installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from .revo2_mapping import Revo2FingerTargets, to_sdk_positions

logger = logging.getLogger(__name__)


DEFAULT_REVO2_SPEED = 1000
DEFAULT_REVO2_BAUDRATE = 460800
DEFAULT_REVO2_LEFT_SLAVE_ID = 0x7E
DEFAULT_REVO2_RIGHT_SLAVE_ID = 0x7F


@dataclass
class Revo2HandConfig:
    """Connection/configuration for a BrainCo Revo2 hand."""

    side: str = "left"
    port: str | None = None
    baudrate: int | None = None
    slave_id: int | None = None
    speed: int = DEFAULT_REVO2_SPEED
    release_on_close: bool = False


class Revo2HandDriver:
    """Async Revo2 hand driver.

    The driver accepts normalized :class:`Revo2FingerTargets` and hides the
    SDK-specific Modbus client, slave id, unit mode, and speed vector.
    """

    def __init__(self, config: Revo2HandConfig | None = None) -> None:
        self.config = config or Revo2HandConfig()
        if self.config.side not in {"left", "right"}:
            raise ValueError("Revo2HandConfig.side must be 'left' or 'right'")
        self.client: Any | None = None
        self.slave_id: int | None = None
        self.device_info: Any | None = None
        self._libstark: Any | None = None
        self._speeds = [int(self.config.speed)] * 6

    @property
    def connected(self) -> bool:
        return self.client is not None and self.slave_id is not None

    @property
    def side(self) -> str:
        return self.config.side

    @property
    def label(self) -> str:
        return f"{self.config.side} Revo2"

    async def connect(self) -> None:
        """Open the Revo2 Modbus connection and switch to normalized units."""
        if self.connected:
            raise RuntimeError("Revo2HandDriver is already connected")

        libstark = self._import_libstark()
        self._libstark = libstark

        if self.config.port is None:
            logger.info("Auto-detecting %s device...", self.label)
            protocol, port_name, baudrate, slave_id = (
                await libstark.auto_detect_modbus_revo2(None, True)
            )
            if protocol != libstark.StarkProtocolType.Modbus:
                raise RuntimeError(f"Unsupported Revo2 protocol: {protocol}")
        else:
            port_name = self.config.port
            baudrate = self._coerce_baudrate(
                libstark,
                self.config.baudrate or DEFAULT_REVO2_BAUDRATE,
            )
            slave_id = self.config.slave_id or self._default_slave_id_for_side(
                self.config.side
            )

        logger.info(
            "Opening %s: port=%s baudrate=%s slave_id=0x%x",
            self.label,
            port_name,
            baudrate,
            slave_id,
        )
        self.client = await libstark.modbus_open(port_name, baudrate)
        self.slave_id = int(slave_id)

        self.device_info = await self.client.get_device_info(self.slave_id)
        logger.info(
            "%s device info: %s",
            self.label,
            getattr(self.device_info, "description", self.device_info),
        )

        await self.client.set_finger_unit_mode(
            self.slave_id,
            libstark.FingerUnitMode.Normalized,
        )

    async def set_normalized_targets(self, targets: Revo2FingerTargets) -> list[int]:
        """Send normalized finger targets and return SDK positions for logging."""
        positions = to_sdk_positions(targets)
        await self.set_sdk_positions(positions)
        return positions

    async def set_sdk_positions(
        self,
        positions: Sequence[int],
        speeds: Sequence[int] | None = None,
    ) -> None:
        """Send SDK-space finger positions.

        Positions are in Revo2 normalized-unit integer space. The expected slot
        order is ``[Thumb Flex, Thumb Aux, Index, Middle, Ring, Pinky]``.
        """
        if not self.connected or self.client is None or self.slave_id is None:
            raise RuntimeError("Revo2HandDriver is not connected")
        pos = [int(v) for v in positions]
        if len(pos) != 6:
            raise ValueError(f"Revo2 positions must have length 6, got {len(pos)}")
        spd = [int(v) for v in (speeds or self._speeds)]
        if len(spd) != 6:
            raise ValueError(f"Revo2 speeds must have length 6, got {len(spd)}")
        await self.client.set_finger_positions_and_speeds(self.slave_id, pos, spd)

    async def release(self) -> None:
        """Open all fingers."""
        await self.set_sdk_positions([0, 0, 0, 0, 0, 0])

    async def close(self) -> None:
        """Close the Revo2 Modbus connection."""
        if self.client is None:
            return

        client = self.client
        libstark = self._libstark
        try:
            if self.config.release_on_close and self.connected:
                await self.release()
        finally:
            try:
                if libstark is not None:
                    libstark.modbus_close(client)
            finally:
                self.client = None
                self.slave_id = None
                logger.info("%s disconnected.", self.label)

    @staticmethod
    def _import_libstark() -> Any:
        try:
            from bc_stark_sdk import main_mod as libstark  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "bc-stark-sdk is not installed: "
                f"{exc}. Install the Revo2 hand dependency, e.g. "
                "`bash requirements/install.sh embodied --env marvin_vr_hand` "
                "or `pip install bc_stark_sdk colorlog`."
            ) from exc
        return libstark

    @staticmethod
    def _default_slave_id_for_side(side: str) -> int:
        return (
            DEFAULT_REVO2_LEFT_SLAVE_ID
            if side == "left"
            else DEFAULT_REVO2_RIGHT_SLAVE_ID
        )

    @staticmethod
    def _coerce_baudrate(libstark: Any, baudrate: Any) -> Any:
        """Convert a plain integer baudrate to the SDK Baudrate enum."""
        if not isinstance(baudrate, int):
            return baudrate
        baudrate_cls = libstark.Baudrate
        if hasattr(baudrate_cls, "from_int"):
            try:
                return baudrate_cls.from_int(baudrate)
            except Exception:  # noqa: BLE001
                pass
        baudrate_map = {
            19200: baudrate_cls.Baud19200,
            57600: baudrate_cls.Baud57600,
            115200: baudrate_cls.Baud115200,
            460800: baudrate_cls.Baud460800,
            1000000: baudrate_cls.Baud1Mbps,
            2000000: baudrate_cls.Baud2Mbps,
            5000000: baudrate_cls.Baud5Mbps,
        }
        try:
            return baudrate_map[baudrate]
        except KeyError as exc:
            supported = ", ".join(str(k) for k in sorted(baudrate_map))
            raise ValueError(
                f"Unsupported Revo2 baudrate={baudrate}, supported: {supported}"
            ) from exc


__all__ = [
    "DEFAULT_REVO2_SPEED",
    "DEFAULT_REVO2_BAUDRATE",
    "DEFAULT_REVO2_LEFT_SLAVE_ID",
    "DEFAULT_REVO2_RIGHT_SLAVE_ID",
    "Revo2HandConfig",
    "Revo2HandDriver",
]

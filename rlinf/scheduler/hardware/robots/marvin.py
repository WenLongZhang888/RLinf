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

import ipaddress
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class MarvinHWInfo(HardwareInfo):
    """Hardware information for a Marvin robotic system."""

    config: "MarvinConfig"


@Hardware.register()
class MarvinRobot(Hardware):
    """Hardware policy for Marvin robotic systems."""

    HW_TYPE = "Marvin"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["MarvinConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the Marvin robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Hardware resource descriptor, or ``None`` when the node has no
            matching Marvin configuration.
        """
        assert configs is not None, (
            "Marvin hardware requires explicit configurations for robot IP "
            "and camera serials."
        )
        robot_configs: list[MarvinConfig] = []
        for config in configs:
            if isinstance(config, MarvinConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        infos = [
            MarvinHWInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=config)
            for config in robot_configs
        ]
        return HardwareResource(type=cls.HW_TYPE, infos=infos)


@NodeHardwareConfig.register_hardware_config(MarvinRobot.HW_TYPE)
@dataclass
class MarvinConfig(HardwareConfig):
    """Configuration for a Marvin robotic system."""

    robot_ip: str
    """IP address of the Marvin robotic system."""

    camera_serials: Optional[list[str]] = None
    """Camera serial numbers associated with the robot."""

    disable_validate: bool = False
    """Reserved for parity with other real-world robot configs."""

    def __post_init__(self):
        """Validate user-facing hardware configuration."""
        super().__post_init__()
        try:
            ipaddress.ip_address(self.robot_ip)
        except ValueError as exc:
            raise ValueError(
                "'robot_ip' in Marvin config must be a valid IP address. "
                f"Got {self.robot_ip!r}."
            ) from exc
        if self.camera_serials is not None:
            self.camera_serials = [str(serial) for serial in self.camera_serials]

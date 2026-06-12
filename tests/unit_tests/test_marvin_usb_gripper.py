"""Unit tests for the Marvin USB serial gripper backend."""

from __future__ import annotations

import sys
import types
import importlib.util
import logging
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_usb_gripper_module():
    _ensure_package("rlinf", REPO_ROOT / "rlinf")
    _ensure_package("rlinf.envs", REPO_ROOT / "rlinf" / "envs")
    _ensure_package("rlinf.envs.realworld", REPO_ROOT / "rlinf" / "envs" / "realworld")
    _ensure_package(
        "rlinf.envs.realworld.marvin",
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin",
    )
    _ensure_package("rlinf.utils", REPO_ROOT / "rlinf" / "utils")

    logging_module = types.ModuleType("rlinf.utils.logging")
    logger = logging.getLogger("marvin-usb-gripper-test")
    logging_module.get_logger = lambda: logger
    sys.modules["rlinf.utils.logging"] = logging_module

    module_name = "rlinf.envs.realworld.marvin.marvin_usb_gripper"
    module_path = (
        REPO_ROOT / "rlinf" / "envs" / "realworld" / "marvin" / "marvin_usb_gripper.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSerial:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.writes = []
        self.closed = False
        _FakeSerial.instances.append(self)

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.closed = True


def test_usb_gripper_sends_configured_rtu_frames(monkeypatch):
    marvin_usb_gripper = _load_usb_gripper_module()
    serial_module = types.ModuleType("serial")
    serial_module.Serial = _FakeSerial
    monkeypatch.setitem(sys.modules, "serial", serial_module)
    monkeypatch.setattr(marvin_usb_gripper.time, "sleep", lambda _: None)
    monkeypatch.setenv("MARVIN_GRIPPER_SERIAL_PORT_B", "/dev/ttyUSB7")
    monkeypatch.setenv("MARVIN_GRIPPER_BAUDRATE", "115200")
    monkeypatch.setenv("MARVIN_GRIPPER_INIT_HEX", "01 06 00 00 00 01 48 0A")
    monkeypatch.setenv(
        "MARVIN_GRIPPER_OPEN_HEX_B",
        "01 10 00 02 00 02 04 00 00 00 00 72 76",
    )
    monkeypatch.setenv(
        "MARVIN_GRIPPER_CLOSE_HEX_B",
        "01 10 00 02 00 02 04 20 42 00 00 D9 A2",
    )

    _FakeSerial.instances = []
    gripper = marvin_usb_gripper.MarvinUsbGripper.from_env("B")
    assert gripper.open() is True
    assert gripper.close() is True
    gripper.cleanup()

    serial_instance = _FakeSerial.instances[-1]
    assert serial_instance.kwargs["port"] == "/dev/ttyUSB7"
    assert serial_instance.kwargs["baudrate"] == 115200
    assert serial_instance.writes == [
        bytes.fromhex("01 06 00 00 00 01 48 0A"),
        bytes.fromhex("01 10 00 02 00 02 04 00 00 00 00 72 76"),
        bytes.fromhex("01 10 00 02 00 02 04 20 42 00 00 D9 A2"),
    ]
    assert serial_instance.closed is True

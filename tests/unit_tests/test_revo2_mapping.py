"""Unit tests for Revo2 hand retarget mapping (pure logic, no hardware)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    REPO_ROOT
    / "rlinf"
    / "envs"
    / "realworld"
    / "common"
    / "hand"
    / "revo2_mapping.py"
)


def _load_mapping():
    spec = importlib.util.spec_from_file_location("revo2_mapping_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-ref resolution can find the module.
    sys.modules["revo2_mapping_test"] = module
    spec.loader.exec_module(module)
    return module


M = _load_mapping()


def test_gripper_mode_syncs_all_fingers():
    t = M.compute_revo2_targets(trigger=1.0, grip=0.0, mode=M.HAND_MODE_GRIPPER)
    assert t.index == t.middle == t.ring == t.pinky == 1.0
    assert t.thumb_bend == 1.0  # gripper mode: thumb follows trigger


def test_two_channel_thumb_follows_grip():
    t = M.compute_revo2_targets(trigger=0.5, grip=0.3, mode=M.HAND_MODE_TWO_CHANNEL)
    assert t.index == t.middle == t.ring == t.pinky == 0.5
    assert t.thumb_bend == 0.3


def test_to_sdk_positions_ranges():
    full = M.Revo2FingerTargets(
        index=1.0, middle=1.0, ring=1.0, pinky=1.0,
        thumb_bend=1.0, thumb_opposition=1.0,
    )
    pos = M.to_sdk_positions(full)
    # order: [Thumb Flex, Thumb Aux, Index, Middle, Ring, Pinky]
    assert pos[0] == M.SDK_POS_THUMB_FLEX_MAX  # 700
    assert pos[1] == M.SDK_POS_THUMB_AUX_MAX  # 1000
    assert pos[2:] == [M.SDK_POS_MAX] * 4


def test_clamp_and_vector_roundtrip():
    t = M.compute_revo2_targets(trigger=2.0, grip=-1.0, mode=M.HAND_MODE_TWO_CHANNEL)
    assert t.index == 1.0 and t.thumb_bend == 0.0  # clamped to [0, 1]
    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rt = M.Revo2FingerTargets.from_normalized_vector(vec).to_normalized_vector()
    assert rt == vec


def test_normalize_hand_mode_aliases():
    assert M.normalize_hand_mode("grasp") == M.HAND_MODE_GRIPPER
    assert M.normalize_hand_mode("two_channel") == M.HAND_MODE_TWO_CHANNEL

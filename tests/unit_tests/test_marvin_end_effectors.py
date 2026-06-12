"""Unit tests for Marvin end-effectors (MarvinGripper adapter, Revo2Hand).

These import the full ``rlinf`` package and so require the project env.
No real hardware is touched: MarvinGripper uses a fake gripper_fn and Revo2Hand
runs in its un-initialized (dummy) path where commands only record intent.
"""

from __future__ import annotations

import numpy as np

from rlinf.envs.realworld.common.end_effectors import (
    EndEffectorType,
    create_end_effector,
    normalize_end_effector_type,
)
from rlinf.envs.realworld.common.hand.revo2_hand import Revo2Hand
from rlinf.envs.realworld.marvin.end_effectors.marvin_gripper import MarvinGripper


def test_enum_extends_marvin_and_revo2():
    assert EndEffectorType.MARVIN_GRIPPER.is_gripper
    assert EndEffectorType.REVO2_HAND.is_hand
    assert normalize_end_effector_type("marvin_gripper") == EndEffectorType.MARVIN_GRIPPER


def test_marvin_gripper_adapter_delegates_with_scale():
    calls = []

    def fake_gripper_fn(pos):
        calls.append(pos)
        return pos <= -0.5 or pos >= 0.5

    class _State:
        gripper_position = 0.7

    g = MarvinGripper(
        gripper_fn=fake_gripper_fn,
        state_getter=lambda: _State(),
        action_scale=lambda: 2.0,
    )
    assert g.action_dim == 1 and g.control_mode == "binary"
    effective = g.command(np.array([0.5]))  # 0.5 * 2.0 = 1.0 -> effective
    assert effective is True and calls[-1] == 1.0
    assert np.allclose(g.get_state(), [0.7])
    # Below threshold -> no change.
    assert g.command(np.array([0.1])) is False


def test_revo2_hand_dummy_records_intent():
    h = Revo2Hand(side="right")  # not initialized -> no SDK, no hardware
    assert h.action_dim == 6 and h.state_dim == 6 and h.control_mode == "continuous"
    target = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert h.command(target) is True
    assert np.allclose(h.get_state(), target)
    # Same target -> not a meaningful change.
    assert h.command(target) is False


def test_revo2_hand_action_length_validation():
    h = Revo2Hand(side="left")
    try:
        h.command(np.zeros(5))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for wrong-length action")


def test_factory_builds_revo2_hand():
    hand = create_end_effector("revo2_hand", side="left")
    assert isinstance(hand, Revo2Hand)

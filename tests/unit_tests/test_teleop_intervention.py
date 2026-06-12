"""Unit tests for composable teleop sources and the generic intervention wrapper.

Imports the full ``rlinf`` package (project env required). No hardware: the
arm teleop is faked and the end-effector teleop is the pure retarget/button
logic.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.teleop import (
    ButtonGripperTeleop,
    VRHandRetargetTeleop,
    VRTriggerGripperTeleop,
)
from rlinf.envs.realworld.common.teleop.protocols import ArmReading, ArmTeleop
from rlinf.envs.realworld.common.wrappers.teleop_intervention import (
    TeleopInterventionWrapper,
)


class _FakeEnv(gym.Env):
    def __init__(self, action_dim: int):
        self.action_space = gym.spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
        self.config = type("C", (), {"is_dummy": True})()
        self.last_action = None

    def get_tcp_pose(self):
        return np.array([0, 0, 0, 0, 0, 0, 1.0])

    def get_action_scale(self):
        return np.array([0.004, 0.1, 1.0])

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        self.last_action = np.asarray(action)
        return {}, 0.0, False, False, {}


class _FixedArm(ArmTeleop):
    def __init__(self, delta, active, aux):
        self._delta, self._active, self._aux = delta, active, aux

    def read(self, tcp_pose7, action_scale):
        return ArmReading(delta6=self._delta, active=self._active, aux=dict(self._aux))


def test_button_gripper_holds_last():
    btn = ButtonGripperTeleop()
    assert btn.action_dim == 1
    close = btn.compute({"buttons": [True, False]})
    assert close.active and close.command[0] <= -0.9
    hold = btn.compute({"buttons": [False, False]})
    assert not hold.active and close.command[0] == hold.command[0]


def test_vr_trigger_gripper_edges():
    g = VRTriggerGripperTeleop(gripper_threshold=0.6)
    assert g.compute({"trigger": 0.0}).command[0] == 0.0  # latch
    assert g.compute({"trigger": 0.9}).command[0] == -1.0  # close edge
    assert g.compute({"trigger": 0.9}).command[0] == 0.0  # held
    assert g.compute({"trigger": 0.1}).command[0] == 1.0  # open edge


def test_vr_hand_retarget_vector_order():
    hand = VRHandRetargetTeleop(mode="gripper", thumb_opposition=0.8)
    r = hand.compute({"trigger": 1.0, "grip": 0.0})
    assert r.active and r.command.shape == (6,)
    # [thumb_bend, thumb_opposition, index, middle, ring, pinky]
    assert np.allclose(r.command, [1.0, 0.8, 1.0, 1.0, 1.0, 1.0])


def test_wrapper_merges_arm_and_hand_12d():
    env = _FakeEnv(action_dim=12)
    arm = _FixedArm(np.ones(6) * 0.5, active=True, aux={"trigger": 1.0, "grip": 0.0})
    w = TeleopInterventionWrapper(env, arm, VRHandRetargetTeleop(mode="gripper"), hold_time=0.5)
    w.reset()
    _, _, _, _, info = w.step(np.zeros(12, dtype=np.float32))
    assert env.last_action.shape == (12,)
    assert np.allclose(env.last_action[:6], 0.5)
    assert np.allclose(env.last_action[6:], [1.0, 0.8, 1.0, 1.0, 1.0, 1.0])
    assert "intervene_action" in info and info["intervene_flag"][0] == 1


def test_wrapper_holds_ee_on_release():
    env = _FakeEnv(action_dim=12)
    arm = _FixedArm(np.zeros(6), active=False, aux={"trigger": 0.0, "grip": 0.0})
    w = TeleopInterventionWrapper(env, arm, VRHandRetargetTeleop(mode="gripper"), hold_time=0.0)
    w.reset()
    policy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9, 9, 9, 9, 9, 9], dtype=np.float32)
    w.step(policy)
    # Released: policy arm kept, last hand target held (not the policy's 9s).
    assert np.allclose(env.last_action[:6], policy[:6])
    assert np.allclose(env.last_action[6:], [0.0, 0.8, 0.0, 0.0, 0.0, 0.0])


def test_wrapper_action_dim_mismatch_raises():
    env = _FakeEnv(action_dim=7)  # gripper-sized, but hand teleop is 6-D -> expects 12
    arm = _FixedArm(np.zeros(6), active=False, aux={})
    try:
        TeleopInterventionWrapper(env, arm, VRHandRetargetTeleop())
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AssertionError on action-dim mismatch")

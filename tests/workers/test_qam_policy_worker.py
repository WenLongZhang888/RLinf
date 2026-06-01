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

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_qam_policy_worker import (
    EmbodiedQAMFSDPPolicy,
    qam_bootstrap_target,
    qam_reduce_ensemble,
)


def test_qam_reduce_ensemble_mean_minus_std_matches_reference():
    q_values = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, 6.0],
        ]
    )

    reduced = qam_reduce_ensemble(
        q_values,
        reduction="mean_minus_std",
        rho=0.5,
    )
    expected = q_values.mean(dim=1, keepdim=True) - 0.5 * q_values.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    )

    assert torch.allclose(reduced, expected)


def test_qam_reduce_ensemble_supports_mean_and_min():
    q_values = torch.tensor([[1.0, 3.0], [2.0, 6.0]])

    assert torch.allclose(
        qam_reduce_ensemble(q_values, reduction="mean"),
        torch.tensor([[2.0], [4.0]]),
    )
    assert torch.allclose(
        qam_reduce_ensemble(q_values, reduction="min"),
        torch.tensor([[1.0], [2.0]]),
    )


def test_qam_reduce_ensemble_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="Unsupported QAM ensemble reduction"):
        qam_reduce_ensemble(torch.ones(2, 2), reduction="median")


def test_qam_bootstrap_target_uses_gamma_to_action_horizon_and_done_mask():
    rewards = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    terminations = torch.tensor(
        [
            [False, False, False],
            [False, True, False],
        ]
    )
    next_q_values = torch.tensor(
        [
            [10.0, 14.0],
            [20.0, 28.0],
        ]
    )

    target = qam_bootstrap_target(
        rewards=rewards,
        terminations=terminations,
        next_q_values=next_q_values,
        gamma=0.9,
        action_horizon=3,
        reduction="mean",
        bootstrap_type="standard",
    )

    expected_first = rewards[0].sum() + (0.9**3) * next_q_values[0].mean()
    expected_second = rewards[1].sum()
    expected = torch.tensor([[expected_first], [expected_second]])

    assert torch.allclose(target, expected)


def test_qam_bootstrap_target_can_always_bootstrap():
    rewards = torch.tensor([[1.0, 2.0]])
    terminations = torch.tensor([[True, True]])
    next_q_values = torch.tensor([[4.0, 8.0]])

    target = qam_bootstrap_target(
        rewards=rewards,
        terminations=terminations,
        next_q_values=next_q_values,
        gamma=0.5,
        action_horizon=2,
        reduction="mean",
        bootstrap_type="always",
    )

    expected = torch.tensor([[3.0 + (0.5**2) * 6.0]])
    assert torch.allclose(target, expected)


def test_qam_infer_batch_size_from_obs_uses_first_tensor():
    obs = {
        "tokenized_prompt": torch.ones(7, 48, dtype=torch.long),
        "metadata": "ignored",
    }

    assert EmbodiedQAMFSDPPolicy._infer_batch_size_from_obs(obs) == 7


def test_qam_infer_batch_size_from_obs_rejects_tensorless_obs():
    with pytest.raises(ValueError, match="batched tensor"):
        EmbodiedQAMFSDPPolicy._infer_batch_size_from_obs({"metadata": "ignored"})


def test_qam_action_shape_prefers_critic_env_fields_without_model():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "num_action_chunks": 5,
                    "action_dim": 7,
                    "openpi": {
                        "action_horizon": 8,
                        "action_env_dim": 6,
                    },
                }
            }
        }
    )

    assert worker.qam_action_shape(batch_size=3) == (3, 8, 6)
    assert worker.qam_critic_action_shape(batch_size=3) == (3, 8, 6)


class _DummyOpenPiModule:
    def __init__(self):
        self.config = type(
            "Config",
            (),
            {
                "action_horizon": 5,
                "action_chunk": 5,
                "action_dim": 32,
                "action_env_dim": 7,
            },
        )()


class _DummyFSDPWrapper:
    def __init__(self):
        self.module = _DummyOpenPiModule()


class _EncodeOnlyModel(nn.Module):
    def __init__(self, pooled_z):
        super().__init__()
        self.pooled_z = pooled_z
        self.calls = []

    def forward(self, forward_type, **kwargs):
        self.calls.append((forward_type, kwargs))
        assert forward_type == ForwardType.QAM_ENCODE
        return self.pooled_z


class _LinearQHead(nn.Module):
    def forward(self, pooled_z, actions):
        del pooled_z
        return actions.sum(dim=-1, keepdim=True)


def test_qam_flow_and_critic_shapes_use_distinct_openpi_dims():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create({"actor": {"model": {}}})
    worker.model = _DummyFSDPWrapper()

    assert worker.qam_flow_action_shape(batch_size=2) == (2, 5, 32)
    assert worker.qam_critic_action_shape(batch_size=2) == (2, 5, 7)
    assert worker.qam_action_shape(batch_size=2) == (2, 5, 7)


def test_qam_critic_actions_from_flow_slices_model_action_space():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create({"actor": {"model": {}}})
    worker.model = _DummyFSDPWrapper()
    flow_actions = torch.arange(2 * 5 * 32).reshape(2, 5, 32)

    critic_actions = worker._critic_actions_from_flow(flow_actions)

    assert critic_actions.shape == (2, 5, 7)
    assert torch.equal(critic_actions, flow_actions[:, :5, :7])


def test_qam_builds_independent_target_head_from_config():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "openpi": {
                        "config_name": "pi05_libero",
                        "action_chunk": 5,
                        "action_env_dim": 7,
                        "qam_q_hidden_dims": [16],
                        "qam_num_q_heads": 2,
                    }
                }
            }
        }
    )
    worker.model = None

    q_head = worker._build_qam_q_head()

    out = q_head(torch.randn(3, 2048), torch.randn(3, 35))
    assert isinstance(q_head, nn.Module)
    assert out.shape == (3, 2)


def test_qam_soft_update_target_head_tau_one_matches_online():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create({"algorithm": {"tau": 0.1}})
    worker.q_head_qam = nn.Linear(3, 2)
    worker.target_q_head_qam = nn.Linear(3, 2)
    with torch.no_grad():
        worker.q_head_qam.weight.fill_(2.0)
        worker.q_head_qam.bias.fill_(3.0)
        worker.target_q_head_qam.weight.zero_()
        worker.target_q_head_qam.bias.zero_()

    worker.soft_update_target_model(tau=1.0)

    for online_param, target_param in zip(
        worker.q_head_qam.parameters(),
        worker.target_q_head_qam.parameters(),
    ):
        assert torch.allclose(target_param, online_param)


def test_qam_soft_update_target_head_tau_half_only_changes_target():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create({"algorithm": {"tau": 0.1}})
    worker.q_head_qam = nn.Linear(1, 1, bias=False)
    worker.target_q_head_qam = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        worker.q_head_qam.weight.fill_(4.0)
        worker.target_q_head_qam.weight.fill_(2.0)

    worker.soft_update_target_model(tau=0.5)

    assert torch.allclose(worker.q_head_qam.weight, torch.tensor([[4.0]]))
    assert torch.allclose(worker.target_q_head_qam.weight, torch.tensor([[3.0]]))


def test_qam_q_values_uses_worker_owned_online_and_target_heads():
    pooled_z = torch.randn(2, 4)
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.model = _EncodeOnlyModel(pooled_z)
    worker.q_head_qam = _LinearQHead()
    worker.target_q_head_qam = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        worker.target_q_head_qam.weight.fill_(2.0)
    obs = {"states": torch.zeros(2, 1)}
    actions = torch.ones(2, 3)

    online_q = worker._qam_q_values(obs, actions, target=False)
    target_q = worker._qam_q_values(obs, actions, target=True)

    assert torch.allclose(online_q, torch.full((2, 1), 3.0))
    assert torch.allclose(target_q, torch.full((2, 1), 6.0))
    assert not hasattr(worker, "target_model")


def test_qam_q_grad_fn_pads_critic_gradient_to_flow_shape():
    worker = object.__new__(EmbodiedQAMFSDPPolicy)
    worker.cfg = OmegaConf.create({"actor": {"model": {}}})
    worker.model = _DummyFSDPWrapper()
    worker.f_beta_model = nn.Identity()
    worker._qam_q_values = lambda obs, actions, target: actions.sum(dim=-1, keepdim=True)
    obs = {"states": torch.zeros(2, 1)}

    _, _, q_grad_fn = worker._make_qam_closures(obs)
    grad_flow = q_grad_fn(torch.zeros(2, 5, 32))

    assert grad_flow.shape == (2, 5, 32)
    assert torch.allclose(grad_flow[:, :, :7], torch.ones(2, 5, 7))
    assert torch.allclose(grad_flow[:, :, 7:], torch.zeros(2, 5, 25))

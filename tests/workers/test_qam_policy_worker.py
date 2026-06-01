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
from omegaconf import OmegaConf

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


def test_qam_action_shape_prefers_openpi_model_fields():
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

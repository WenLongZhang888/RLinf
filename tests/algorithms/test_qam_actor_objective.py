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

import math

import pytest
import torch

from rlinf.algorithms.embodiment import (
    compute_qam_actor_objective,
    sample_forward_sde,
)


def test_sample_forward_sde_single_step_uses_beta_ode():
    """With W=1, the sampler should skip SDE noise and use the beta ODE step."""
    action_shape = (2, 3, 1)
    obs = torch.zeros(action_shape, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    expected_generator = torch.Generator().manual_seed(0)
    x0 = torch.randn(action_shape, dtype=torch.float64, generator=expected_generator)
    beta_velocity = torch.full(action_shape, 0.25, dtype=torch.float64)

    def f_theta_fn(obs_payload, x_t, timestep):
        raise AssertionError("f_theta_fn should not be called when num_steps=1")

    def f_beta_fn(obs_payload, x_t, timestep):
        assert obs_payload is obs
        assert torch.allclose(timestep, torch.zeros(x_t.shape[0], dtype=x_t.dtype))
        return beta_velocity.to(device=x_t.device, dtype=x_t.dtype)

    xs = sample_forward_sde(
        f_theta_fn=f_theta_fn,
        f_beta_fn=f_beta_fn,
        obs=obs,
        action_shape=action_shape,
        num_steps=1,
        generator=generator,
    )

    expected = torch.stack([x0, x0 + beta_velocity], dim=0)
    assert xs.shape == (2, *action_shape)
    assert torch.allclose(xs, expected)


def test_sample_forward_sde_matches_h_shifted_noise_reference():
    """Fixed generator should make the h-shifted SDE update exactly checkable."""
    action_shape = (1, 2)
    num_steps = 3
    h = 1.0 / num_steps
    obs = torch.zeros(action_shape, dtype=torch.float64)
    theta_velocity = torch.full(action_shape, 0.4, dtype=torch.float64)
    beta_velocity = torch.full(action_shape, -0.2, dtype=torch.float64)

    generator = torch.Generator().manual_seed(123)
    reference_generator = torch.Generator().manual_seed(123)

    def f_theta_fn(obs_payload, x_t, timestep):
        return theta_velocity.to(device=x_t.device, dtype=x_t.dtype)

    def f_beta_fn(obs_payload, x_t, timestep):
        return beta_velocity.to(device=x_t.device, dtype=x_t.dtype)

    xs = sample_forward_sde(
        f_theta_fn=f_theta_fn,
        f_beta_fn=f_beta_fn,
        obs=obs,
        action_shape=action_shape,
        num_steps=num_steps,
        generator=generator,
    )

    x_t = torch.randn(action_shape, dtype=torch.float64, generator=reference_generator)
    expected = [x_t.clone()]
    for step in range(num_steps - 1):
        t_value = step / num_steps
        sigma = math.sqrt(2.0 * (1.0 - t_value + h) / (t_value + h))
        noise = torch.randn(
            action_shape,
            dtype=torch.float64,
            generator=reference_generator,
        )
        drift = 2.0 * theta_velocity - x_t / (t_value + h)
        x_t = x_t + h * drift + math.sqrt(h) * sigma * noise
        expected.append(x_t.clone())
    x_t = x_t + h * beta_velocity
    expected.append(x_t.clone())

    assert torch.allclose(xs, torch.stack(expected, dim=0))


def test_sample_forward_sde_rejects_invalid_inputs():
    obs = torch.zeros(2, 3)

    def f_theta_fn(obs_payload, x_t, timestep):
        return torch.zeros_like(x_t)

    def f_beta_fn(obs_payload, x_t, timestep):
        return torch.zeros_like(x_t)

    with pytest.raises(ValueError, match="num_steps"):
        sample_forward_sde(f_theta_fn, f_beta_fn, obs, (2, 3), 0)
    with pytest.raises(ValueError, match="action_shape"):
        sample_forward_sde(f_theta_fn, f_beta_fn, obs, (2,), 1)
    with pytest.raises(ValueError, match="positive"):
        sample_forward_sde(f_theta_fn, f_beta_fn, obs, (2, 0), 1)


def test_sample_forward_sde_rejects_bad_velocity_shape():
    obs = torch.zeros(2, 3)

    def f_theta_fn(obs_payload, x_t, timestep):
        return torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

    def f_beta_fn(obs_payload, x_t, timestep):
        return torch.zeros_like(x_t)

    with pytest.raises(ValueError, match="same shape"):
        sample_forward_sde(f_theta_fn, f_beta_fn, obs, (2, 3), 2)


def test_compute_qam_actor_objective_returns_metrics_and_actor_gradients():
    torch.manual_seed(0)
    action_shape = (2, 3)
    obs = torch.zeros(action_shape, dtype=torch.float64)
    theta_weight = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    beta_weight = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    generator = torch.Generator().manual_seed(11)

    def f_theta_fn(obs_payload, x_t, timestep):
        return theta_weight * x_t + timestep.view(-1, 1) * 0.05

    def f_beta_fn(obs_payload, x_t, timestep):
        return beta_weight.detach() * x_t

    def q_grad_fn(x1):
        return x1.detach().clone()

    loss, metrics = compute_qam_actor_objective(
        f_theta_fn=f_theta_fn,
        f_beta_fn=f_beta_fn,
        q_grad_fn=q_grad_fn,
        obs=obs,
        action_shape=action_shape,
        num_steps=4,
        inv_temp=0.3,
        generator=generator,
    )

    assert torch.isfinite(loss)
    assert set(metrics) >= {
        "actor/qam_loss",
        "actor/qam_residual_abs",
        "actor/qam_velocity_delta_abs",
        "actor/qam_adj_abs",
        "actor/qam_sigma_min",
        "actor/qam_sigma_max",
        "actor/qam_valid_count",
    }

    loss.backward()
    assert theta_weight.grad is not None
    assert theta_weight.grad.abs() > 0
    assert beta_weight.grad is None


def test_compute_qam_actor_objective_rejects_bad_q_grad_shape():
    action_shape = (2, 3)
    obs = torch.zeros(action_shape, dtype=torch.float64)

    def f_theta_fn(obs_payload, x_t, timestep):
        return torch.zeros_like(x_t)

    def f_beta_fn(obs_payload, x_t, timestep):
        return torch.zeros_like(x_t)

    def q_grad_fn(x1):
        return torch.zeros(x1.shape[0], device=x1.device, dtype=x1.dtype)

    with pytest.raises(ValueError, match="same shape"):
        compute_qam_actor_objective(
            f_theta_fn=f_theta_fn,
            f_beta_fn=f_beta_fn,
            q_grad_fn=q_grad_fn,
            obs=obs,
            action_shape=action_shape,
            num_steps=2,
            inv_temp=0.3,
        )

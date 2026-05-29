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

from rlinf.algorithms.embodiment import compute_adjoint_states


def test_compute_adjoint_states_constant_field_matches_analytic_solution():
    """Toy P3.3 sanity: Q(a)=||a||^2 and f_beta(x, w)=c."""
    torch.manual_seed(0)
    num_steps = 5
    lambda_ = 2.0
    x1 = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    q_value = x1.square().sum()
    (q_grad_at_1,) = torch.autograd.grad(q_value, x1)
    x1 = x1.detach()
    q_grad_at_1 = q_grad_at_1.detach()
    constant_velocity = torch.randn_like(x1)
    obs = {"source": "toy"}

    def f_beta_fn(obs_payload, x_t, timestep):
        assert obs_payload is obs
        assert timestep.shape == (x_t.shape[0],)
        return constant_velocity.to(device=x_t.device, dtype=x_t.dtype)

    xs, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=obs,
        x1=x1,
        Q_grad_at_1=q_grad_at_1,
        num_steps=num_steps,
        lambda_=lambda_,
    )

    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=x1.dtype)
    expected_xs = torch.stack(
        [x1 - (1.0 - t) * constant_velocity for t in times],
        dim=0,
    )
    expected_adj = -q_grad_at_1 / lambda_
    expected_adjs = expected_adj.expand(num_steps + 1, *expected_adj.shape)

    assert xs.shape == (num_steps + 1, *x1.shape)
    assert adjs.shape == (num_steps + 1, *x1.shape)
    assert torch.allclose(xs, expected_xs)
    assert torch.allclose(adjs, expected_adjs)
    assert torch.allclose(xs[-1], x1)


def test_compute_adjoint_states_linear_field_uses_vjp_recursion():
    """For f_beta(x, w)=alpha*x, adjoints follow the discrete VJP recurrence."""
    torch.manual_seed(1)
    num_steps = 4
    alpha = 0.3
    x1 = torch.randn(2, 2, 3, dtype=torch.float64)
    q_grad_at_1 = torch.randn_like(x1)

    def f_beta_fn(obs, x_t, timestep):
        return alpha * x_t

    _, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=None,
        x1=x1,
        Q_grad_at_1=q_grad_at_1,
        num_steps=num_steps,
        lambda_=1.0,
    )

    h = 1.0 / num_steps
    terminal_adj = -q_grad_at_1
    expected = []
    for index in range(num_steps + 1):
        power = num_steps - index
        expected.append(((1.0 + h * alpha) ** power) * terminal_adj)
    expected_adjs = torch.stack(expected, dim=0)

    assert torch.allclose(adjs, expected_adjs)


def test_compute_adjoint_states_rejects_invalid_inputs():
    x1 = torch.zeros(2, 3, 4)
    q_grad = torch.zeros_like(x1)

    def f_beta_fn(obs, x_t, timestep):
        return torch.zeros_like(x_t)

    with pytest.raises(ValueError, match="num_steps"):
        compute_adjoint_states(f_beta_fn, None, x1, q_grad, 0, 1.0)
    with pytest.raises(ValueError, match="same shape"):
        compute_adjoint_states(f_beta_fn, None, x1, q_grad[:, :, :3], 2, 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        compute_adjoint_states(f_beta_fn, None, x1, q_grad, 2, 0.0)

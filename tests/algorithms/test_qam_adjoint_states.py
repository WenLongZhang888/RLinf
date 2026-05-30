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


def test_compute_adjoint_states_constant_field_matches_paper_drift():
    """Toy P3.3 sanity: Q(a)=||a||^2 and b_beta(x,t)=2c-x/t."""
    torch.manual_seed(0)
    num_steps = 5
    lambda_ = 2.0
    xs = torch.randn(num_steps + 1, 2, 3, 4, dtype=torch.float64)
    x1 = xs[-1].detach().requires_grad_(True)
    q_value = x1.square().sum()
    (q_grad_at_1,) = torch.autograd.grad(q_value, x1)
    q_grad_at_1 = q_grad_at_1.detach()
    constant_velocity = torch.randn_like(xs[-1])
    obs = {"source": "toy"}

    def f_beta_fn(obs_payload, x_t, timestep):
        assert obs_payload is obs
        assert timestep.shape == (x_t.shape[0],)
        return constant_velocity.to(device=x_t.device, dtype=x_t.dtype)

    returned_xs, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=obs,
        xs=xs,
        Q_grad_at_1=q_grad_at_1,
        lambda_=lambda_,
    )

    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=x1.dtype)
    terminal_adj = -q_grad_at_1 / lambda_
    expected_adjs = times.view(-1, 1, 1, 1) * terminal_adj

    assert returned_xs.shape == xs.shape
    assert adjs.shape == (num_steps + 1, *x1.shape)
    assert torch.allclose(returned_xs, xs)
    assert torch.allclose(adjs, expected_adjs)
    assert torch.allclose(returned_xs[-1], x1.detach())


def test_compute_adjoint_states_linear_field_uses_full_drift_vjp():
    """For f_beta=alpha*x, VJP uses 2*alpha*x-x/t, not f_beta alone."""
    torch.manual_seed(1)
    num_steps = 4
    alpha = 0.3
    xs = torch.randn(num_steps + 1, 2, 2, 3, dtype=torch.float64)
    q_grad_at_1 = torch.randn_like(xs[-1])

    def f_beta_fn(obs, x_t, timestep):
        return alpha * x_t

    _, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=None,
        xs=xs,
        Q_grad_at_1=q_grad_at_1,
        lambda_=1.0,
    )

    h = 1.0 / num_steps
    terminal_adj = -q_grad_at_1
    expected: list[torch.Tensor] = [torch.empty_like(terminal_adj)] * (
        num_steps + 1
    )
    expected[-1] = terminal_adj
    for step in range(num_steps, 0, -1):
        timestep = step / num_steps
        factor = 1.0 + h * (2.0 * alpha - 1.0 / timestep)
        expected[step - 1] = factor * expected[step]
    expected_adjs = torch.stack(expected, dim=0)

    assert torch.allclose(adjs, expected_adjs)


def test_compute_adjoint_states_state_dependent_jacobian_uses_source_state():
    """f_beta(x)=x^2 has a state-dependent Jacobian, so the VJP must be taken at
    the source state traj[step] (QAM paper Eq. 25), not traj[step-1]."""
    torch.manual_seed(2)
    num_steps = 4
    xs = torch.randn(num_steps + 1, 2, 2, 3, dtype=torch.float64)
    q_grad_at_1 = torch.randn_like(xs[-1])

    def f_beta_fn(obs, x_t, timestep):
        # Elementwise square -> Jacobian diag(2 x), i.e. genuinely depends on x.
        return x_t * x_t

    _, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=None,
        xs=xs,
        Q_grad_at_1=q_grad_at_1,
        lambda_=1.0,
    )

    # b_beta = 2 x^2 - x / t  ->  d b/d x = diag(4 x - 1/t).
    # Reference linearizes at the SOURCE state xs[step] with t = step/W.
    h = 1.0 / num_steps
    expected = [torch.empty_like(q_grad_at_1)] * (num_steps + 1)
    expected[-1] = -q_grad_at_1
    for step in range(num_steps, 0, -1):
        timestep = step / num_steps
        jac_diag = 4.0 * xs[step] - 1.0 / timestep
        expected[step - 1] = expected[step] + h * jac_diag * expected[step]
    expected_adjs = torch.stack(expected, dim=0)

    assert torch.allclose(adjs, expected_adjs)

    # Sanity: evaluating at the WRONG state (traj[step-1]) would differ.
    wrong = [torch.empty_like(q_grad_at_1)] * (num_steps + 1)
    wrong[-1] = -q_grad_at_1
    for step in range(num_steps, 0, -1):
        timestep = step / num_steps
        jac_diag = 4.0 * xs[step - 1] - 1.0 / timestep
        wrong[step - 1] = wrong[step] + h * jac_diag * wrong[step]
    assert not torch.allclose(adjs, torch.stack(wrong, dim=0))


def test_compute_adjoint_states_rejects_invalid_inputs():
    xs = torch.zeros(3, 2, 3, 4)
    q_grad = torch.zeros_like(xs[-1])

    def f_beta_fn(obs, x_t, timestep):
        return torch.zeros_like(x_t)

    with pytest.raises(ValueError, match="at least two states"):
        compute_adjoint_states(f_beta_fn, None, xs[:1], q_grad, 1.0)
    with pytest.raises(ValueError, match="same shape"):
        compute_adjoint_states(f_beta_fn, None, xs, q_grad[:, :, :3], 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        compute_adjoint_states(f_beta_fn, None, xs, q_grad, 0.0)

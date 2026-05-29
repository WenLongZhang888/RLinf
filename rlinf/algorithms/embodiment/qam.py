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

from collections.abc import Callable
from typing import Any

import torch


def compute_adjoint_states(
    f_beta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    obs: Any,
    x1: torch.Tensor,
    Q_grad_at_1: torch.Tensor,
    num_steps: int,
    lambda_: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QAM adjoint states along a frozen base-flow trajectory.

    This is the model-agnostic P3.3 primitive for plain QAM.  It only assumes
    that ``f_beta_fn(obs, x_t, timestep)`` returns the frozen behavior velocity
    with the same shape as ``x_t``.  No worker, FSDP, or OpenPI-specific state
    is accessed here.

    The input ``x1`` is the terminal action at flow time ``w=1`` where the
    critic action gradient was evaluated.  The function first reconstructs the
    frozen base trajectory with explicit reverse Euler, then integrates the
    adjoint state backward using VJP products:

    ``g_w = g_{w+h} + h * J_x f_beta(x_w, w+h)^T g_{w+h}``.

    Returned tensors are ordered by increasing flow time, so ``xs[0]`` and
    ``adjs[0]`` correspond to ``w=0``, while ``xs[-1] == x1`` and ``adjs[-1]``
    correspond to ``w=1``.

    Args:
        f_beta_fn: Frozen base velocity function.
        obs: Observation payload passed through to ``f_beta_fn``.
        x1: Terminal action tensor, shape ``[B, H, A]``.
        Q_grad_at_1: Raw critic action gradient at ``x1`` with the same shape.
        num_steps: Number of uniform Euler/VJP intervals on ``[0, 1]``.
        lambda_: QAM temperature/regularization scale.  The terminal adjoint is
            ``-Q_grad_at_1 / lambda_``.

    Returns:
        A pair ``(xs, adjs)`` with shape ``[num_steps + 1, *x1.shape]``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if x1.shape != Q_grad_at_1.shape:
        raise ValueError(
            "x1 and Q_grad_at_1 must have the same shape, got "
            f"{tuple(x1.shape)} and {tuple(Q_grad_at_1.shape)}"
        )
    if not x1.is_floating_point():
        raise TypeError(f"x1 must be a floating point tensor, got {x1.dtype}")
    if not Q_grad_at_1.is_floating_point():
        raise TypeError(
            "Q_grad_at_1 must be a floating point tensor, "
            f"got {Q_grad_at_1.dtype}"
        )

    lambda_t = torch.as_tensor(lambda_, device=x1.device, dtype=x1.dtype)
    if bool(torch.any(lambda_t == 0).item()):
        raise ValueError("lambda_ must be non-zero")

    h = 1.0 / num_steps
    batch_size = x1.shape[0]

    def _time(value: float) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            value,
            device=x1.device,
            dtype=x1.dtype,
        )

    reverse_xs = [x1.detach()]
    for step in range(num_steps, 0, -1):
        t_next = _time(step / num_steps)
        x_next = reverse_xs[-1]
        with torch.no_grad():
            velocity = f_beta_fn(obs, x_next, t_next)
        if velocity.shape != x1.shape:
            raise ValueError(
                "f_beta_fn must return the same shape as x1, got "
                f"{tuple(velocity.shape)} and expected {tuple(x1.shape)}"
            )
        reverse_xs.append((x_next - h * velocity).detach())

    xs = torch.stack(list(reversed(reverse_xs)), dim=0)

    adjs: list[torch.Tensor] = [torch.empty_like(x1) for _ in range(num_steps + 1)]
    adjs[-1] = (-Q_grad_at_1.to(device=x1.device, dtype=x1.dtype) / lambda_t).detach()

    for step in range(num_steps, 0, -1):
        t_next = _time(step / num_steps)
        x_curr = xs[step - 1].detach().requires_grad_(True)
        adj_next = adjs[step].detach()

        def velocity_at_x(input_x: torch.Tensor) -> torch.Tensor:
            velocity = f_beta_fn(obs, input_x, t_next)
            if velocity.shape != input_x.shape:
                raise ValueError(
                    "f_beta_fn must return the same shape as its x_t input, "
                    f"got {tuple(velocity.shape)} and expected "
                    f"{tuple(input_x.shape)}"
                )
            return velocity

        _, vjp_x = torch.autograd.functional.vjp(
            velocity_at_x,
            x_curr,
            v=adj_next,
            create_graph=False,
            strict=False,
        )
        adjs[step - 1] = (adj_next + h * vjp_x).detach()

    return xs, torch.stack(adjs, dim=0)

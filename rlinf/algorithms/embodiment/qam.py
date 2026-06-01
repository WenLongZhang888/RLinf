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
from collections.abc import Callable, Sequence
from typing import Any

import torch


def _validate_same_shape(
    name: str,
    tensor: torch.Tensor,
    reference_name: str,
    reference: torch.Tensor,
) -> None:
    if tensor.shape != reference.shape:
        raise ValueError(
            f"{name} and {reference_name} must have the same shape, got "
            f"{tuple(tensor.shape)} and {tuple(reference.shape)}"
        )


def _validate_velocity_output(
    name: str,
    velocity: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    if velocity.shape != x_t.shape:
        raise ValueError(
            f"{name} must return the same shape as its x_t input, got "
            f"{tuple(velocity.shape)} and expected {tuple(x_t.shape)}"
        )
    if not velocity.is_floating_point():
        raise TypeError(
            f"{name} must return a floating point tensor, got {velocity.dtype}"
        )
    return velocity.to(device=x_t.device, dtype=x_t.dtype)


def _as_action_shape(action_shape: Sequence[int] | torch.Size) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in action_shape)
    except TypeError as exc:
        raise TypeError("action_shape must be a sequence of positive integers") from exc
    if len(shape) < 2:
        raise ValueError(
            "action_shape must include batch and at least one action dimension, "
            f"got {shape}"
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"action_shape dimensions must be positive, got {shape}")
    return shape


def _find_first_tensor(payload: Any) -> torch.Tensor | None:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            tensor = _find_first_tensor(value)
            if tensor is not None:
                return tensor
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            tensor = _find_first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _infer_device_dtype(payload: Any) -> tuple[torch.device, torch.dtype]:
    tensor = _find_first_tensor(payload)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    return tensor.device, dtype


def _time_batch(
    batch_size: int,
    value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.full((batch_size,), value, device=device, dtype=dtype)


def sample_forward_sde(
    f_theta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    f_beta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    obs: Any,
    action_shape: Sequence[int] | torch.Size,
    num_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a fixed QAM forward SDE trajectory.

    The returned trajectory is ordered by increasing flow time and has shape
    ``[num_steps + 1, B, *action_dims]``. Following the official QAM sampler,
    velocity fields are evaluated on the unshifted grid ``t = i / W`` while the
    diffusion and singular drift denominator use the h-shift ``t + h``.

    Args:
        f_theta_fn: Trainable velocity closure with signature
            ``(obs, x_t, timestep) -> velocity``.
        f_beta_fn: Frozen behavior velocity closure used for the final pure
            ODE Euler step.
        obs: Observation payload passed through to both velocity closures.
        action_shape: Full batched action shape ``[B, *action_dims]``.
        num_steps: Number of flow steps ``W``.
        generator: Optional random generator for reproducible noise.

    Returns:
        Sampled trajectory ``xs`` with shape ``[W + 1, B, *action_dims]``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    shape = _as_action_shape(action_shape)
    batch_size = shape[0]
    device, dtype = _infer_device_dtype(obs)
    h = 1.0 / num_steps
    sqrt_h = math.sqrt(h)

    with torch.no_grad():
        x_t = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        xs = [x_t.clone()]

        for step in range(num_steps - 1):
            t_value = step / num_steps
            timestep = _time_batch(batch_size, t_value, device, dtype)
            velocity = _validate_velocity_output(
                "f_theta_fn",
                f_theta_fn(obs, x_t, timestep),
                x_t,
            )
            sigma = math.sqrt(2.0 * (1.0 - t_value + h) / (t_value + h))
            noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            drift = 2.0 * velocity - x_t / (t_value + h)
            x_t = x_t + h * drift + sqrt_h * sigma * noise
            xs.append(x_t.clone())

        timestep = _time_batch(batch_size, (num_steps - 1) / num_steps, device, dtype)
        beta_velocity = _validate_velocity_output(
            "f_beta_fn",
            f_beta_fn(obs, x_t, timestep),
            x_t,
        )
        x_t = x_t + h * beta_velocity
        xs.append(x_t.clone())

    return torch.stack(xs, dim=0)


def compute_qam_actor_objective(
    f_theta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    f_beta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    q_grad_fn: Callable[[torch.Tensor], torch.Tensor],
    obs: Any,
    action_shape: Sequence[int] | torch.Size,
    num_steps: int,
    inv_temp: float | torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the plain QAM actor objective with closure-provided components.

    This orchestrates trajectory sampling, target-critic terminal seeding,
    adjoint propagation, and step-wise adjoint matching while remaining
    independent of workers, FSDP, and OpenPI-specific state.

    Gradients flow through ``f_theta_fn`` only when velocities are re-evaluated
    on the fixed sampled trajectory. Sampling, the frozen behavior velocity,
    and propagated adjoints are detached from the actor update.

    Args:
        f_theta_fn: Trainable actor velocity closure.
        f_beta_fn: Frozen behavior velocity closure.
        q_grad_fn: Closure returning the raw terminal critic action gradient at
            ``xs[-1]`` with the same shape as the action trajectory state.
        obs: Observation payload passed through to velocity closures.
        action_shape: Full batched action shape ``[B, *action_dims]``.
        num_steps: Number of flow steps ``W``.
        inv_temp: QAM inverse temperature ``tau``. The adjoint helper receives
            ``lambda_ = 1 / inv_temp``.
        loss_mask: Optional mask forwarded to :func:`compute_qam_actor_loss`.
        generator: Optional random generator for reproducible trajectory
            sampling.

    Returns:
        ``(loss, metrics)`` from :func:`compute_qam_actor_loss`.
    """
    xs = sample_forward_sde(
        f_theta_fn=f_theta_fn,
        f_beta_fn=f_beta_fn,
        obs=obs,
        action_shape=action_shape,
        num_steps=num_steps,
        generator=generator,
    )

    inv_temp_t = torch.as_tensor(inv_temp, device=xs.device, dtype=xs.dtype)
    if bool(torch.any(inv_temp_t == 0).item()):
        raise ValueError("inv_temp must be non-zero")

    q_grad_at_1 = q_grad_fn(xs[-1])
    _validate_same_shape("q_grad_fn output", q_grad_at_1, "xs[-1]", xs[-1])
    if not q_grad_at_1.is_floating_point():
        raise TypeError(
            "q_grad_fn must return a floating point tensor, "
            f"got {q_grad_at_1.dtype}"
        )

    _, adjs = compute_adjoint_states(
        f_beta_fn=f_beta_fn,
        obs=obs,
        xs=xs,
        Q_grad_at_1=q_grad_at_1,
        lambda_=1.0 / inv_temp_t,
    )

    batch_size = xs.shape[1]
    vf_fine: list[torch.Tensor] = []
    vf_base: list[torch.Tensor] = []
    for step in range(num_steps):
        x_t = xs[step].detach()
        timestep = _time_batch(
            batch_size,
            step / num_steps,
            x_t.device,
            x_t.dtype,
        )
        vf_fine.append(
            _validate_velocity_output(
                "f_theta_fn",
                f_theta_fn(obs, x_t, timestep),
                x_t,
            )
        )
        with torch.no_grad():
            vf_base.append(
                _validate_velocity_output(
                    "f_beta_fn",
                    f_beta_fn(obs, x_t, timestep),
                    x_t,
                )
            )

    # ``compute_qam_actor_loss`` uses the full ``[W + 1, B, ...]`` shape to
    # infer ``h = 1 / W``, then drops the terminal row when ``skip_terminal`` is
    # true. Padding avoids an unnecessary actor call at t=1.
    vf_fine.append(torch.zeros_like(xs[-1]))
    vf_base.append(torch.zeros_like(xs[-1]))

    return compute_qam_actor_loss(
        vf_fine=torch.stack(vf_fine, dim=0),
        vf_base=torch.stack(vf_base, dim=0),
        adjs=adjs,
        loss_mask=loss_mask,
        skip_terminal=True,
    )


def compute_adjoint_states(
    f_beta_fn: Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor],
    obs: Any,
    xs: torch.Tensor,
    Q_grad_at_1: torch.Tensor,
    lambda_: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QAM lean adjoint states along a forward actor trajectory.

    The caller provides the forward trajectory ``xs`` generated by the current
    actor ``f_theta``. This helper only uses the frozen behavior velocity
    ``f_beta_fn`` to propagate the terminal critic gradient backward in time.
    It does not depend on workers, FSDP, or OpenPI-specific state.

    Following QAM Eq. (25), the VJP is taken through the complete behavior
    drift

    ``b_beta(x, t) = 2 * f_beta(obs, x, t) - x / t``,

    not through ``f_beta`` alone. The trajectory and returned adjoints are
    ordered by increasing flow time: ``xs[0]`` / ``adjs[0]`` correspond to
    ``t=0`` and ``xs[-1]`` / ``adjs[-1]`` correspond to ``t=1``. The reverse
    recursion only evaluates the drift at ``t > 0``.

    Args:
        f_beta_fn: Frozen behavior velocity function with signature
            ``(obs, x_t, timestep) -> velocity``. It may have frozen
            parameters, but must still build an autograd graph with respect to
            ``x_t`` so that VJP can be computed.
        obs: Observation payload passed through to ``f_beta_fn``.
        xs: Forward actor trajectory, shape ``[num_steps + 1, B, ...]``.
        Q_grad_at_1: Raw critic action gradient at ``xs[-1]``.
        lambda_: QAM temperature/regularization scale. The terminal adjoint is
            ``-Q_grad_at_1 / lambda_``.

    Returns:
        A pair ``(xs, adjs)`` with matching shape ``[num_steps + 1, B, ...]``.
    """
    if xs.ndim < 2:
        raise ValueError(f"xs must have shape [W+1, B, ...], got {tuple(xs.shape)}")
    if not xs.is_floating_point():
        raise TypeError(f"xs must be a floating point tensor, got {xs.dtype}")

    traj = xs.detach()
    num_steps = traj.shape[0] - 1
    if num_steps <= 0:
        raise ValueError(f"xs must contain at least two states, got {traj.shape[0]}")

    x1 = traj[-1]
    if x1.shape != Q_grad_at_1.shape:
        raise ValueError(
            "xs[-1] and Q_grad_at_1 must have the same shape, got "
            f"{tuple(x1.shape)} and {tuple(Q_grad_at_1.shape)}"
        )
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

    def _time_like(timestep: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return timestep.view(batch_size, *([1] * (x_t.ndim - 1)))

    adjs: list[torch.Tensor] = [torch.empty_like(x1) for _ in range(num_steps + 1)]
    adjs[-1] = (-Q_grad_at_1.to(device=x1.device, dtype=x1.dtype) / lambda_t).detach()

    for step in range(num_steps, 0, -1):
        timestep = _time(step / num_steps)
        x_t = traj[step].detach().requires_grad_(True)
        adj_next = adjs[step].detach()

        def beta_drift_at_x(input_x: torch.Tensor) -> torch.Tensor:
            velocity = f_beta_fn(obs, input_x, timestep)
            if velocity.shape != input_x.shape:
                raise ValueError(
                    "f_beta_fn must return the same shape as its x_t input, "
                    f"got {tuple(velocity.shape)} and expected "
                    f"{tuple(input_x.shape)}"
                )
            velocity = velocity.to(device=input_x.device, dtype=input_x.dtype)
            return 2.0 * velocity - input_x / _time_like(timestep, input_x)

        drift = beta_drift_at_x(x_t)
        x_t.grad = None
        torch.autograd.backward(drift, grad_tensors=adj_next)
        vjp_x = x_t.grad
        if vjp_x is None:
            raise RuntimeError("Failed to compute QAM adjoint VJP.")
        adjs[step - 1] = (adj_next + h * vjp_x).detach()

    return traj, torch.stack(adjs, dim=0)


def compute_qam_actor_loss(
    vf_fine: torch.Tensor,
    vf_base: torch.Tensor,
    adjs: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    skip_terminal: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the plain QAM actor loss on a fixed trajectory.

    This helper implements the step-wise adjoint matching objective

    ``|| 2 * (f_theta - f_beta) / sigma_t + sigma_t * adj_t ||^2``

    using velocities already evaluated on the same actor trajectory. It is a
    pure tensor helper: callers are responsible for producing ``vf_fine`` from
    the trainable actor, ``vf_base`` from frozen ``f_beta``, and ``adjs`` from
    :func:`compute_adjoint_states`.

    The noise schedule ``sigma_t`` is the diffusion coefficient of the QAM
    forward SDE (``da = (2 f - a / t) dt + sigma_t dB``), evaluated with the
    h-shift used by the official QAM code so it stays finite at both ends::

        sigma_t = sqrt(2 * (1 - t + h) / (t + h)),   h = 1 / W.

    This must match the drift ``2 f_beta - a / t`` that
    :func:`compute_adjoint_states` integrates; using ``sqrt(2 (1 - t))`` here
    would be inconsistent with that adjoint.

    The reduction matches official QAM: the squared residual is summed over the
    action dimensions and the flow-time dimension, then averaged over the batch
    (over the valid batch entries when ``loss_mask`` is given).

    Args:
        vf_fine: Trainable actor velocities, shape ``[W+1, B, ...]``.
        vf_base: Frozen behavior velocities, same shape as ``vf_fine``.
        adjs: Lean adjoint states, same shape as ``vf_fine``.
        loss_mask: Optional per-(time, batch) validity mask. Supported shapes
            are ``[W+1]`` / ``[W]`` (time), ``[B]`` (batch), or the 2-D
            ``[W+1, B]`` / ``[W, B]`` forms.
        skip_terminal: Whether to drop ``t=1`` from the objective. This should
            normally stay ``True`` for parity with official QAM, which only
            uses the positions ``0..W-1``.

    Returns:
        A pair ``(loss, metrics)``. Gradients flow only through ``vf_fine``.
    """
    if vf_fine.ndim < 2:
        raise ValueError(
            f"vf_fine must have shape [W+1, B, ...], got {tuple(vf_fine.shape)}"
        )
    _validate_same_shape("vf_base", vf_base, "vf_fine", vf_fine)
    _validate_same_shape("adjs", adjs, "vf_fine", vf_fine)
    if not vf_fine.is_floating_point():
        raise TypeError(f"vf_fine must be a floating point tensor, got {vf_fine.dtype}")
    if not vf_base.is_floating_point():
        raise TypeError(f"vf_base must be a floating point tensor, got {vf_base.dtype}")
    if not adjs.is_floating_point():
        raise TypeError(f"adjs must be a floating point tensor, got {adjs.dtype}")

    num_steps = vf_fine.shape[0] - 1
    if num_steps <= 0:
        raise ValueError(
            f"vf_fine must contain at least two time states, got {vf_fine.shape[0]}"
        )

    calc_dtype = (
        torch.float32
        if vf_fine.dtype in (torch.float16, torch.bfloat16)
        else vf_fine.dtype
    )
    fine = vf_fine.to(dtype=calc_dtype)
    base = vf_base.detach().to(device=vf_fine.device, dtype=calc_dtype)
    adj = adjs.detach().to(device=vf_fine.device, dtype=calc_dtype)

    h = 1.0 / num_steps
    times = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=vf_fine.device,
        dtype=calc_dtype,
    )
    if skip_terminal:
        fine = fine[:-1]
        base = base[:-1]
        adj = adj[:-1]
        times = times[:-1]

    # h-shifted QAM diffusion coefficient: finite at t=0 and t=1.
    sigma = torch.sqrt(2.0 * (1.0 - times + h) / (times + h))
    sigma = sigma.view(-1, *([1] * (fine.ndim - 1)))

    velocity_delta = fine - base
    residual = velocity_delta * (2.0 / sigma) + sigma * adj
    squared_residual = residual.square()

    # Official QAM reduction: sum over action dims and flow-time, mean over the
    # (valid) batch. ``squared_residual`` is [W_sel, B, *action_dims].
    selected_steps, batch_size = squared_residual.shape[:2]
    action_dims = tuple(range(2, squared_residual.ndim))
    per_step_sample = (
        squared_residual.sum(dim=action_dims) if action_dims else squared_residual
    )  # [W_sel, B]

    mask = _prepare_qam_loss_mask(
        loss_mask=loss_mask,
        selected_steps=selected_steps,
        batch_size=batch_size,
        full_num_steps=num_steps + 1,
        skip_terminal=skip_terminal,
        device=squared_residual.device,
        dtype=calc_dtype,
    )
    if mask is None:
        per_sample = per_step_sample.sum(dim=0)  # [B]
        loss = per_sample.mean()
        denominator = torch.tensor(
            float(batch_size), device=squared_residual.device, dtype=calc_dtype
        )
    else:
        per_sample = (per_step_sample * mask).sum(dim=0)  # [B]
        denominator = (mask.sum(dim=0) > 0).sum().clamp_min(1).to(calc_dtype)
        loss = per_sample.sum() / denominator

    full_mask = None
    if mask is not None:
        full_mask = mask.view(
            selected_steps, batch_size, *([1] * len(action_dims))
        ).broadcast_to(squared_residual.shape)

    with torch.no_grad():
        residual_abs = _masked_mean_for_qam(residual.detach().abs(), full_mask)
        delta_abs = _masked_mean_for_qam(velocity_delta.detach().abs(), full_mask)
        adj_abs = _masked_mean_for_qam(adj.detach().abs(), full_mask)
        metrics = {
            "actor/qam_loss": loss.detach(),
            "actor/qam_residual_abs": residual_abs,
            "actor/qam_velocity_delta_abs": delta_abs,
            "actor/qam_adj_abs": adj_abs,
            "actor/qam_sigma_min": sigma.detach().min(),
            "actor/qam_sigma_max": sigma.detach().max(),
            "actor/qam_valid_count": denominator.detach(),
        }

    return loss, metrics


def _prepare_qam_loss_mask(
    loss_mask: torch.Tensor | None,
    selected_steps: int,
    batch_size: int,
    full_num_steps: int,
    skip_terminal: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Normalize ``loss_mask`` to a ``[W_sel, B]`` float mask (or ``None``)."""
    if loss_mask is None:
        return None

    mask = loss_mask.to(device=device, dtype=dtype)

    if mask.ndim == 1:
        if mask.shape[0] == batch_size:
            mask = mask.view(1, batch_size).expand(selected_steps, batch_size)
        elif mask.shape[0] == full_num_steps:
            mask = mask[:-1] if skip_terminal else mask
            mask = mask.view(selected_steps, 1).expand(selected_steps, batch_size)
        elif mask.shape[0] == selected_steps:
            mask = mask.view(selected_steps, 1).expand(selected_steps, batch_size)
        else:
            raise ValueError(
                "loss_mask must have leading dimension W+1, selected W, or batch "
                f"B; got shape {tuple(loss_mask.shape)} for [W={selected_steps}, "
                f"B={batch_size}]"
            )
    elif mask.ndim == 2:
        if mask.shape[0] == full_num_steps:
            mask = mask[:-1] if skip_terminal else mask
        if mask.shape != (selected_steps, batch_size):
            raise ValueError(
                "loss_mask must be broadcastable to [W, B] = "
                f"[{selected_steps}, {batch_size}]; got {tuple(loss_mask.shape)}"
            )
    else:
        raise ValueError(
            f"loss_mask must be 1-D or 2-D, got shape {tuple(loss_mask.shape)}"
        )

    return mask.contiguous()


def _masked_mean_for_qam(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return values.mean()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)

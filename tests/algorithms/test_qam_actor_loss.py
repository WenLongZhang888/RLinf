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

from rlinf.algorithms.embodiment import compute_qam_actor_loss


def _reference_mask(
    loss_mask: torch.Tensor,
    selected_steps: int,
    batch_size: int,
    full_num_steps: int,
    skip_terminal: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = loss_mask.to(dtype=dtype)
    if mask.ndim == 1:
        if mask.shape[0] == batch_size:
            mask = mask.view(1, batch_size).expand(selected_steps, batch_size)
        elif mask.shape[0] == full_num_steps:
            mask = mask[:-1] if skip_terminal else mask
            mask = mask.view(selected_steps, 1).expand(selected_steps, batch_size)
        else:
            mask = mask.view(selected_steps, 1).expand(selected_steps, batch_size)
    else:
        if mask.shape[0] == full_num_steps:
            mask = mask[:-1] if skip_terminal else mask
    return mask.contiguous()


def _manual_qam_loss(
    vf_fine: torch.Tensor,
    vf_base: torch.Tensor,
    adjs: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    skip_terminal: bool = True,
) -> torch.Tensor:
    """Reference for the official-QAM sigma + reduction.

    sigma_t = sqrt(2 (1 - t + h) / (t + h)), h = 1 / W;
    loss = mean_over_valid_batch( sum_steps sum_action_dims residual^2 ).
    """
    num_steps = vf_fine.shape[0] - 1
    h = 1.0 / num_steps
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=vf_fine.dtype)
    if skip_terminal:
        fine, base, adj = vf_fine[:-1], vf_base[:-1], adjs[:-1]
        times = times[:-1]
    else:
        fine, base, adj = vf_fine, vf_base, adjs
    sigma = torch.sqrt(2.0 * (1.0 - times + h) / (times + h))
    sigma = sigma.view(-1, *([1] * (fine.ndim - 1)))
    residual = 2.0 * (fine - base) / sigma + sigma * adj
    squared = residual.square()
    action_dims = tuple(range(2, squared.ndim))
    per_step = squared.sum(dim=action_dims) if action_dims else squared  # [W, B]

    if loss_mask is None:
        return per_step.sum(dim=0).mean()

    selected_steps, batch_size = per_step.shape
    mask = _reference_mask(
        loss_mask, selected_steps, batch_size, num_steps + 1, skip_terminal,
        vf_fine.dtype,
    )
    per_sample = (per_step * mask).sum(dim=0)
    denom = (mask.sum(dim=0) > 0).sum().clamp_min(1).to(per_sample.dtype)
    return per_sample.sum() / denom


def test_compute_qam_actor_loss_matches_manual_formula_and_gradients():
    """QAM loss should match Eq. 21 (h-shifted sigma) and backprop only vf_fine."""
    torch.manual_seed(0)
    vf_fine = torch.randn(6, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    vf_base = torch.randn_like(vf_fine, requires_grad=True)
    adjs = torch.randn_like(vf_fine, requires_grad=True)

    loss, metrics = compute_qam_actor_loss(vf_fine, vf_base, adjs)
    expected = _manual_qam_loss(vf_fine, vf_base, adjs)

    assert torch.allclose(loss, expected)
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
    assert vf_fine.grad is not None
    assert vf_fine.grad.abs().sum() > 0
    assert vf_base.grad is None
    assert adjs.grad is None


def test_compute_qam_actor_loss_applies_time_mask_and_skips_terminal():
    """Masking should exclude invalid entries and the t=1 endpoint by default."""
    vf_fine = torch.ones(4, 2, 1, 1, dtype=torch.float64)
    vf_base = torch.zeros_like(vf_fine)
    adjs = torch.zeros_like(vf_fine)
    loss_mask = torch.tensor(
        [
            [True, False],
            [False, True],
            [True, True],
            [True, True],  # Terminal row should be ignored.
        ]
    )

    loss, metrics = compute_qam_actor_loss(
        vf_fine=vf_fine,
        vf_base=vf_base,
        adjs=adjs,
        loss_mask=loss_mask,
    )
    expected = _manual_qam_loss(vf_fine, vf_base, adjs, loss_mask=loss_mask)

    assert torch.allclose(loss, expected)
    # Both batch columns keep at least one valid step -> valid batch count = 2.
    assert torch.allclose(
        metrics["actor/qam_valid_count"],
        torch.tensor(2.0, dtype=loss.dtype, device=loss.device),
    )


def test_compute_qam_actor_loss_supports_batch_mask():
    """A [B] mask should select valid batch entries; loss averages over them."""
    vf_fine = torch.ones(3, 2, 2, 1, dtype=torch.float64)
    vf_base = torch.zeros_like(vf_fine)
    adjs = torch.zeros_like(vf_fine)
    loss_mask = torch.tensor([True, False])

    loss, metrics = compute_qam_actor_loss(
        vf_fine=vf_fine,
        vf_base=vf_base,
        adjs=adjs,
        loss_mask=loss_mask,
    )
    expected = _manual_qam_loss(vf_fine, vf_base, adjs, loss_mask=loss_mask)

    # Closed form: W=2, h=0.5 -> sigma = [sqrt(6), sqrt(2)]; 2 action elems each.
    times = torch.tensor([0.0, 0.5], dtype=vf_fine.dtype)
    sigma = torch.sqrt(2.0 * (1.0 - times + 0.5) / (times + 0.5))
    closed_form = (2.0 * ((2.0 / sigma) ** 2)).sum()  # only batch col 0 survives

    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, closed_form)
    assert torch.allclose(
        metrics["actor/qam_valid_count"],
        torch.tensor(1.0, dtype=loss.dtype, device=loss.device),
    )


def test_compute_qam_actor_loss_can_include_terminal():
    """With skip_terminal=False, t=1 is finite thanks to the h-shifted sigma."""
    vf_fine = torch.ones(2, 1, 1, 1, dtype=torch.float64)
    vf_base = torch.zeros_like(vf_fine)
    adjs = torch.zeros_like(vf_fine)

    loss, metrics = compute_qam_actor_loss(
        vf_fine=vf_fine,
        vf_base=vf_base,
        adjs=adjs,
        skip_terminal=False,
    )
    expected = _manual_qam_loss(vf_fine, vf_base, adjs, skip_terminal=False)

    # W=1, h=1 -> sigma_0 = sqrt(4) = 2, sigma_1 = sqrt(1) = 1.
    residual_t0 = 2.0 / torch.tensor(2.0, dtype=vf_fine.dtype)
    residual_t1 = 2.0 / torch.tensor(1.0, dtype=vf_fine.dtype)
    closed_form = residual_t0.square() + residual_t1.square()  # sum over steps, B=1

    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, closed_form)
    assert torch.allclose(
        metrics["actor/qam_sigma_min"],
        torch.tensor(1.0, dtype=loss.dtype, device=loss.device),
    )
    assert torch.allclose(
        metrics["actor/qam_sigma_max"],
        torch.tensor(2.0, dtype=loss.dtype, device=loss.device),
    )


def test_compute_qam_actor_loss_rejects_invalid_inputs():
    vf_fine = torch.zeros(3, 2, 1, 1)
    vf_base = torch.zeros_like(vf_fine)
    adjs = torch.zeros_like(vf_fine)

    with pytest.raises(ValueError, match="same shape"):
        compute_qam_actor_loss(vf_fine, vf_base[:-1], adjs)
    with pytest.raises(ValueError, match="at least two"):
        compute_qam_actor_loss(vf_fine[:1], vf_base[:1], adjs[:1])
    with pytest.raises(ValueError, match="loss_mask"):
        compute_qam_actor_loss(
            vf_fine,
            vf_base,
            adjs,
            loss_mask=torch.ones(5, 2, dtype=torch.bool),
        )

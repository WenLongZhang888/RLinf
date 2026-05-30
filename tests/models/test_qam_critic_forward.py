"""P2.1 unit tests: QAM critic forward components.

Covers the pure-CPU surface of P2.1:
    - MultiQHead shape sanity in QAM dimensions (pooled_z=2048, action=35).
    - _obs_processor_for_qam: replay → OpenPI observation/* key remapping.
    - _pool_prefix_for_qam: pooling shape and correctness across modes.

The full qam_q_forward / qam_velocity_forward end-to-end (PaliGemma +
input_transform pipeline) is exercised by smoke runs (docs/lwd.md P5),
not unit tests, because it requires loading the OpenPI checkpoint.

NOTE: f_β handling has moved out of OpenPi0ForRLActionPrediction. It is
now built by the QAM actor worker as a peer model instance, following
the SAC target_model pattern (fsdp_sac_policy_worker.py). The previous
snapshot_f_beta / _swap_to_fbeta tests have been removed along with
that mechanism.
"""
import pytest
import torch

# openpi is required by the module-level imports of openpi_action_model.
pytest.importorskip("openpi", reason="openpi not installed")

from rlinf.models.embodiment.modules.q_head import MultiQHead  # noqa: E402
from rlinf.models.embodiment.openpi.openpi_action_model import (  # noqa: E402
    OpenPi0ForRLActionPrediction,
)


# ============ minimal mock self for unbound-method calls ============


class _MockConfig:
    def __init__(
        self,
        config_name: str = "pi05_libero",
        num_images_in_input: int = 2,
        qam_pool_mode: str = "mean_token",
    ):
        self.config_name = config_name
        self.num_images_in_input = num_images_in_input
        self.qam_pool_mode = qam_pool_mode


class _MockSelf:
    def __init__(self, **kwargs):
        self.config = _MockConfig(**kwargs)


# ============ MultiQHead in QAM shapes ============


def test_multi_q_head_qam_output_shape():
    """q_head_qam(z [B, 2048], a [B, 35]) → [B, num_q_heads]."""
    q_head = MultiQHead(
        hidden_size=2048,
        action_feature_dim=5 * 7,  # H=5, action_env_dim=7
        hidden_dims=[512, 512],
        num_q_heads=2,
        output_dim=1,
        train_action_encoder=False,
    )
    B = 4
    z = torch.randn(B, 2048)
    a = torch.randn(B, 35)
    q = q_head(z, a)
    assert q.shape == (B, 2), f"expected (B, 2), got {q.shape}"


def test_multi_q_head_backward_flows_to_action():
    """Critic gradient w.r.t. action must exist — QAM adjoint depends on it."""
    q_head = MultiQHead(
        hidden_size=2048,
        action_feature_dim=35,
        hidden_dims=[512, 512],
        num_q_heads=2,
        output_dim=1,
        train_action_encoder=False,
    )
    z = torch.randn(4, 2048)
    a = torch.randn(4, 35, requires_grad=True)
    q = q_head(z, a)
    q.sum().backward()
    assert a.grad is not None
    assert a.grad.abs().sum() > 0
    head_grads = [p.grad for p in q_head.parameters() if p.requires_grad]
    assert all(g is not None for g in head_grads)


# ============ _obs_processor_for_qam ============


def _fake_replay_obs(B: int = 3, wrist: bool = True, extra: bool = False):
    obs = {
        "main_images": torch.zeros(B, 3, 224, 224, dtype=torch.uint8),
        "states": torch.zeros(B, 8),
        "tokenized_prompt": torch.randint(0, 30000, (B, 48), dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(B, 48, dtype=torch.bool),
    }
    obs["wrist_images"] = (
        torch.zeros(B, 3, 224, 224, dtype=torch.uint8) if wrist else None
    )
    obs["extra_view_images"] = (
        torch.zeros(B, 1, 3, 224, 224, dtype=torch.uint8) if extra else None
    )
    return obs


def test_obs_processor_for_qam_libero_remapping():
    """LIBERO replay obs → expected observation/* layout with tokenized prompt."""
    mock = _MockSelf(config_name="pi05_libero")
    obs = _fake_replay_obs(B=3, wrist=True)
    out = OpenPi0ForRLActionPrediction._obs_processor_for_qam(mock, obs)

    assert out["observation/image"] is obs["main_images"]
    assert out["observation/wrist_image"] is obs["wrist_images"]
    assert out["observation/state"] is obs["states"]
    assert out["tokenized_prompt"] is obs["tokenized_prompt"]
    assert out["tokenized_prompt_mask"] is obs["tokenized_prompt_mask"]
    # task_descriptions absent — P1 pops it during transition collection.
    assert "task_descriptions" not in out
    # No "prompt" key → input_transform takes the not-first-process branch.
    assert "prompt" not in out


def test_obs_processor_for_qam_skips_optional_views_when_none():
    """When wrist/extra views are None, their observation/* keys are absent."""
    mock = _MockSelf(config_name="pi05_libero")
    obs = _fake_replay_obs(B=2, wrist=False, extra=False)
    out = OpenPi0ForRLActionPrediction._obs_processor_for_qam(mock, obs)
    assert "observation/wrist_image" not in out
    assert "observation/extra_view_image" not in out


def test_obs_processor_for_qam_calvin_splits_state():
    """Calvin uses split state keys (ee_pos / ee_rot / gripper)."""
    mock = _MockSelf(config_name="pi05_calvin")
    obs = _fake_replay_obs(B=2)
    out = OpenPi0ForRLActionPrediction._obs_processor_for_qam(mock, obs)
    assert "observation/state_ee_pos" in out
    assert "observation/state_ee_rot" in out
    assert "observation/state_gripper" in out
    assert "observation/state" not in out


# ============ _pool_prefix_for_qam ============


@pytest.mark.parametrize("mode", ["mean_token", "last_token", "first_token"])
def test_pool_prefix_shape_pi05(mode):
    """Pool [B, 968, 2048] → [B, 2048] regardless of mode for pi05."""
    mock = _MockSelf(config_name="pi05_libero", qam_pool_mode=mode)
    prefix = torch.randn(4, 968, 2048)
    pooled = OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)
    assert pooled.shape == (4, 2048)


def test_pool_prefix_shape_pi0():
    """Pool [B, 816, 1024] → [B, 1024] for pi0 (different all_token_length)."""
    mock = _MockSelf(config_name="pi0_libero", qam_pool_mode="mean_token")
    prefix = torch.randn(2, 816, 1024)
    pooled = OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)
    assert pooled.shape == (2, 1024)


def test_pool_prefix_first_token_correctness():
    """first_token mode must equal prefix_output[:, 0, :]."""
    mock = _MockSelf(config_name="pi05_libero", qam_pool_mode="first_token")
    prefix = torch.randn(2, 968, 2048)
    pooled = OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)
    assert torch.allclose(pooled, prefix[:, 0, :])


def test_pool_prefix_last_token_correctness():
    """last_token mode must equal prefix_output[:, -1, :]."""
    mock = _MockSelf(config_name="pi05_libero", qam_pool_mode="last_token")
    prefix = torch.randn(2, 968, 2048)
    pooled = OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)
    assert torch.allclose(pooled, prefix[:, -1, :])


def test_pool_prefix_mean_token_correctness_pi05():
    """mean_token mode averages over (image tokens used) + (language tokens)."""
    mock = _MockSelf(
        config_name="pi05_libero",
        num_images_in_input=2,
        qam_pool_mode="mean_token",
    )
    prefix = torch.randn(2, 968, 2048)
    pooled = OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)
    # With 2 images: 256*2 image tokens + 0 padding + 200 lang tokens = 712 used.
    selected = torch.cat(
        [prefix[:, : 256 * 2, :], prefix[:, -200:, :]],
        dim=1,
    )
    assert torch.allclose(pooled, selected.mean(dim=1))


def test_pool_prefix_invalid_mode_raises():
    """Unknown qam_pool_mode raises ValueError with the field name."""
    mock = _MockSelf(config_name="pi05_libero", qam_pool_mode="bogus")
    prefix = torch.randn(2, 968, 2048)
    with pytest.raises(ValueError, match="qam_pool_mode"):
        OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)


def test_pool_prefix_unknown_config_name_raises():
    """Unknown config_name raises ValueError with the field name."""
    mock = _MockSelf(config_name="totally_unknown", qam_pool_mode="mean_token")
    prefix = torch.randn(2, 968, 2048)
    with pytest.raises(ValueError, match="config_name"):
        OpenPi0ForRLActionPrediction._pool_prefix_for_qam(mock, prefix)


# ============ QAM ↔ OpenPI flow-time / velocity convention ============


def test_qam_to_openpi_flow_time_is_mirror():
    """QAM w↔OpenPI t mirror: t = 1 - w. Pinned because the velocity field
    is trained in OpenPI's convention but called from QAM coordinates."""
    w = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    t = OpenPi0ForRLActionPrediction._qam_to_openpi_flow_time(w)
    assert torch.allclose(t, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_qam_to_openpi_flow_time_preserves_dtype_and_shape():
    """Mirror must be dtype/shape-transparent so callers don't need to recast."""
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        w = torch.tensor([0.1, 0.9], dtype=dtype)
        t = OpenPi0ForRLActionPrediction._qam_to_openpi_flow_time(w)
        assert t.dtype == dtype
        assert t.shape == w.shape

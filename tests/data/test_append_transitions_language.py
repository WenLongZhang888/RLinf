"""P1 unit tests: replay must preserve task language as tokenized tensors."""
import numpy as np
import pytest
import torch

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult


def _make_obs(batch_size: int, with_lang_str: bool = True):
    obs = {
        "states": torch.zeros(batch_size, 8),
        "main_images": torch.zeros(batch_size, 3, 64, 64, dtype=torch.uint8),
    }
    if with_lang_str:
        obs["task_descriptions"] = [f"task_{i}" for i in range(batch_size)]
    return obs


def _fake_language(batch_size: int, seq_len: int = 48):
    return {
        "tokenized_prompt": torch.randint(
            0, 30000, (batch_size, seq_len), dtype=torch.long
        ),
        "tokenized_prompt_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


def test_append_drops_task_descriptions_and_attaches_tokens():
    rr = EmbodiedRolloutResult(max_episode_length=4)
    rr.append_transitions(
        curr_obs=_make_obs(2),
        next_obs=_make_obs(2),
        curr_language=_fake_language(2),
        next_language=_fake_language(2),
    )

    # Strings gone.
    assert "task_descriptions" not in rr.curr_obs[0]
    assert "task_descriptions" not in rr.next_obs[0]
    # Tensors attached.
    assert rr.curr_obs[0]["tokenized_prompt"].shape == (2, 48)
    assert rr.curr_obs[0]["tokenized_prompt_mask"].dtype == torch.bool
    assert rr.next_obs[0]["tokenized_prompt"].shape == (2, 48)


def test_curr_and_next_language_are_independent():
    """After auto-reset next_obs carries a new task — must be tokenized separately."""
    rr = EmbodiedRolloutResult(max_episode_length=4)
    curr_lang = _fake_language(2)
    next_lang = _fake_language(2)
    # Force them to differ so we can detect cross-talk.
    next_lang["tokenized_prompt"] = curr_lang["tokenized_prompt"] + 1

    rr.append_transitions(
        curr_obs=_make_obs(2),
        next_obs=_make_obs(2),
        curr_language=curr_lang,
        next_language=next_lang,
    )
    assert not torch.equal(
        rr.curr_obs[0]["tokenized_prompt"],
        rr.next_obs[0]["tokenized_prompt"],
    )


def test_missing_language_does_not_break_existing_paths():
    """Models like OpenVLA/GR00T don't need tokenized prompt — must still work."""
    rr = EmbodiedRolloutResult(max_episode_length=4)
    rr.append_transitions(
        curr_obs=_make_obs(2, with_lang_str=False),
        next_obs=_make_obs(2, with_lang_str=False),
        curr_language=None,
        next_language=None,
    )
    assert "tokenized_prompt" not in rr.curr_obs[0]
    assert "task_descriptions" not in rr.curr_obs[0]


def test_to_trajectory_stacks_language_tensors():
    """Stacked obs dict should contain tokenized_prompt of shape [T, B, L]."""
    rr = EmbodiedRolloutResult(max_episode_length=4)
    for _ in range(3):
        rr.append_transitions(
            curr_obs=_make_obs(2),
            next_obs=_make_obs(2),
            curr_language=_fake_language(2),
            next_language=_fake_language(2),
        )
    # Minimal tensor fields so to_trajectory doesn't trip on empty stacks.
    rr.actions = [torch.zeros(2, 7) for _ in range(3)]
    rr.versions = [torch.zeros(2, 1) for _ in range(3)]

    traj = rr.to_trajectory()
    assert traj.curr_obs["tokenized_prompt"].shape == (3, 2, 48)
    assert traj.next_obs["tokenized_prompt"].shape == (3, 2, 48)
    assert traj.curr_obs["tokenized_prompt_mask"].dtype == torch.bool


@pytest.mark.skipif(
    pytest.importorskip("openpi", reason="openpi not installed") is None,
    reason="openpi tokenizer not available",
)
def test_paligemma_tokenizer_produces_expected_shape():
    """Smoke test that the real openpi tokenizer gives [L] arrays of int."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    tok = PaligemmaTokenizer(max_len=48)
    tokens, mask = tok.tokenize("pick up the can", state=None)
    assert tokens.shape == (48,)
    assert mask.shape == (48,)
    assert mask.dtype == np.bool_

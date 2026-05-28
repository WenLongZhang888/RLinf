# QAM 实现

## P1：保留 task language 进 replay

### 解决的问题

QAM critic 训练时要在 replay 上重算 `z_t = pool(PaliGemma_prefix(s_t))`，但原代码 `EmbodiedRolloutResult.append_transitions` 把 `task_descriptions` 字符串直接 pop 丢掉，replay 里没有语言条件，PaliGemma 没法跑。

### 思路

**rollout 自己 tokenize 是给当下推理用的，用完即丢，不回传**。所以 env_worker 在每一帧收集 transition 时**自己再 tokenize 一次**写进 replay，把 `task_descriptions: list[str]` 换成等价的 `tokenized_prompt + tokenized_prompt_mask` 张量。

### 改了哪些文件

| 文件 | 改动 |
|---|---|
| `rlinf/data/embodied_io_struct.py` | `append_transitions` 签名加 `curr_language` / `next_language` 两个可选 dict；pop 掉 string 后 attach tokenized 张量 |
| `rlinf/workers/env/env_worker.py` | 新增 `_maybe_init_language_tokenizer`（`init_worker` 末尾调用，仅 `collect_transitions=True` 且 `model_type ∈ {openpi, openpi_pi05}` 时载入 PaliGemma tokenizer） |
| `rlinf/workers/env/env_worker.py` | 新增 `_tokenize_language(strings)`，返回 dict 或 None |
| `rlinf/workers/env/env_worker.py` | chunk-step 调用 `append_transitions` 处分别对 `curr_obs` / `next_obs` 的 `task_descriptions` 各 tokenize 一次，传入 |
| `tests/data/test_append_transitions_language.py` | 新增 5 个单测 |

### 数据契约（每帧 `append_transitions` 写进 list 的 dict）

```
{
  main_images:           Tensor[B, ...]
  states:                Tensor[B, ...]
  # task_descriptions 已被 pop
  tokenized_prompt:      Tensor[B, 48]  long   ← 新增
  tokenized_prompt_mask: Tensor[B, 48]  bool   ← 新增
}
```

`to_trajectory()` 后 stack 成 `[T, B, 48]` 进 replay。

### 关键设计点

- **每帧都 tokenize**，不是 episode 结束才做。原因：QAM 训练 sample 的是单帧 transition，每帧必须自带语言；且 auto-reset 后 next_obs 可能是新任务，必须 `curr_language` / `next_language` 分别 tokenize。
- **PaliGemma tokenizer 在 env_worker 进程内 lazy-init**：CPU 上 SentencePiece，单 prompt < 1ms，可忽略。
- **`max_len=48`** 与 OpenPI 官方一致；可通过 `actor.model.paligemma_max_token_len` 配置覆盖。
- **`state=None`**：本仓库 Pi05 LIBERO config 全部 `discrete_state_input=False`。

### 向后兼容
非 OpenPI 路径完全不动，三道闸门保护：
1. `collect_transitions=False` → tokenizer 不载入；
2. `model_type` 非 openpi → tokenizer 不载入；
3. `curr_language=None` → `append_transitions` 内 attach 跳过。

→ OpenVLA / GR00T / MLP / SAC / PPO / GRPO 路径不受影响。

### 验证
```bash
pytest tests/data/test_append_transitions_language.py -v
```

### 下一步（P2）
在 `OpenPi0ForRLActionPrediction` 加 `qam_q_forward`：从 `curr_obs` 取 `tokenized_prompt` + images + states → `_build_prefix_cache` → pool → `z_t` → `Q_head(z_t, action_chunk)`。

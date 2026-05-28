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

---

## P2.1：critic 前向（`qam_q_forward`）

### P2.1 解决的问题

P1 把 language 放进 replay 了，但还没人会算 `Q_φ(s, a)`。QAM 训练 step 里 critic 会被调三次：

1. current Q：`Q_φ(s_t, a_t)`（replay 里的 action）
2. target Q：`Q_φ_target(s_{t+1}, a')`（target policy 采的 action）
3. actor adjoint：`∇_a Q_φ(s, a¹)`（flow 终端 action，P3 才用）

三次都要走同一个入口函数 —— `qam_q_forward`。**P2.1 就是把这个入口建好**，让模型能算 Q。**不做 loss、不做 target 网络、不做 actor 更新**。

### 决策：方案 A — plain QAM

参考 [QAM 官方代码](https://github.com/ColinQiyangLi/qam/blob/main/agents/qam.py) 和 LWD §III.C：

- ✅ trainable flow `f_θ` + 冻结 reference `f_β` + adjoint matching
- ❌ 不加 `edit_actor`（QAM_EDIT 才需要）
- ❌ 不加 `actor_fast` residual（用 non-residual 分支：`f_δ = f_θ - f_β`）
- ❌ 不加 BC flow-matching 持续训（`f_β` 直接用 SFT checkpoint）

跟 LWD 论文公式 (9) 完全对齐：

```text
L_QAM(θ) = E[ ∫₀¹ ‖(f_θ - f_β) · 2/σ_w + σ_w · g̃_w‖² dw ]
σ_w = √(2(1-w)),   g̃_1 = -∇_a [Q_φ(z_t, a¹) / λ]
```

但 P2.1 **只做 `Q_φ` 那一块**，actor 那块全部丢 P3。

### 改了哪些位置（全在 `rlinf/models/embodiment/openpi/openpi_action_model.py`）

| # | 改动 | 作用 |
| --- | --- | --- |
| 1 | `OpenPi0Config` 加 `use_qam` / `qam_num_q_heads` / `qam_q_hidden_dims` / `qam_pool_mode` | 配置层 |
| 2 | `__init__` 里 `if use_qam: self.q_head_qam = MultiQHead(...)` + 与 `use_dsrl` 互斥校验 | 实例化 critic 网络 |
| 3a | 新增 `_obs_processor_for_qam(replay_obs)` | replay 格式 → OpenPI `observation/*` 格式（直接消费 `tokenized_prompt`，不再注入字符串） |
| 3b | 新增 `_pool_prefix_for_qam(prefix_output)` | PaliGemma `[B, 968, 2048]` → `[B, 2048]`（沿用 `get_value_from_vlm` 的 mask 规则） |
| 4 | 新增 `qam_q_forward(obs, actions, detach_vlm=True, ...)` | **核心入口**，串完整 pipeline |
| 5 | `forward()` dispatch 加 `ForwardType.QAM_Q` 分支 | 让外面通过 `model.forward(forward_type=QAM_Q, ...)` 调到它 |

### 网络结构（`q_head_qam = MultiQHead`）

```text
state_features  [B, 2048] ─┐
                            ├─ concat → [B, 2083]
action_features [B, 35]   ─┘                │
                                            ▼
                                  Linear(2083, 512) + LN + tanh
                                  Linear(512, 512)  + LN + tanh
                                  Linear(512, 1)
                                            │
                                            ▼
                                       Q   [B, 1]
                                    (这是一个 head)

MultiQHead = 2 个上面这样的 head 并行 → concat → [B, 2]
```

- `2048` = PaliGemma π0.5 隐藏维
- `35 = action_horizon * action_env_dim = 5 * 7`（LIBERO chunk=5, 7 维 action）
- 2 head ensemble：target 侧通常取 `min(Q_1, Q_2)` 防过估

### `qam_q_forward` 流水线（6 步）

```text
obs (replay dict: main_images, states, tokenized_prompt, mask, ...)
    │
    ▼  ① _obs_processor_for_qam       键重排（main_images → observation/image）
    ▼  ② input_transform              ★ 真实数据处理：image resize/normalize
    ▼  ③ precision_processor          搬到 GPU、contiguous
    ▼  ④ _model.Observation.from_dict 包成 OpenPI dataclass
    ▼  ⑤ _preprocess_observation      拆成 (images, img_masks, lang_tokens, lang_masks, state)
    ▼  ⑥ _build_prefix_cache          PaliGemma forward → prefix_output [B, 968, 2048]
    ▼     prefix_output.detach()       ★ 切断梯度回 PaliGemma（VLM 全程冻结）
    ▼     _pool_prefix_for_qam        pool → z_t [B, 2048]

actions [B, 5, 7] → reshape → [B, 35] → .to(device, dtype)

q_head_qam(z_t, actions) → [B, 2]   ← 返回
```

> ①③④ 是纯格式/搬运/打包；② 是真正改张量的（image 224×224 归一化等）；⑤⑥ 是跑 PaliGemma。

### P2.1 关键设计点

- **`detach_vlm=True` 默认**：plain QAM 要求 PaliGemma 全程冻结。`detach()` 是工程双保险 —— 即使将来谁不小心把 PaliGemma 的 `requires_grad` 设错，梯度也回不去。
- **`actions` 由外部传入**：critic 自己不知道当前是 current Q 还是 target Q 还是 actor adjoint —— 调用方负责喂对应的 action。这让同一函数三个场景共用。
- **mutex with DSRL**：同一个 OpenPI 模型不能同时开 `use_dsrl` 和 `use_qam`，`__init__` 里直接 raise。
- **不加新模块**：`MultiQHead` 已经在 [rlinf/models/embodiment/modules/q_head.py:120](../rlinf/models/embodiment/modules/q_head.py#L120) 现成，直接复用。

### 还没做的事（明确边界）

- ❌ critic TD loss / target critic 软更新 / critic optimizer —— P2.4 / worker 侧。
- ❌ 冻结 `f_β` 快照 / `qam_velocity_forward` / adjoint state 反向积分 —— P3。
- ❌ `EmbodiedQAMFSDPPolicy` worker —— P3。
- ❌ YAML 配置 —— 等 worker 搭好再写。

### 下一步（P2.2）

写一个最小单测：build 一个 mock OpenPI 模型 with `use_qam=True`，喂 fake obs + action，验证：

1. `model.forward(forward_type=QAM_Q, obs=..., actions=...)` 输出 shape = `[B, num_q_heads]`
2. backward 后 `q_head_qam.parameters()` 有 grad、`paligemma_with_expert` 的所有 param `.grad is None`
3. obs 缺 `tokenized_prompt` 时报清晰错误

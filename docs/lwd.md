# QAM 实现

本仓库在 OpenPI π0.5 + LIBERO 上落地 [QAM (Q-learning with Adjoint Matching)](https://github.com/ColinQiyangLi/qam) 的 plain 变体（对应 LWD 论文 §III.C，关闭官方 QAM 仓库的 `edit_scale` / `fql_alpha`）。本文档按阶段记录已完成的实现与遗留事项。

## 总体路线

```text
P1  ✅ 把 task language 保留进 replay
P2  critic 前向 (Q_φ)
    P2.1 ✅ qam_q_forward + Q-head（不含 loss / target / optim）
    P2.2 ✅ 单元测试
    P2.4 ⏳ TD loss + target critic + critic optimizer（worker 侧）
P3  actor 训练
    P3.1 ✅ 冻结 f_β 快照
    P3.2 ⏳ qam_velocity_forward（同时算 f_θ 和 f_β 的 velocity）
    P3.3 ⏳ adjoint state 反向积分（最难）
    P3.4 ⏳ QAM actor loss
    P3.5 ⏳ EmbodiedQAMFSDPPolicy worker
P5  ⏳ YAML + smoke + LIBERO eval
```

LWD 论文公式 (9) —— V1 实现按此对齐：

```text
L_QAM(θ) = E[ ∫₀¹ ‖(f_θ - f_β) · 2/σ_w + σ_w · g̃_w‖² dw ]
σ_w = √(2(1-w)),   g̃_1 = -∇_a [Q_φ(z_t, a¹) / λ]
```

---

## P1：保留 task language 进 replay

### P1 问题

QAM critic 训练时需要在 replay 上重算 `z_t = pool(PaliGemma_prefix(s_t))`，但原代码 `EmbodiedRolloutResult.append_transitions` 把 `task_descriptions` 字符串直接 pop 丢掉。

### P1 思路

rollout 自己 tokenize 是给当下推理用的、用完即丢、不回传。所以 env_worker 在每帧收集 transition 时**自己再 tokenize 一次写进 replay**，把 `task_descriptions: list[str]` 换成等价的 `tokenized_prompt + tokenized_prompt_mask` 张量（list[str] 无法走 `stack_list_of_dict_tensor`，必须换成 tensor 形式）。

### P1 文件改动

| 文件 | 改动 |
| --- | --- |
| `rlinf/data/embodied_io_struct.py` | `append_transitions` 加 `curr_language` / `next_language` 两个可选 dict；pop 字符串后 attach tokenized 张量 |
| `rlinf/workers/env/env_worker.py` | 新增 `_maybe_init_language_tokenizer`（`init_worker` 末尾调用，仅 `collect_transitions=True` 且 `model_type == "openpi"` 时载入 PaliGemma tokenizer） |
| `rlinf/workers/env/env_worker.py` | 新增 `_tokenize_language(strings)`，返回 dict 或 None |
| `rlinf/workers/env/env_worker.py` | chunk-step 调用 `append_transitions` 处对 `curr_obs` / `next_obs` 分别 tokenize |
| `tests/data/test_append_transitions_language.py` | 5 个单测 |

### P1 数据契约

`append_transitions` 写进 list 的每个 obs dict：

```text
{
  main_images:           Tensor[B, ...]
  states:                Tensor[B, ...]
  # task_descriptions 已被 pop
  tokenized_prompt:      Tensor[B, 48]  long   ← 新增
  tokenized_prompt_mask: Tensor[B, 48]  bool   ← 新增
}
```

`to_trajectory()` 后 stack 成 `[T, B, 48]` 进 replay。

### P1 设计决策

- **每帧都 tokenize**，不是 episode 结束才做。原因：QAM 训练 sample 单帧 transition，每帧必须自带语言；auto-reset 后 next_obs 可能是新任务，必须 `curr_language` / `next_language` 分别 tokenize。
- **tokenizer 在 env_worker 进程内 lazy-init**：CPU 上 SentencePiece，单 prompt < 1ms。
- **`max_len=48`** 与 OpenPI 官方一致。
- **`state=None`**：本仓库 Pi05 LIBERO config 全部 `discrete_state_input=False`。
- **三道闸门保护非 OpenPI 路径**：`collect_transitions=False` → tokenizer 不载入；`model_type` 非 openpi → tokenizer 不载入；`curr_language=None` → attach 跳过。OpenVLA / GR00T / MLP / SAC / PPO / GRPO 不受影响。

### P1 验证

```bash
pytest tests/data/test_append_transitions_language.py -v
```

---

## P2.1：critic 前向（`qam_q_forward`）

### P2.1 问题

P1 把 language 放进 replay 了，但还没人会算 `Q_φ(s, a)`。QAM 训练 step 里 critic 会被调三次：

1. **current Q**：`Q_φ(s_t, a_t)`（replay 里的 action）
2. **target Q**：`Q_φ_target(s_{t+1}, a')`（target policy 采的 action）
3. **actor adjoint**：`∇_a Q_φ(s, a¹)`（flow 终端 action，P3 才用）

三次都走同一个入口函数。**P2.1 只把这个入口建好**，不做 loss、不做 target、不做 actor 更新。

### P2.1 文件改动（全在 `rlinf/models/embodiment/openpi/openpi_action_model.py`）

| # | 改动 | 作用 |
| --- | --- | --- |
| 1 | `OpenPi0Config` 加 `use_qam` / `qam_num_q_heads` / `qam_q_hidden_dims` / `qam_pool_mode` | 配置层 |
| 2 | `__init__` 里 `if use_qam: self.q_head_qam = MultiQHead(...)` + 与 `use_dsrl` 互斥校验 | 实例化 critic 网络 |
| 3a | 新增 `_obs_processor_for_qam(replay_obs)` | replay 格式 → OpenPI `observation/*` 格式（直接消费 `tokenized_prompt`） |
| 3b | 新增 `_pool_prefix_for_qam(prefix_output)` | PaliGemma `[B, 968, 2048]` → `[B, 2048]` |
| 4 | 新增 `qam_q_forward(obs, actions, detach_vlm=True, ...)` | **核心入口** |
| 5 | `forward()` dispatch 加 `ForwardType.QAM_Q` 分支 | 让外面通过 `forward(forward_type=QAM_Q)` 调到它 |

### P2.1 网络结构（`q_head_qam = MultiQHead`）

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
                                       Q   [B, 1]   ← 单 head

MultiQHead = 2 个上面这样的 head 并行 → concat → [B, 2]
```

- `2048` = PaliGemma π0.5 隐藏维
- `35 = action_horizon * action_env_dim = 5 * 7`（LIBERO chunk=5, 7 维 action）
- 2 head ensemble：target 侧通常取 `min(Q_1, Q_2)` 防过估

### P2.1 流水线（6 步）

```text
obs (replay dict: main_images, states, tokenized_prompt, mask, ...)
    │
    ▼  ① _obs_processor_for_qam       键重排（main_images → observation/image）
    ▼  ② input_transform              ★ 真实数据处理：image resize/normalize
    ▼  ③ precision_processor          搬到 GPU、contiguous
    ▼  ④ _model.Observation.from_dict 包成 OpenPI dataclass
    ▼  ⑤ _preprocess_observation      拆成 (images, img_masks, lang_tokens, lang_masks, state)
    ▼  ⑥ _build_prefix_cache          PaliGemma forward → prefix_output [B, 968, 2048]
    ▼     prefix_output.detach()       ★ 切断梯度回 PaliGemma
    ▼     _pool_prefix_for_qam        pool → z_t [B, 2048]

actions [B, 5, 7] → reshape → [B, 35] → .to(device, dtype)

q_head_qam(z_t, actions) → [B, 2]   ← 返回
```

> ①③④ 是纯格式/搬运/打包；② 是真正改张量的（image 224×224 归一化）；⑤⑥ 是跑 PaliGemma。

### P2.1 设计决策

- **`detach_vlm=True` 默认**：plain QAM 全程冻结 PaliGemma。`detach()` 是工程双保险。
- **`actions` 由外部传入**：critic 自己不知道当前是 current Q / target Q / actor adjoint，调用方负责喂对应 action。同一函数三场景共用。
- **mutex with DSRL**：同一 OpenPI 模型不能同时开 `use_dsrl` 和 `use_qam`，`__init__` raise。
- **不加新模块**：`MultiQHead` 已在 [rlinf/models/embodiment/modules/q_head.py:120](../rlinf/models/embodiment/modules/q_head.py#L120) 现成，直接复用。

---

## P2.2：critic 前向单元测试

### P2.2 范围

只测 P2.1 的**纯 CPU 静态契约**（不加载真 OpenPI checkpoint）：

| 测试族 | 验证 |
| --- | --- |
| `MultiQHead` | shape `(B, 2)`；critic 梯度能回到 action（P3 adjoint 前置） |
| `_obs_processor_for_qam` | LIBERO/Calvin 键映射；wrist/extra view 缺失时不报错；不重新注入 `prompt` |
| `_pool_prefix_for_qam` | mean/first/last token mode 的 shape 和数值正确性；非法 mode / config_name 报错 |

完整 `qam_q_forward` 端到端（含 PaliGemma 前向）放到 P5 smoke 验证。

### P2.2 验证

```bash
pytest tests/models/test_qam_critic_forward.py -v
```

---

## P3.1：冻结 `f_β` 快照

### P3.1 问题

QAM actor 的损失里有 `f_δ = f_θ - f_β`。其中：

- `f_θ` = **可训练的** action expert（就是现有 `paligemma_with_expert.gemma_expert`，继续 in-place 训）
- `f_β` = **冻结的** reference action expert（SFT checkpoint 加载完后**克隆一份**，全程 `requires_grad=False`）

没有 `f_β` 的话 P3.2 的 `qam_velocity_forward` 无法同时算出 `(vf_fine, vf_base)`，整个 adjoint loss 无源头。

### P3.1 文件改动

| 文件 | 改动 |
| --- | --- |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | `__init__` 末尾加 `self._f_beta_paligemma_with_expert = None` 占位 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 新增 `snapshot_f_beta()` 方法 + `has_f_beta_snapshot` property |
| `tests/models/test_qam_critic_forward.py` | 加 `_MockExpertHolder` + 4 个单测 |

### P3.1 实现细节

`snapshot_f_beta()` 的核心动作：

```text
copy.deepcopy(self.paligemma_with_expert)
  → 所有参数 requires_grad_(False)
  → .eval()
  → self.add_module("_f_beta_paligemma_with_expert", snapshot)
```

**为什么 deepcopy 整个 `paligemma_with_expert` 而不是只克隆 `gemma_expert`**：
克隆整个对象（含 VLM 引用）保证 forward 路径与原版完全一致，工程上最简单。代价是 VLM 多占 ~6GB bf16（pi05 上），H200 80GB 显存够。如果将来要省内存，可以只克隆 expert 部分 + 周边小投影模块（`action_in_proj` 等 6 个），但 V1 不做。

### P3.1 设计决策

- **调用时机**：必须在 **SFT checkpoint 加载完之后** 且 **FSDP wrap 之前**。worker 侧（P3.5）的 `setup_model_and_optimizer` 负责。
- **`add_module` 注册**：让 PyTorch 把它当 child module，`.to(device)` / `.eval()` / state_dict 自动包含。代价是 ckpt 文件多 ~6GB；V1 接受这个代价。
- **idempotent 校验**：第二次调用 raise，防止误用导致 `f_β` 被覆盖。
- **`has_f_beta_snapshot` property**：P3.2 forward 之前 assert 这个为 True，给出清晰错误。

### P3.1 还没做的事

- ❌ 调用 `snapshot_f_beta()` —— P3.5 worker 侧。
- ❌ 用 `f_β` 算 velocity —— P3.2 `qam_velocity_forward`。
- ❌ 显存优化（只克隆 expert）—— V2。
- ❌ 排除 `_f_beta_*` 出 state_dict —— V2。

### P3.1 验证

```bash
pytest tests/models/test_qam_critic_forward.py -v -k snapshot
```

---

## 下一步：P3.2 `qam_velocity_forward`

给定 `(obs, x_t, t)`，分别用 `f_θ`（trainable）和 `f_β`（frozen snapshot）算两个 velocity：

```python
vf_fine = paligemma_with_expert.expert(prefix_kv, x_t, t)        # f_θ, gradient on
vf_base = _f_beta_paligemma_with_expert.expert(prefix_kv, x_t, t) # f_β, frozen
return vf_fine, vf_base
```

这是 P3.3 adjoint matching loss `‖(vf_fine - vf_base) * 2/σ + σ * adj‖²` 的左半部分。

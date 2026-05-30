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
    P3.2 ✅ qam_velocity_forward（单模型 velocity；worker 持有两份模型实例）
    P3.3 ✅ compute_adjoint_states 反向积分 + 单测（含 state-dependent toy）
    P3.4 🟡 QAM actor loss ✅（σ 已对齐官方 h-shift）/ 前向 SDE sampler + objective ⏳
    P3.5 ⏳ EmbodiedQAMFSDPPolicy worker（同 SAC target_model 模式）
P5  ⏳ YAML + smoke + LIBERO eval
```

> P3.1（冻结 f_β 快照）的初版尝试**已撤回**。理由：f_β 应由 worker 持有、独立 FSDP wrap，沿用现有 SAC `target_model` pattern，而不是塞进 model class 里。详见下方 P3.2 章节。

LWD 论文公式 (9) —— V1 实现按此对齐：

```text
L_QAM(θ) = E[ ∫₀¹ ‖(f_θ - f_β) · 2/σ_w + σ_w · g̃_w‖² dw ]
σ_w = √(2(1-w)/w),   g̃_1 = -∇_a [Q_φ(z_t, a¹) / λ]
```

> ⚠️ σ 必须与 adjoint 反向积分用的漂移 `b = 2f_β - a/t`（QAM 前向 SDE，见
> `qam_training_flow.md` Eq.19）取自**同一条 SDE**，所以是 √(2(1-w)/w)，不是
> √(2(1-w))。代码 (`compute_qam_actor_loss`) 用官方 QAM 的 h-shift 数值安全写法
> `σ_w = √(2(1-w+h)/(w+h))`，`h = 1/W`，在 w=0 与 w=1 两端都有限。

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

## P3.2：单模型 velocity 前向（`qam_velocity_forward`）

### P3.2 问题

QAM actor loss 需要 `f_δ = f_θ - f_β`：

- `f_θ` = **可训练的** action expert（live model 的 `paligemma_with_expert.gemma_expert` + 6 个 projection）
- `f_β` = **冻结的** reference action expert（SFT 加载后保持不变）

**初版尝试（P3.1）走错了层级**：在 `OpenPi0ForRLActionPrediction` 里加 `snapshot_f_beta()` 用 `add_module` 把 f_β 注册成 child module，再用 `_swap_to_fbeta` context manager 临时改 `self._modules`。Review 发现 3 个 FSDP 阻塞风险：

1. `add_module` 让 f_β 进入 `state_dict` / FSDP wrap / weight_syncer 作用域
2. `_swap_to_fbeta` 在 FSDP 下原地改 module tree 不稳定
3. live prefix 与 f_β 共享，依赖"VLM 永远冻结"这个隐式不变量

### P3.2 决策：照搬 SAC `target_model` pattern

RLinf 已有成熟的"frozen peer model"模式 —— [fsdp_sac_policy_worker.py:78-110](../rlinf/workers/actor/fsdp_sac_policy_worker.py#L78-L110)：

```python
module = self.model_provider_func()
target_module = self.model_provider_func()            # 第二个独立实例
self.model = self._strategy.wrap_model(module, ...)
self.target_model = self._strategy.wrap_model(         # 独立 FSDP wrap
    target_module, device_mesh=self._device_mesh
)
self.target_model.requires_grad_(False)
```

特点：

- `target_model` 由 **worker 持有**，不嵌入 model class
- **各自独立 FSDP wrap**，FSDP 不困惑
- 有独立 ckpt / soft-update 路径，已经验证多卡
- 多卡 device/dtype/sync 都走 FSDP 自己的管理

把这个模式套到 QAM `f_β` 上，3 个 review 风险**全部消失**。

### P3.2 文件改动

| 文件 | 改动 |
| --- | --- |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | **删除** P3.1 的 `snapshot_f_beta` / `has_f_beta_snapshot` / `_swap_to_fbeta` / `_f_beta_*` 占位（及 `contextlib` import） |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 新增 `qam_velocity_forward(obs, x_t, timestep)` — 单模型 velocity，返回 `[B, H, A]` |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | `forward()` dispatch 加 `ForwardType.QAM_VELOCITY` 分支（之前已加） |
| `tests/models/test_qam_critic_forward.py` | 删 4 个 snapshot/swap 单测；P2.2 critic 单测保留 |

### `qam_velocity_forward` 流水线（与 `qam_q_forward` 同样的 obs 处理）

```text
obs (replay dict) + x_t [B, H, A] + timestep [B]
    │
    ▼  ① _obs_processor_for_qam
    ▼  ② input_transform           (image resize/normalize)
    ▼  ③ precision_processor
    ▼  ④ _model.Observation.from_dict
    ▼  ⑤ _preprocess_observation → (images, masks, lang_tokens, lang_masks, state)
    ▼  ⑥ device/dtype 校正 x_t / timestep
    ▼  ⑦ with no_grad: _build_prefix_cache → prefix_pad_masks, past_key_values
    ▼  ⑧ get_velocity(state, x_t, t, prefix_pad_masks, past_key_values) → v_t

v_t [B, H, A]   ← 返回
```

⚠️ 入口 **assert PaliGemma frozen** —— 防止配置漂移让两个 model 实例的 VLM 不一致。

### P3.2 设计决策

- **不在 model class 持有 f_β**。worker 持有两个独立 model 实例（live + f_β），各自走完整 FSDP wrap。
- **f_β model 与 live model 共用 `model_provider_func` + 同一 SFT checkpoint**，加载完后整个 `requires_grad_(False)` + `eval()`。
- **VLM 各自跑 prefix**，不复用 live KV cache。两边 VLM 都已冻结、权重一致，结果数值上等价；省下的 "prefix 复用"优化放 V2。
- **device/dtype/shape 显式处理**：`qam_velocity_forward` 入口把 `x_t`、`timestep` 校正到 `state.device/dtype`，`timestep` 统一成 `[B]`。
- **VLM 冻结由两侧保证**：YAML `train_expert_only: true` + worker 显式 `freeze_vlm()`；`qam_velocity_forward` 入口再 assert 一次。

### P3.2 还没做的事

- ❌ Worker 侧建第二份 f_β 实例 —— P3.5。
- ❌ Adjoint state 反向积分 —— P3.3。
- ❌ Actor loss 组装 `(vf_fine - vf_base)·2/σ + σ·adj` —— P3.4。
- ❌ 端到端 GPU smoke —— P5。

### P3.2 验证

```bash
pytest tests/models/test_qam_critic_forward.py -v
```

P2.1 + P2.2 的 14 个 critic 单测继续 pass。`qam_velocity_forward` 本身的 GPU 端到端验证在 P5 smoke 里。

---

## 下一步：P3.3 adjoint state 反向积分

QAM 官方代码 [`adj_matching`](https://github.com/ColinQiyangLi/qam/blob/main/agents/qam.py#L50)：从 noise 出发跑 flow 前向积分得到 `xs`、再从终端 `∇_a Q` 出发**反向 vjp** 积分得到每步 adj，最终为 P3.4 的 loss 提供 `g̃_w` 序列。

```python
g̃_1 = -∇_a [Q_φ(z_t, a¹) / λ]
for w in reversed(grid):
    g̃_w = g̃_{w+h} + h · vjp(f_β at (s, x_w, w+h), g̃_{w+h})
```

PyTorch 用 `torch.autograd.functional.vjp` 实现。这是 V1 最难的一步，建议先在 toy `Q(a) = ‖a‖²` 上 sanity 后再上 OpenPI。

---

## 接力区（给下一个对话用）

### 当前快照（2026-05-29）

| 项 | 值 |
| --- | --- |
| 分支 | `lwd` |
| HEAD | `160ea0a fix(lwd): return tokenized prompt dict from _tokenize_language` |
| 上一步业务提交 | `fca840f refactor(lwd): move QAM f_β handling from model to worker (P3.1 redo)` |
| 路线进度 | P1 ✅ / P2.1 ✅ / P2.2 ✅ / P3.2 ✅ |
| 下一步 | **P3.3 adjoint state 反向积分**（最难） |

### 不变量（review 时必须保证不破坏）

1. **f_β 由 worker 持有**，不进 model class（参考 `rlinf/workers/actor/fsdp_sac_policy_worker.py` 的 `target_model` 模式）。
2. **plain QAM**（关闭 `edit_scale` / `fql_alpha`），不做 DIVL。
3. **PaliGemma 始终冻结**，YAML `train_expert_only: true` + worker `freeze_vlm()` + `qam_velocity_forward` 入口 assert 三处保险。
4. **replay obs 契约**：`tokenized_prompt + tokenized_prompt_mask` 走 `_obs_processor_for_qam` 喂给 OpenPI，**不要**再 carry `task_descriptions: list[str]`。
5. **新加东西配单测**：P2.1/P2.2 在 `tests/models/test_qam_critic_forward.py`（14 个 pass），P1 在 `tests/data/test_append_transitions_language.py`（5 个 pass）。

### P3.3 验收目标

- 在 `rlinf/algorithms/embodiment/` 下新加一个**独立函数** `compute_adjoint_states(f_beta_fn, obs, x1, Q_grad_at_1, num_steps, lambda)`，返回 `xs: [W+1, B, H, A]` 与 `adjs: [W+1, B, H, A]`，**不依赖** worker / FSDP / OpenPI 具体实现。
- 必须包含 toy sanity 单测：用 `Q(a) = ‖a‖²` + 解析常向量场 `f_β(x, w) = c`，对照解析 adj。
- 通过后再接 OpenPI 的 `qam_velocity_forward` 当 `f_beta_fn`。

### 协作约定（这位用户的偏好）

- 每个 phase 先**给 patch review**，确认后再 apply；得到 "直接帮我修改代码并提交推送" 才能 commit + push。
- commit 用 `git commit -s`（DCO 强制），格式 [Conventional Commits](https://www.conventionalcommits.org/)，scope 用 `lwd`。
- 一个 phase = 一次 commit；commit message 中文 OK，subject < 72 字符。
- 推完之后给一条"服务器侧验证命令"（通常是一行 `git pull && pytest ...`）。

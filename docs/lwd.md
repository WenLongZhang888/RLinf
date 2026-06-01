# QAM 实现

本仓库在 OpenPI π0.5 + LIBERO 上落地 [QAM (Q-learning with Adjoint Matching)](https://github.com/ColinQiyangLi/qam) 的 plain 变体（对应 LWD 论文 §III.C，关闭官方 QAM 仓库的 `edit_scale` / `fql_alpha`）。本文档按阶段记录已完成的实现与遗留事项。

## 总体路线

```text
P1  ✅ 把 task language 保留进 replay
P2  critic 前向 (Q_φ)
    P2.1 ✅ qam_q_forward + Q-head（不含 loss / target / optim）
    P2.2 ✅ 单元测试
    P2.4 ✅ TD loss + target critic + critic optimizer（worker 侧）
P3  actor 训练
    P3.2 ✅ qam_velocity_forward（单模型 velocity；worker 持有两份模型实例）
    P3.3 ✅ compute_adjoint_states 反向积分 + 单测（含 state-dependent toy）
    P3.4 ✅ QAM actor loss + 前向 SDE sampler + actor objective
    P3.5 ✅ EmbodiedQAMFSDPPolicy worker（同 SAC target_model 模式）
P5  🟡 YAML + smoke debug + LIBERO eval
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
- **device/dtype/shape 显式处理**：`qam_velocity_forward` 入口把 `state`、`x_t`、`timestep` 校正到 `action_in_proj.weight.dtype`，`timestep` 统一成 `[B]`。不能跟随 replay `states.dtype`，因为 replay 可能是 `float64`，而 OpenPI projection 权重通常是 `float32` / bf16。
- **VLM 冻结由两侧保证**：YAML `train_expert_only: true` + worker 显式 `freeze_vlm()`；`qam_velocity_forward` 入口再 assert 一次。
- **QAM ↔ OpenPI flow-time 镜像**：QAM (LWD §III.C) 约定 `w=0` 噪声、`w=1` 数据；OpenPI 内部 (`_get_timesteps` 单调递减) 约定 `t=0` 数据、`t=1` 噪声。两者镜像 `t = 1 - w`，且 velocity 方向相反（`v_QAM = dx/dw = -v_OpenPI`）。这个转换**仅**封装在 `qam_velocity_forward` 内（唯一了解 OpenPI 的入口）；上层 `rlinf.algorithms.embodiment.qam` 全部保持 QAM 坐标。`_qam_to_openpi_flow_time` 是 staticmethod，单测钉死。

### P3.2 后续完成项

- ✅ Worker 侧建第二份 f_β 实例 —— P3.5。
- ✅ Adjoint state 反向积分 —— P3.3。
- ✅ Actor loss 组装 `(vf_fine - vf_base)·2/σ + σ·adj` —— P3.4。
- 🟡 端到端 GPU smoke / LIBERO eval —— P5。

### P3.2 验证

```bash
pytest tests/models/test_qam_critic_forward.py -v
```

P2.1 + P2.2 的 14 个 critic 单测继续 pass。`qam_velocity_forward` 本身的 GPU 端到端验证在 P5 smoke 里。

---

## P2.4/P3.4/P3.5：QAM worker 数据流与 smoke 问题复盘

### 两个 action space

这次 smoke 暴露的核心问题是：LIBERO 的 env action 维度和 OpenPI action expert 的内部 flow 维度不是同一个东西。

| 名称 | shape | 用途 |
| --- | --- | --- |
| OpenPI flow / model action | `[B, H, 32]` | `action_in_proj`、`get_velocity`、`qam_velocity_forward`、QAM SDE trajectory、adjoint 都在这个空间里算 |
| LIBERO env / critic action | `[B, H, 7]` | 真正执行到环境、写入 replay、`QAM_Q` critic 输入、TD target critic 输入 |

因此 YAML 里需要同时表达这两层：

```yaml
actor:
  model:
    action_dim: 7          # env/critic action dim
    openpi:
      action_dim: 32       # internal OpenPI flow/action expert dim
      action_env_dim: 7
```

如果 QAM worker 错把 `[B, H, 7]` 当成 velocity trajectory 喂给 `qam_velocity_forward`，OpenPI 的 `action_in_proj` 会收到 7 维输入，但它的权重期望 32 维，于是报：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (...x7 and 32x1024)
```

当前修复把 worker 明确分成两套 shape：

- `qam_flow_action_shape(batch_size)`：返回 `[B, H, 32]`，只给 OpenPI velocity / SDE / adjoint 用。
- `qam_critic_action_shape(batch_size)`：返回 `[B, H, 7]`，只给 env action / replay action / critic 用。
- `_critic_actions_from_flow(actions)`：把 `[B, H, 32]` 切到 `[B, H, 7]`，再喂 critic 或 env-side contract。

### dtype 问题

另一个 smoke 报错是：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

根因是 replay 里的 `states` 可能是 `float64`。如果 `qam_velocity_forward` 把 `x_t` / `timestep` 跟随 `state.dtype` 转成 Double，再进入 OpenPI `action_in_proj`，就会和 projection 权重的 `float32` / bf16 dtype 冲突。

当前实现以 `self.action_in_proj.weight.dtype` 为准，把 `state`、`x_t`、`timestep` 都转到这个 dtype。这样 replay dtype 只影响输入来源，不再污染 action expert 的线性层输入。

### FSDP + 显式梯度问题

QAM 还有一个普通 SAC / PPO 很少碰到的工程问题：它不是只调用 `loss.backward()`，而是需要把两个中间梯度**显式取出来当张量用**。

| 位置 | 需要的数学量 | 当前触发点 |
| --- | --- | --- |
| `q_grad_fn(xs[-1])` | terminal seed `∇_a Q(s, a_1)` | 对 FSDP-wrapped `target_model(QAM_Q)` 的 action 输入调用 `torch.autograd.grad` |
| `compute_adjoint_states` | VJP `(∂b_β/∂x)^T g`，其中 `b_β=2f_β-x/t` | `torch.autograd.functional.vjp` 内部仍调用 `torch.autograd.grad`，forward 里走 FSDP-wrapped `f_beta_model(QAM_VELOCITY)` |

smoke 中对应报错是：

```text
RuntimeError: A leaf node was passed to _will_engine_execute_node
but we are currently running autograd.grad().
```

根因不是 QAM 公式错，而是 PyTorch FSDP 对“wrapped module forward 图里，对手工创建的 leaf input 调 `autograd.grad` / `functional.vjp`”支持不好。SAC/DSRL 通常不会撞到，因为它们虽然也让 Q 的梯度回到 actor，但路径是：

```text
actor_loss = alpha * log_pi - Q(s, pi(s))
actor_loss.backward()
```

梯度是隐式通过普通 backward 回传的；PPO/value critic 也是普通 `value_loss.backward()`。只有 QAM 需要把 `∇Q` 和 VJP 结果拿出来喂给 adjoint recursion。

建议修复策略是：在 FSDP worker 路径里避免 `autograd.grad`，改用普通 backward 并读取输入 `.grad`：

```python
# terminal critic gradient
flat_action = critic_action.reshape(B, -1).detach().requires_grad_(True)
q_mean = reduce(target_model(QAM_Q, actions=flat_action))
flat_action.grad = None
q_mean.sum().backward()
grad = flat_action.grad

# adjoint VJP
x_t = traj[step].detach().requires_grad_(True)
drift = beta_drift_at_x(x_t)
x_t.grad = None
torch.autograd.backward(drift, grad_tensors=adj_next)
vjp_x = x_t.grad
```

这和 `autograd.grad` / `functional.vjp` 在数学上等价，都是取 input-Jacobian 的 vector product；区别是它走 FSDP 更常规的 backward 路径。安全前提是 `target_model.requires_grad_(False)`、`f_beta_model.requires_grad_(False)`，当前 worker 已满足，所以 backward 只应给局部输入 `flat_action` / `x_t` 产生梯度，不训练 target critic 或 frozen `f_β`。

### rollout / eval 数据流

rollout 和 eval 执行的是 live policy `f_θ`，不使用 critic、不使用 f_β、不跑 adjoint。

```text
env obs
  └─ OpenPI obs_processor / input_transform
      └─ sample_actions()
          ├─ 初始 latent action: [B, H, 32]
          ├─ action expert 在 32-D 内部空间做 flow denoising
          └─ output_transform 裁剪/反归一化到 env action: [B, H, 7]
                └─ LIBERO env.step(action)
```

写入 replay 的 `forward_inputs["action"]` 是 env 执行过的 action，shape 展平成 `[B, H*7]`；`forward_inputs["model_action"]` 保留 OpenPI 原始输出，shape 展平成 `[B, H*32]`。QAM critic 训练使用前者，也就是 env/critic action。

eval 时同样测的是 `f_θ` 当前策略。它不应该加入 QAM forward SDE 中每一步的 `σ_t z_t` 扩散噪声；QAM 的 SDE 噪声只服务于 actor objective 的训练轨迹采样。OpenPI policy 本身仍然会从初始 latent noise 出发做 flow 采样/ODE denoising，这是底层 flow policy 的采样机制。

### critic TD 数据流（P2.4）

```text
batch from replay
  ├─ curr_obs / next_obs: main_images, states, tokenized_prompt, tokenized_prompt_mask, ...
  ├─ actions: replay env action [B, H, 7] 或 flat [B, H*7]
  ├─ rewards
  └─ terminations

current Q:
  curr_obs + replay actions
    └─ model(forward_type=QAM_Q)
        ├─ _obs_processor_for_qam
        ├─ PaliGemma prefix → pool → z_t [B, 2048]
        ├─ actions reshape → [B, H*7]
        └─ q_head_qam(z_t, action) → [B, num_q_heads]

target Q:
  next_obs
    └─ _sample_qam_ode_actions()
        ├─ sample x_0 ~ N(0, I) in OpenPI flow space [B, H, 32]
        ├─ deterministic ODE rollout with live f_θ
        ├─ slice first env dims → [B, H, 7]
        └─ clamp to [-1, 1]
    └─ target_model(forward_type=QAM_Q, obs=next_obs, actions=next_actions)
    └─ target = r + γ^H · mask · reduce(Q_target)
```

TD loss 是 `F.mse_loss(all_data_q_values, target_q_values.expand_as(all_data_q_values))`。target 侧默认用 `mean - ρ·std` 做 pessimistic ensemble reduction；actor gradient 侧用 ensemble mean。

### target_model 语义与更新范围

QAM worker 里的 `target_model` 在**工程对象**上是一整个 OpenPI 模型拷贝：

```text
target_model
  ├─ PaliGemma / VLM
  ├─ action expert + projections
  └─ q_head_qam
```

但在**算法语义**上它只作为 target critic 使用：

```text
Q_target(s, a)
  = q_head_qam_target(pool(PaliGemma_frozen(s)), a)
```

之所以 clone 整个 OpenPI 模型，是因为 `QAM_Q` forward 需要复用完整 obs pipeline：`_obs_processor_for_qam`、`input_transform`、PaliGemma prefix、prefix pooling，然后才进入 `q_head_qam`。但是 plain QAM 冻结 PaliGemma，真正通过 critic loss 学习的是 `q_head_qam`。

因此 target update 的正确范围是：

```text
target_model.q_head_qam ← EMA(self.model.q_head_qam)
```

不要 soft-update：

- PaliGemma / VLM：frozen feature extractor，同 SFT checkpoint 初始化即可。
- action expert / projections：属于 live actor `f_θ` 的 velocity field，不是 QAM target critic 的可训练 head。
- `f_beta_model`：plain QAM 的 frozen SFT reference，永远不 soft-update。

这也是为什么 SAC worker 里“全模型 `named_parameters()` zip 后 assert 名字一致”的 target update 不能直接照搬到 QAM。QAM + FSDP 下，live model 是“部分 frozen + 部分 trainable”，target_model 又整体 `requires_grad_(False)`；FSDP 对 frozen/trainable 参数的暴露方式可能不同，导致全模型参数集合或子模块 `named_parameters()` 不稳定。smoke 中已经出现过两类症状：

```text
missing=['...paligemma...']              # target 暴露 frozen VLM，live 没有同名项
missing=['qs.0.net.0.weight', ...]       # q_head 子模块级别也可能因 FSDP 参数托管而不直接暴露
```

稳妥做法是用 FSDP-safe 的方式只收集 `q_head_qam` 参数：必要时 `summon_full_params`，从 root wrapper 的 `named_parameters()` 中筛 `q_head_qam.`，用 `q_head_qam.` 后面的相对名匹配，然后只 EMA 写回 target q-head。

这点和当前 RLInf 的 OpenPI SAC/DSRL critic 不同：DSRL 的 `sac_q_forward` 不走 PaliGemma prefix pooling，也不使用这里的 `z_t`。它用轻量 image/state encoder：

```text
main_images / states
  → critic_image_encoder / critic_state_encoder
  → q_head(state_features, image_features, action_noise)
```

QAM critic 才是 LWD 风格的：

```text
obs
  → PaliGemma prefix_output
  → pool prefix 得到 z_t
  → q_head_qam(z_t, env_action)
```

所以 SAC/DSRL 的 target update pattern 只能借用“有 target_model”这个工程壳，不能照搬“全模型参数一一同步”的假设。

### actor objective 数据流（P3.4/P3.5）

QAM actor 更新全程在 OpenPI 32-D flow space 里做，只有 terminal critic gradient 需要跨到 7-D critic action 再补回 32-D。

```text
curr_obs
  └─ compute_qam_actor_objective(...)
      ├─ sample_forward_sde(...)
      │   ├─ x_0 ~ N(0, I), shape [B, H, 32]
      │   ├─ steps 0..W-2:
      │   │   x ← x + h · (2f_θ(obs,x,t) - x/(t+h)) + √h · σ_t · z
      │   │   σ_t = √(2(1-t+h)/(t+h))
      │   └─ last step:
      │       x ← x + h · f_β(obs,x,(W-1)/W)
      │
      ├─ q_grad_fn(xs[-1])
      │   ├─ slice terminal flow action [B,H,32] → critic action [B,H,7]
      │   ├─ target_model(QAM_Q) 取 ensemble mean
      │   ├─ ∇_a Q 得到 [B,H,7]
      │   └─ zero-pad 回 [B,H,32] 作为 terminal adjoint seed
      │
      ├─ compute_adjoint_states(f_β, ..., Q_grad_at_1)
      │   └─ 反向 VJP 积分，得到 adjs [W+1,B,H,32]
      │
      ├─ 重新计算 vf_fine = f_θ(xs[:-1], t_k)  # 有梯度
      ├─ 重新计算 vf_base = f_β(xs[:-1], t_k)   # no_grad / frozen
      └─ compute_qam_actor_loss(vf_fine, vf_base, adjs[:-1])
          └─ loss = Σ_time Σ_action_dims residual² 后按 batch mean
```

这里的角色对应关系：

- `self.model` = live `f_θ`：从 SFT checkpoint 初始化，训练 action expert / projections / Q head，PaliGemma 冻结。
- `self.f_beta_model` = frozen `f_β`：同一个 SFT checkpoint 初始化，`requires_grad_(False)`，不 soft-update。
- `self.target_model` = target critic/model：工程上是完整 OpenPI copy；算法上只用于 target Q 和 actor terminal gradient，soft-update 范围应限制在 `q_head_qam`。

---

## P3.3：adjoint state 反向积分

QAM 官方代码 [`adj_matching`](https://github.com/ColinQiyangLi/qam/blob/main/agents/qam.py#L50)：从 noise 出发跑 flow 前向积分得到 `xs`、再从终端 `∇_a Q` 出发**反向 vjp** 积分得到每步 adj，最终为 P3.4 的 loss 提供 `g̃_w` 序列。

```python
g̃_1 = -∇_a [Q_φ(z_t, a¹) / λ]
for w in reversed(grid):
    g̃_w = g̃_{w+h} + h · vjp(f_β at (s, x_w, w+h), g̃_{w+h})
```

纯 helper 单测里可以用 `torch.autograd.functional.vjp` 表达这一步；但在 FSDP worker 路径里，`functional.vjp` 内部调用 `autograd.grad`，会触发上面的 FSDP 显式梯度限制。因此上线实现需要在 worker/FSDP 路径中用普通 backward 取 `x_t.grad` 来等价计算 VJP。state-dependent toy 单测用于钉住 VJP 使用的 trajectory source point。

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

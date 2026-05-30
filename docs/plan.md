# QAM V1 — 接下来要做什么（P3.4 / P2.4 / P3.5 / P5）

本文件承接 `docs/lwd.md`（总体路线 + 接力区）与 `docs/qam_training_flow.md`（按 QAM
论文公式的训练流），给出**下一步的详细执行步骤**。算法层两个纯函数已就位且对齐论文：

- `compute_adjoint_states`（`rlinf/algorithms/embodiment/qam.py`）—— lean adjoint 反向积分，
  按 QAM 论文 Eq.(25)，VJP 取自完整漂移 `b = 2f_β − a/t`，在**源点** `(traj[step], step/W)`
  线性化，种子 `adjs[-1] = −∇Q/λ = −τ∇Q`。✅ 含 state-dependent toy 单测。
- `compute_qam_actor_loss`（同文件）—— 步进 adjoint-matching 损失。σ 已对齐官方
  h-shift 写法 `√(2(1−t+h)/(t+h))`，reduction 对齐官方（对 action 维与 flow-step 求和、
  对 batch 求均值）。✅

> ⚠️ 当前仓库 `rlinf.algorithms.__init__` 链路会 `import ray`，`qc` 环境无 ray 时
> `pytest tests/algorithms/` 收集即报错。算法纯逻辑已用「直接加载 qam.py 模块文件」的方式
> 离线验证全部通过；**请在装有 ray 的环境跑一次**
> `pytest tests/algorithms/test_qam_actor_loss.py tests/algorithms/test_qam_adjoint_states.py -v`
> 做回归确认。

---

## 跨阶段必须遵守的契约（最易踩坑处）

1. **两套时间网格。** 前向 SDE 与 loss 里的速度场 `f_θ`、`f_β` 都在**未平移**的
   `t_k = k/W` 上求值；而 σ（sampler 与 loss 都要用）用**h-shift** 的
   `√(2(1−t+h)/(t+h))`，`h = 1/W`。对应官方：`actor(xs, ts)` 用未平移 ts，`sigma` 用平移。
2. **轨迹形状。** `xs : [W+1, B, H, A]`。loss 只用 `0..W−1` 位（`skip_terminal=True`）；
   终端 `x_1 = xs[-1]` 仅用来取 critic 梯度。
3. **critic ↔ flow 形状桥。** critic 吃扁平 action `[B, H·A]`，flow 用 `[B, H, A]`。
   终端梯度 `∇_a Q` 出来是扁平的，喂 `compute_adjoint_states` 前要 reshape 成 `[B, H, A]`。
4. **角色映射（plain QAM）。** `f_θ` = live action expert（可训练），`f_β` = 冻结 SFT 快照。
   `f_β` 充当官方的 `(target_)actor_slow`；前向 SDE 的**最后一步是纯 ODE Euler、用 `f_β`**
   （无噪声、无 `−a/t` 项），与官方一致。
5. **终端种子。** `g_1 = −τ·∇_a Q_φ(z, clip(a_1))`，用 **target** critic、ensemble **均值**
   （不是 min）、action clip 到 `[−1,1]`。把原始 `∇Q` 传给 helper，`lambda_ = 1/inv_temp`
   （τ = `inv_temp`，官方默认 0.3）。

---

## P3.4 — 前向 SDE sampler + actor objective（纯函数 + 单测）

在 `rlinf/algorithms/embodiment/qam.py` 新增两个**框架无关**纯函数，延续「纯函数 +
closure 驱动 + 可单测」的既有风格。

### 1) `sample_forward_sde(f_theta_fn, f_beta_fn, obs, action_shape, num_steps, generator=None)`

返回 `xs : [W+1, B, H, A]`，实现官方 Eq.(24)（带 h-shift）：

- `x_0 ~ N(0, I)`；`h = 1/W`。
- `i = 0..W−2`（SDE，用 `f_θ`）：`t = i/W`，
  `x ← x + h·(2·f_θ(obs,x,t) − x/(t+h)) + √h·σ_t·z`，`σ_t = √(2(1−t+h)/(t+h))`，`z~N`。
- `i = W−1`（最后一步，纯 ODE，用 `f_β`）：`x ← x + h·f_β(obs, x, (W−1)/W)`。
- 整个采样在 `torch.no_grad()` 下做（轨迹是固定的采样目标；loss 会用 grad 重新过 `f_θ`）。
- `generator` 传入以保证可复现；toy 单测用常向量场（确定性 ODE 路径可解析对照）。

### 2) `compute_qam_actor_objective(f_theta_fn, f_beta_fn, q_grad_fn, obs, action_shape, num_steps, inv_temp, loss_mask=None)`

worker 调用的编排（也保持纯函数、可用 toy closure 单测）：

1. `xs = sample_forward_sde(f_theta_fn, f_beta_fn, obs, action_shape, num_steps)`。
2. `g_at_1 = q_grad_fn(xs[-1])`（worker closure：target critic、ensemble 均值、clip、
   reshape 到 `[B,H,A]`）。
3. `_, adjs = compute_adjoint_states(f_beta_fn, obs, xs, Q_grad_at_1=g_at_1, lambda_=1/inv_temp)`。
4. 在未平移 `t_k = k/W`（`k=0..W−1`）、状态 `xs[:-1]` 上求速度：
   `vf_fine = f_theta_fn(...)`（**带 grad**），`vf_base = f_beta_fn(...)`（**no_grad**），
   stack 成 `[W, B, H, A]`。
5. `loss, metrics = compute_qam_actor_loss(vf_fine, vf_base, adjs[:-1], loss_mask)`
   （喂传播后的 adjoint `0..W−1`）。
6. 返回 `(loss, metrics)`。worker 只需提供三个真实 closure。

**单测：** toy `f_θ=f_β=const` → `loss≈0`（速度差为 0、adj 由常漂移决定）；
检查梯度只流经 `f_θ`（`f_β`、`adjs`、`xs` 无梯度）。

---

## P2.4 — critic TD loss + target critic + critic optimizer（worker 侧）

标准 QAM TD（`qam_training_flow.md` Eq.23）：

- `Q_φ(s,a)`：对 `curr_obs` + replay action 走 `forward_type=QAM_Q`。
- target：`r + γ^H · mask · Q_φ̄(s', a')`，其中 `a' ~ f_θ` 用 ODE roller 采（复用
  `compute_flow_actions` 式的整段积分），`Q_φ̄` 为 EMA target critic。
- ensemble 规约：**target backup** 用 `mean − ρ·std`（悲观），但 **actor 梯度**用 `mean`。
- `F.mse_loss` 对 bootstrapped target；独立 critic optimizer（镜像 SAC 的 `qf_optimizer`）。

参考 `rlinf/workers/actor/fsdp_sac_policy_worker.py` 的 `forward_critic` /
`qf_optimizer` / `soft_update_target_model`。

---

## P3.5 — `EmbodiedQAMFSDPPolicy` worker（照搬 SAC target_model 模式）

继承 `EmbodiedFSDPActor`（参照 `rlinf/workers/actor/fsdp_sac_policy_worker.py` 的
`EmbodiedSACFSDPPolicy`）。**复用，不重造**：

- `setup_model_and_optimizer(initialize_target=True)`：两个独立 `model_provider_func()`
  实例 → live `self.model`（f_θ）+ `self.f_beta_model`，各自独立 FSDP wrap；
  `f_beta_model.requires_grad_(False).eval()`，且**永不软更新**（冻结 SFT，区别于 SAC 的 EMA）。
  两者都加载同一 SFT checkpoint。
- 两侧都 `freeze_vlm()`；依赖 `qam_velocity_forward` 入口对 PaliGemma 冻结的 assert。
- 用 `forward_type=QAM_VELOCITY`（f_θ live / f_β frozen）与 `QAM_Q`（target critic 梯度）
  构造 `compute_qam_actor_objective` 的三个 closure。
- 训练步顺序（对齐官方 `total_loss`）：critic TD 步（P2.4）→ actor adjoint 步（P3.4）
  → **仅** soft-update critic target。
- replay 采样：复用 SAC 的 `buffer_dataloader_iter`；batch dict 键
  `curr_obs/next_obs`（含 P1 写入的 `tokenized_prompt`/`mask`）、`actions`、`rewards`、
  `terminations`。
- checkpoint：单独 save/load `f_beta_model`（镜像 SAC 的 `sac_components/target_model` 块），
  resume 时重建冻结参考。
- 这是**直接算 loss 的专用 worker**（像 SAC），**不要**把 QAM 走 `policy_loss` registry——
  actor loss 需要轨迹生成 + adjoint + critic 梯度，不符合「kwargs 进、loss 出」的契约。

---

## P5 — YAML + smoke + LIBERO eval

- 新增 `examples/embodiment/config/` 下的 QAM config：`actor.model.use_qam: true`、
  `train_expert_only: true`、`inv_temp`、`flow_steps`、`qam_*` head 维度、critic/actor
  optim、replay 容量。
- GPU smoke：端到端跑一个 critic+actor step（用来抓 `[B,H·A]↔[B,H,A]` 形状桥、σ 网格、
  VLM 冻结 assert）。
- LIBERO eval：对照 `docs/lwd_exp.md` 已记录的 SFT 基线。

---

## 验证

- 算法层（装 ray 的环境）：
  `pytest tests/algorithms/test_qam_actor_loss.py tests/algorithms/test_qam_adjoint_states.py -v`。
- P3.4 后：`sample_forward_sde`（常向量场→确定性 ODE 路径）与
  `compute_qam_actor_objective`（梯度只流经 f_θ）的 toy 单测。
- P3.5 后：`bash examples/embodiment/run_embodiment.sh <qam_smoke_cfg>` 单步 GPU smoke，
  再跑独立 LIBERO eval 脚本。
- 每次提交后服务器侧一行：`git pull && pytest tests/algorithms/ -q`。

## 协作约定

每个 phase 先给 patch review，确认 “直接帮我修改代码并提交推送” 后再 `git commit -s`
（Conventional Commits，scope `lwd`，subject < 72 字符）；一个 phase = 一次 commit。

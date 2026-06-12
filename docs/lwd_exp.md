# LWD 实验计划

本文档记录 RLinf 上 LWD 的计划实验流程。它是当前分支内的实验规划文件，后续应在实现和配置落地后继续修订。

## Baseline

- sft 训练实验：chunk10 seed0 libero40 20260527 在OpenPI官方仓库进行训练
- 单卡H200 batch_size为32
- 抽取的40条数据序号为：36, 37, 54, 87, 128, 133, 139, 331, 340, 362,
                    407, 432, 445, 481, 489, 505, 516, 553, 644, 746,
                    845，825, 882, 886, 904, 907, 946, 1082, 1093, 1199，
                    1284,1325, 1329, 1372, 1408, 1426, 1449, 1555, 1626，1659，

```bash
export HF_LEROBOT_HOME=/mnt/kpfs/zhangwenlong/openpi/.cache/openpi/datasets
export OPENPI_DATA_HOME=/mnt/kpfs/zhangwenlong/openpi/.cache/openpi
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero_40_sft \
    --exp-name=libero_40_sft \
    --overwrite \
  > logs/pi05_libero_40_sft.log 2>&1 &
```

```bash
# 测评实验 20250527
# spatial goal object 10 num_steps均为10,seed为0
# spatial goal object chunk为5, 10的chunk为10
nohup bash -c '
set -e

export EMBODIED_PATH=$PWD/examples/embodiment
export REPO_PATH=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export HYDRA_FULL_ERROR=1

for CFG in \
  libero40_spatial_sft_eval_pi05 \
  libero40_goal_sft_eval_pi05 \
  libero40_object_sft_eval_pi05 \
  libero40_10_sft_eval_pi05
do
  echo "========== $(date) START $CFG =========="
  bash examples/embodiment/eval_embodiment.sh "$CFG"
  echo "========== $(date) DONE  $CFG =========="
done
' > libero40_eval_all.log 2>&1 &
```



## 结果记录

在本节记录简要实验结果。

公共设置：`pi05_libero_40_sft_pytorch` checkpoint；eval `seed=0`；`num_steps=10`；`temperature_eval=0.6`。

| 任务环境 | seed | chunk | num_steps | 并行环境 | eval epoch | 配置名 | Checkpoint | 5.28 成功率 | 5.29 成功率（seed0 第二次实验） | 5.29 成功率（seed0 第三次实验） | 6.1 成功率（seed0 第四次实验） | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_spatial | 0 | 5 | 10 | 25 | 20 | libero40_spatial_sft_eval_pi05 | pi05_libero_40_sft_pytorch | success_once **84.8%** / success_at_end **76.6%** | success_once **84.4%** / success_at_end **74.6%** | success_once **83.8%** / success_at_end **76.2%** | success_once **84.8%** / success_at_end **75.4%** | 500 traj；episode_len=240 |
| libero_goal | 0 | 5 | 10 | 25 | 20 | libero40_goal_sft_eval_pi05 | pi05_libero_40_sft_pytorch | success_once **75.8%** / success_at_end **64.8%** | success_once **75.0%** / success_at_end **63.4%** | success_once **75.6%** / success_at_end **64.4%** | success_once **73.0%** / success_at_end **63.8%** | 500 traj；episode_len=320 |
| libero_object | 0 | 5 | 10 | 25 | 20 | libero40_object_sft_eval_pi05 | pi05_libero_40_sft_pytorch | success_once **76.6%** / success_at_end **76.4%** | success_once **72.0%** / success_at_end **71.2%** | success_once **74.2%** / success_at_end **73.4%** | success_once **73.4%** / success_at_end **72.6%** | 500 traj；episode_len=240 |
| libero_10 | 0 | 10 | 10 | 25 | 20 | libero40_10_sft_eval_pi05 | pi05_libero_40_sft_pytorch | success_once **46.2%** / success_at_end **31.0%** | success_once **44.8%** / success_at_end **32.6%** | success_once **46.6%** / success_at_end **32.2%** | success_once **45.6%** / success_at_end **31.6%** | 500 traj；episode_len=480 |

## 诊断实验 `libero40_spatial_qam_diag_terms_run0609`（6.9）

**背景：** 上一轮 QAM 训练 `libero40_spatial_qam_openpi_pi05_run0603_run2`（wandb `i9s79k4h`，
H200，step 1199）eval `success_once 85.6% / success_at_end 77.8%`，与同一 checkpoint 的
spatial SFT 基线 `success_once 84.8% / success_at_end 76.6%`（见上表）几乎完全相等 ——
说明 QAM 训练对策略**没有产生任何净改变**，actor 在 SFT 上原地踏步，没有学习信号。

**怀疑根因：** actor loss 形如 `‖ 2(f_θ-f_β)/σ + σ·adj ‖²`，是「行为正则项 `2·Δv/σ`」与
「Q 引导项 `σ·adj`」之和。从日志反推，引导项（`qam_adj_abs≈6e-5`）比正则项小约 30–200 倍，
即 reward 信号被行为正则淹没；策略只被拉回 f_β（SFT），Q 几乎不起作用。两个叠加原因：
①`inv_temp=0.1` 使 `adj=-dQ/da·inv_temp` 先天被砍 10 倍；②critic 的 Q 平（`q_data/q_target`
都在 0 附近 ±0.1），`dQ/da≈1e-4`。

**本次只加诊断指标、不改任何超参/行为**，沿用上一轮配置重跑一段（保证 ≥20 次 actor 更新）。
新增指标：

- `actor/qam_reg_term_abs`、`actor/qam_guidance_term_abs`、`actor/qam_term_ratio`
- `actor/qam_q_grad_abs`（未经 `1/inv_temp` 缩放的原始 `|dQ/da|`）
- `critic/q_data_sep`（成功 transition 与零奖励 transition 的 Q 均值之差）

**这次检查的意义：** 把「引导项被淹没」从推断变成可量化的读数，并定位是哪一腿出问题，
避免盲调超参——

- `qam_term_ratio` 直接量化两项比例：预期 ≪0.1（证实淹没），健康应 ~0.3–1.0。
- `qam_q_grad_abs` 区分「引导项小」的来源：若它本身就 ~1e-4 → **critic 平**；
  若它健康但 `qam_adj_abs` 小 ~10 倍 → **`inv_temp=0.1` 是元凶**。
- `q_data_sep` 判断 critic 是否真学到价值：长期 ≈0 → critic 没学会区分好坏动作，
  下一轮修复应从 critic 端（reward 尺度、悲观项/head 数、表征、critic warmup）入手，
  而不是先动 `inv_temp`。

读数与结论对应：`term_ratio ≪0.1` 证实淹没；`q_grad_abs~1e-4 且 q_data_sep 平`→治 critic；
`q_grad_abs 健康但 adj 小 10 倍`→调 inv_temp / 重新平衡两项。


### 诊断结果 `libero40_spatial_qam_diag_terms_run0609`（wandb 2nq3xbc5，commit 493ebf0）

跑诊断配置后，三个新指标读数（≥33 次 actor 更新）：

| 指标 | 读数 | 结论 |
| --- | --- | --- |
| `actor/qam_term_ratio`（正则项和引导项的比例） | ≈ **0.005**（稳定） | Q 引导项比行为正则项小 ~200 倍 → reward 在 actor loss 里几乎不起作用 ✅ 证实「淹没」 |
| `actor/qam_q_grad_abs`（原始 \|dQ/da\|，Q对最终动作a导数的平均值） | **0.0024 → 0.0039**（增长） | critic 梯度健康，约 2.4e-3，**不是** 1e-4 ❌ 推翻「critic 平」假设 |
| `critic/q_data_sep` | 从 ~0 涨到 **+0.1 ~ +0.4** | critic 确实在学，越来越会给成功 transition 更高的 Q |

（`qam_term_ratio` 首值 1.91e7 为 t=0 启动伪影：该步 f_θ=f_β、正则项=0 被 eps 除，忽略。）

**根因定性：** 不是代码 bug，也不是 critic 学不好。唯一瓶颈是**缩放**——`inv_temp=0.1`
把 adjoint 直接 ×0.1，叠加 σ 加权与「32 维中仅 7 维有梯度」的稀释，引导项 `σ·adj≈2e-5`
而正则项 `2Δv/σ≈4e-3`。于是 QAM 的 loss 退化成「把 f_θ 拉回 f_β」，reward 想要的策略修正
幅度 `σ²·adj/2≈1e-4`（动作量纲 [-1,1]）小到不可能改变行为 → eval 卡在 SFT 基线 84.8%。

**修复方向：** 不动 critic 侧（reward 缩放 / 加 Q head / 换表征都不是瓶颈），直接调大 `inv_temp`。
adjoint 对 inv_temp 严格线性，比值 `qam_term_ratio ≈ 0.05 × inv_temp`：

| inv_temp | 预期 term_ratio | 备注 |
| --- | --- | --- |
| 0.1（现状） | 0.005 | reward 被淹没 |
| 1 | 0.05 | 仍偏弱，大概率不动 |
| 5 | 0.25 | 健康下沿，保守 |
| 10 | 0.5 | 首选 |
| 20 | 1.0 | 引导≈正则（理论最优点），偏激进 |

从动作修正幅度独立验算（要 Δv 达 O(0.01~0.05) 才能改变行为）同样指向 inv_temp ≈ 10~20。
sweep 时盯：①`qam_term_ratio` 是否爬向 0.1~1.0；②`eval/success_once` 是否突破 84.8%；
③`actor/grad_norm` 与 eval 稳定性（inv_temp 过大修正过猛可能发散）。

**配套备注：** ① actor lr=1e-6 偏小且每 12 步才更一次，引导变强后若 eval 起色慢，可把 actor
lr 提到 3e-6~1e-5（先单独验 inv_temp，别一次改两个变量）。② 调大 inv_temp 属「治标」，依赖
当前 Q 尺度；治本是在终端对 `dQ/da` 归一化，使引导强度与 reward 尺度解耦、inv_temp 回到 O(1)。

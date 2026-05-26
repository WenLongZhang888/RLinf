# LWD 实验计划

本文档记录 RLinf 上 LWD 的计划实验流程。它是当前分支内的实验规划文件，后续应在实现和配置落地后继续修订。

## 主要实验

- 任务族：LIBERO spatial。
- Policy backbone：OpenPI PI0.5。
- 训练配置：`examples/embodiment/config/libero_spatial_lwd_openpi_pi05.yaml`。
- SFT warm-start 配置：`examples/sft/config/libero_sft_openpi_pi05.yaml`。
- 运行模式：先做单机实验；如果 placement 或吞吐需要，再扩展到多机。

TODO(agent): 在明确目标机器目录结构后，补充精确的数据集路径、checkpoint 路径和环境变量。

## Baseline

- 如果仓库中已有 OpenPI PI0.5 LIBERO spatial RL 配置，将其作为 baseline。
- 只经过 SFT 的 OpenPI PI0.5 checkpoint，不进行 LWD 更新，直接评估。
- 如果实现允许 ablation，运行去掉 QAM actor 路径的 LWD。
- 如果实现允许 ablation，运行去掉 distributional projection 改动的 LWD。

TODO(agent): 正式对比前，将每个 baseline 映射到具体已入库的配置文件。

## 建议运行顺序

1. 运行 `rlinf/algorithms/lwd/` 的算法单元测试。
2. 使用 mock 依赖运行 OpenPI LWD model 单元测试。
3. 用极小 step 数启动一次 SFT smoke run。
4. 用极小环境数量和 step 数启动一次 embodied RL smoke run。
5. 运行完整 SFT warm-start。
6. 运行完整 LWD RL 实验。
7. 按训练间隔评估多个 checkpoint。

## 命令模板

单机 Ray 启动：

```bash
ray start --head
```

SFT smoke run：

```bash
python examples/sft/train_sft.py --config-name libero_sft_openpi_pi05
```

Embodied LWD smoke run：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name libero_spatial_lwd_openpi_pi05
```

TODO(agent): 在自动化使用该命令前，确认精确的 SFT entry script 名称。

## 需要跟踪的指标

- runner 输出的 `train/` success 或 reward 指标。
- LIBERO spatial 任务上的 `eval/` success rate。
- Actor loss 和 LWD 专用 loss 项。
- Critic loss、target 统计量，以及 projected distribution 诊断信息。
- 如果 actor module 暴露 QAM action 统计量，也需要记录。
- Rollout 吞吐、环境 step 时间和 actor update 时间。
- GPU 显存占用和 rollout engine memory utilization。

## Checkpoint 与评估

- 使用 `runner.save_interval` 保存 checkpoint。
- 生成 smoke checkpoint 后，用 `runner.resume_dir` 验证恢复训练。
- 优先评估多个 checkpoint，而不是只评估最终 checkpoint。
- 保持评估配置静态，并记录所有运行时环境变量。

## 可复现性检查清单

- Commit SHA 和分支名。
- Hydra composition 后的完整配置文件。
- 数据集和资产路径。
- Base OpenPI checkpoint 标识。
- Python 环境或 Docker image tag。
- Ray 版本和启动命令。
- GPU 类型、数量和 placement 配置。
- 随机种子和 evaluation episode 数量。

## 风险

- 通用 RLinf 环境中可能没有安装 OpenPI 依赖。
- 完整 LIBERO/OpenPI 运行可能需要 CI 中不可用的 GPU、资产或 checkpoint。
- Distributional projection 的 bug 可能表现为 RL 不稳定，而不是明显的测试失败，因此边界条件测试很重要。
- Worker 级测试可能需要 mock，以避免启动 Ray 或加载大模型。

## 结果记录

在本节记录简要实验结果。

| 日期 | 配置 | Checkpoint | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| TODO(agent) | TODO(agent) | TODO(agent) | TODO(agent) | TODO(agent) |

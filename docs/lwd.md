# LWD 实现计划

本文档用于记录 RLinf 中 LWD 相关功能的计划实现范围。它是 `lwd` 分支上的工作设计笔记，后续应随着算法细节和代码实现逐步更新。

## 目标

- 在 `rlinf/algorithms/lwd/` 下新增 LWD 算法工具。
- 为 embodied policy 新增分布式 critic、QAM actor 和 action adapter 模块。
- 通过专用 action model 和 factory 将 LWD 接入 OpenPI。
- 新增可训练 LWD policy 路径的 FSDP actor worker。
- 提供 LIBERO spatial OpenPI PI0.5 训练配置和 SFT 示例配置。
- 通过聚焦的单元测试覆盖算法逻辑和模型行为。

TODO(agent): 在锁定公开命名和文档表述前，确认 LWD 的完整含义以及论文级定义。

## 计划文件

### 算法

- `rlinf/algorithms/lwd/__init__.py`
- `rlinf/algorithms/lwd/divl.py`
- `rlinf/algorithms/lwd/qam.py`
- `rlinf/algorithms/lwd/targets.py`
- `rlinf/algorithms/lwd/projection.py`

### 模型模块

- `rlinf/models/embodiment/modules/lwd_distributional_critic.py`
- `rlinf/models/embodiment/modules/lwd_qam_actor.py`
- `rlinf/models/embodiment/modules/lwd_action_adapter.py`

### OpenPI 集成

- `rlinf/models/embodiment/openpi/lwd_openpi_action_model.py`
- `rlinf/models/embodiment/openpi/lwd_factory.py`

### Worker

- `rlinf/workers/actor/fsdp_lwd_policy_worker.py`

### 配置

- `examples/embodiment/config/libero_spatial_lwd_openpi_pi05.yaml`
- `examples/sft/config/libero_sft_openpi_pi05.yaml`

### 测试

- `tests/algorithms/lwd/test_divl.py`
- `tests/algorithms/lwd/test_qam.py`
- `tests/algorithms/lwd/test_targets.py`
- `tests/algorithms/lwd/test_projection.py`
- `tests/models/embodiment/openpi/test_lwd_openpi_action_model.py`

## 组件职责

### `divl.py`

放置 actor/critic 训练中使用的主要 LWD loss 或 divergence 工具函数。输入应当是 worker 已经整理好的 tensor，并且需要显式处理 shape 检查和 mask 逻辑。

TODO(agent): 检查 actor worker 调用点和 LWD 原始公式后，再定义精确的函数签名。

### `qam.py`

包含 QAM 相关的 actor objective 或 scoring helper。这里应尽量只保留纯 tensor 逻辑，使其可以脱离 OpenPI 和 FSDP 单独测试。

TODO(agent): 确认当前实现中的 QAM 是 policy head、action mixture transform，还是训练目标。

### `targets.py`

为 distributional critic 构造 target distribution 或 bootstrapped target。该模块应尽量避免模型特定假设。

### `projection.py`

将 target distribution 投影到 critic support 上。投影实现需要数值稳定、向量化，并通过边界条件测试覆盖。

### `lwd_distributional_critic.py`

定义 critic head/module，包括 support 参数、logits/value 转换，以及 worker 需要调用的辅助方法。

### `lwd_qam_actor.py`

定义 actor 侧的 QAM 模块。该模块应暴露尽量窄的接口，供 OpenPI 集成层调用，避免重复实现 tensor 转换逻辑。

### `lwd_action_adapter.py`

负责 OpenPI action 表示、环境 action，以及 LWD distributional/QAM 专用 action 格式之间的转换。

### `lwd_openpi_action_model.py`

在 OpenPI action generation 外层封装 LWD 专用的 actor/critic 输出。该文件应把 OpenPI 相关 wiring 留在集成层，避免污染通用算法模块。

### `lwd_factory.py`

提供 OpenPI LWD model、processor、adapter 和 critic module 的构造辅助函数。优先沿用仓库中已有的 OpenPI factory 风格。

### `fsdp_lwd_policy_worker.py`

负责 LWD policy 的 FSDP 训练路径。worker 应调用共享的 LWD 算法模块，而不是把 loss 数学逻辑直接写在 worker 代码中。

## 集成注意事项

- 只有当新的算法入口需要通过配置选择时，才将其注册到 RLinf 现有 registry。
- 对 OpenPI 或环境特定依赖保持 lazy import，避免用户安装其他 RLinf target 时在 import 阶段失败。
- 只有当新配置引入无法在局部捕获的用户可见约束时，才在 `rlinf/config.py` 中增加校验。
- 面向用户的配置应使用静态 YAML 值，不要依赖代码中的计算型默认值。
- 公开 API 应包含类型注解和 Google-style docstring。

## Fork 维护流程

本仓库预期基于 fork 进行开发。保持 `origin` 指向自己的 fork，并将官方 RLinf 仓库添加为 `upstream`。

同步上游代码前，建议先提交或 stash 本地修改，保持工作区干净：

```bash
git status
git add docs/lwd.md docs/lwd_exp.md
git commit -s -m "docs: add lwd planning notes"
```

检查 remote：

```bash
git remote -v
```

首次添加 upstream：

```bash
git remote add upstream git@github.com:RLinf/RLinf.git
```

如果没有配置 SSH，也可以使用 HTTPS：

```bash
git remote add upstream https://github.com/RLinf/RLinf.git
```

拉取 fork 和 upstream 的最新引用：

```bash
git fetch origin
git fetch upstream
```

用 upstream 更新本地 `main` 分支，并推回自己的 fork：

```bash
git checkout main
git pull --ff-only upstream main
git push origin main
```

把 upstream `main` 的最新变化合入 LWD 分支：

```bash
git checkout lwd
git merge upstream/main
```

如果出现冲突，解决后完成 merge：

```bash
git status
git add <resolved_files>
git commit
```

将更新后的 LWD 分支推送到自己的 fork：

```bash
git push origin lwd
```

如果希望分支历史更线性，也可以使用 rebase 流程：

```bash
git checkout lwd
git rebase upstream/main
git push --force-with-lease origin lwd
```

如果更重视保留分支历史，优先使用 merge。只有当该分支是个人分支，或协作者都同意重写历史时，才使用 rebase。

## 测试策略

- 使用小规模、确定性的 tensor 测试 LWD 算法函数。
- 覆盖 shape、dtype、device、mask 和边界值场景。
- 测试 support 边界和精确 atom 边界处的 projection 行为。
- OpenPI action model 测试应使用轻量 mock，避免加载完整 checkpoint。
- GPU 或重量级集成测试应添加合适的 skip 标记。

## 待确认问题

- 文档中应使用哪一个精确的 LWD objective 和符号体系？
- 应使用哪个 config key 选择 LWD worker 路径？
- Distributional critic 是否与 OpenPI trunk 共享参数，还是使用独立模块？
- 哪些指标应记录到 `train/`、`eval/` 和 `loss/` namespace？
- 初始目标是否只需要覆盖 LIBERO spatial 和 OpenPI PI0.5？

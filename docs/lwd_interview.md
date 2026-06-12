# QAM / LWD 项目面试手卡

这份文档不是论文式总结，而是面试时可以直接照着思路讲的版本。目标是讲清楚三件事：

- 这个项目为什么有意义。
- 真正难点是什么。
- 我不是只会跑实验，而是真的分析了结果和问题。

## 先用一句话讲项目

我做的是把 QAM 这个适合 flow / diffusion policy 的 RL 微调方法，接到 RLinf 的 OpenPI π0.5 VLA 训练流程里，让一个 SFT 初始化的机器人 VLA 能继续用在线 rollout 和任务 reward 做微调。

更口语一点：

```text
OpenPI 这种 VLA 不是直接输出一个简单动作分布，而是通过 flow/velocity 生成动作。
所以我没有硬套 PPO，而是选了 QAM：用 critic 学任务奖励，再用 adjoint matching 去更新
OpenPI 的 action velocity field。最后把这个算法真的接进了 RLinf 的 Ray、FSDP、rollout、
replay 和 checkpoint 流程里。
```

## 面试开场 30 秒版

如果面试官让你“简单介绍一下这个项目”，可以这样说：

```text
这个项目主要是解决 VLA 做在线 RL 时，OpenPI 这类 flow policy 不太适合直接套 PPO 的问题。
OpenPI 的动作不是一步采样出来的，而是 action expert 通过多步 velocity field 生成的。
所以我用 QAM，把 replay 里学到的 Q critic 的梯度，通过 frozen SFT policy 的 adjoint 反传到
整条 flow trajectory 上，再训练当前 policy 的 velocity field。

我做的工作不是只写一个 loss，而是把它完整接到了 RLinf 的 OpenPI π0.5 + LIBERO 训练栈里。
包括 replay 里保留语言，OpenPI 增加 QAM_ENCODE/QAM_VELOCITY，worker 里维护 live actor、
frozen behavior policy 和 FP32 critic head，还处理了 OpenPI 32 维 flow action 到 LIBERO
7 维控制 action 的映射。最后在 LIBERO Spatial 上，SFT base 大概 84.6%-84.8% success_once，
QAM checkpoint 到了 86%-87%，有小幅提升，而且没有把 policy 推崩。
```

## 面试主回答 2-3 分钟版

我做这个项目的出发点是：VLA 通常会先做 SFT 或 behavior cloning，这一步能让模型学到一个不错的初始策略，但它优化的是“像不像示范动作”，不是“任务最后成不成功”。在 LIBERO 这种长程 manipulation 任务里，经常会看到动作大体合理，但最后没完成，或者成功一下又没保持住。所以我想用在线 RL 的 reward 信号继续微调 VLA。

但 OpenPI / π0.5 这类模型有一个麻烦点：它不是普通 Gaussian policy，也不是 LLM token policy。它更像 flow / diffusion policy，动作是通过 action expert 的 velocity field 一步步生成的。如果硬套 PPO，就要处理 logprob、ratio、action chunk credit assignment，这些都不太自然。所以我关注到了 QAM。QAM 的好处是，它不是绕开 flow 结构，而是直接在 velocity field 上做 RL fine-tuning。

QAM 的核心可以这样理解：我有一个当前正在训练的 velocity `f_theta`，一个冻结的 SFT 参考 velocity `f_beta`，还有一个从 replay 里学出来的 critic `Q(s,a)`。critic 负责判断某个 action chunk 对任务成功有没有帮助。actor 更新时，不是简单对最终 action 做梯度上升，而是先沿着当前 flow 采一条 action 轨迹，在终点拿到 `grad_a Q`，然后通过 frozen `f_beta` 做 adjoint 反传，把这个终点的改进方向传回整条 flow trajectory。最后训练 `f_theta`，让它相对 `f_beta` 朝这个方向调整。

直觉上就是：critic 告诉我“最终动作应该往哪里改”，QAM 把这个信号翻译成“每个 flow step 的 velocity 应该怎么改”。同时因为有 frozen SFT policy `f_beta`，所以更新不会一下子离开原来的动作流形。

我把它用到 VLA 里时，主要做了几件事。第一，critic 要重算 observation feature，所以 replay 里不能只存图像和状态，还要保留语言指令，我把 task description tokenized 后存进 transition。第二，OpenPI 增加了两个 forward type：`QAM_ENCODE` 用 frozen PaliGemma 编码 observation，`QAM_VELOCITY` 计算任意 flow time 上的 action velocity。第三，worker 里有两个 policy：live `f_theta` 用来训练，frozen `f_beta` 作为 SFT reference。第四，我把 critic head 从 FSDP-wrapped OpenPI 里拆出来，做成 worker 自己持有的 FP32 online/target q-head，这样更稳定，也更符合 QAM 里 critic 的语义。

这个项目最大的坑是 action space 和系统边界。OpenPI 内部 flow action 是 32 维，但 LIBERO 环境 action 是 7 维。QAM 的 SDE 和 adjoint 必须在 32 维空间里跑，但 critic 和 env 必须用 7 维。所以我显式做了 slice 和 zero-pad：terminal flow action 先切到 7 维给 critic，critic 的梯度再 pad 回 32 维继续做 adjoint。

结果上，方法是能跑通的，也没有 collapse。SFT base 在 LIBERO Spatial 500 条 eval 上 `success_once` 大概 84.6%-84.8%，`success_at_end` 大概 75%-76%。QAM 的 step 600 是 `success_once=0.860 / success_at_end=0.772`，step 900 是 `0.870 / 0.766`。所以我会比较谨慎地说：它在强 baseline 上有小幅提升，但不是大幅提升。

我也分析了原因。第一，SFT baseline 已经很强，提升空间本来就不大。第二，LIBERO reward 偏稀疏，critic 很难学到特别细的动作级修正。第三，成功样本比较多，失败边界样本相对少，critic 对“哪些动作会失败”的排序可能还不够好。第四，我是冻结 VLM、只训 action expert，这样很稳，但表达能力也受限制。日志里也能看到后半程 actor grad 和 velocity delta 在变大，但 eval 提升进入平台，所以我后续把 run2 改得更保守：更长 critic warm-up、更低 actor lr、更大的 `critic_actor_ratio`，并用更大规模 eval 选 checkpoint。

## 这个项目最应该突出的难点

面试里不要只说“我实现了 QAM”。更有含金量的说法是：

```text
难点不是公式本身，而是 QAM 这个算法刚好踩中了 VLA + 分布式 RL 的几个边界：
它需要显式 grad_a Q 和 adjoint VJP；
OpenPI 内部 action 空间和环境 action 空间不一样；
VLA backbone 冻结但 action expert 要训练；
critic head 不该跟 rollout 同步；
整个东西还要跑在 Ray + FSDP 的 worker 架构里。
```

可以把难点拆成五个：

1. QAM 不是普通 `loss.backward()`，它需要拿到中间梯度。
2. FSDP 对 `autograd.grad` 这类显式梯度路径比较敏感。
3. OpenPI flow action 是 32-D，LIBERO action 是 7-D。
4. PaliGemma / VLM 是 frozen 的，rollout sync 不应该反复导出它。
5. critic 要重新编码 observation，所以 replay 里必须保存 language。

我的解决方案是分层做：

```text
Replay:   保存 tokenized prompt
Model:    QAM_ENCODE / QAM_VELOCITY
Worker:   live f_theta + frozen f_beta + FP32 q-head
Action:   32-D flow action 和 7-D env action 显式转换
Sync:     只同步 trainable actor 参数
```

这比“我调了一个 loss”要强很多，因为它说明你真的处理了 VLA + RL 系统落地的问题。

## 高频追问和回答

### 1. 为什么 VLA 需要 RL，SFT 不够吗？

可以这样答：

```text
SFT 很重要，我不是要替代 SFT。SFT 给了一个很强的初始策略。
但 SFT 学的是模仿示范动作，不直接优化任务成功率。
机器人任务里很多失败发生在长程收尾、状态偏移、没见过的中间状态。
这些地方单靠 imitation objective 不一定能修正，所以我希望用在线 rollout 的 reward 再微调。
```

补一句会更稳：

```text
所以我的设定不是从零 RL，而是从 SFT checkpoint 出发，并且保留 frozen f_beta 作为 behavior prior。
```

### 2. 为什么不用 PPO？

不要说 PPO 不行，说“不自然”：

```text
PPO 当然是一条路，但对 OpenPI 这种 flow policy 来说不是最自然。
它不是一步输出 Gaussian action，也不是 LLM 那种 token policy。
动作是通过多步 velocity field 生成的。
如果用 PPO，要处理 action chunk 的 logprob、ratio 和 credit assignment。
QAM 更顺着 OpenPI 的结构，直接把 critic gradient 变成 velocity field 的训练信号。
```

### 3. QAM 到底在优化什么？

口语版：

```text
critic 先告诉我，最终 action 往哪个方向变会更好。
但 OpenPI 的 action 是一整条 flow 生成出来的，所以我不能只改最后那个 action。
QAM 做的事就是把最终的 Q gradient，通过 frozen behavior flow 反传回每个 flow step，
然后训练当前 velocity field 去匹配这个方向。
```

关键词：

```text
Q critic
terminal action gradient
adjoint backward through f_beta
velocity matching
stay close to SFT behavior
```

### 4. `f_beta` 为什么要冻结？

```text
f_beta 可以理解成 SFT policy 的 reference。
如果没有它，critic 一有噪声，actor 可能就被推到很奇怪的动作区域。
冻结 f_beta 的好处是：一方面 adjoint 有稳定的 reference dynamics，
另一方面 actor 更新是相对 SFT policy 的小改动，更不容易破坏原来的 VLA 能力。
```

### 5. 你怎么把它接到 OpenPI 里？

可以按四步讲：

```text
第一，我让 replay 保存语言，因为 critic 要重算 observation feature。
第二，我给 OpenPI 加了 QAM_ENCODE，用 frozen PaliGemma 编 observation。
第三，我加了 QAM_VELOCITY，让 QAM 可以在任意 flow time 调 action expert velocity。
第四，我写了 QAM worker，里面有 live f_theta、frozen f_beta、online q-head 和 target q-head。
```

### 6. 为什么 q-head 要从 FSDP OpenPI 里拆出来？

```text
一开始也尝试过放在 OpenPI/FSDP 模型里，但后来发现没必要，也不稳定。
QAM critic 的语义其实就是 q_head(pool(PaliGemma_frozen(s)), a)。
PaliGemma 是 frozen encoder，真正训练的是 q-head。
所以我把 q-head 做成 worker-owned 的 FP32 小模块。
这样 target update、checkpoint、offload 都更清楚，也避免 FSDP full state_dict 导出 frozen VLM 时出问题。
```

### 7. 32-D flow action 和 7-D env action 怎么解释？

这个点很适合突出工程理解：

```text
OpenPI action expert 内部用 32 维 action 表示，这是 flow model 的内部空间。
但 LIBERO 环境实际执行的是 7 维控制量。
QAM 的 SDE 和 velocity 必须在 32 维里跑，critic 和 env 必须在 7 维里算。
所以 terminal action 给 critic 前要 slice 到 7 维，critic 梯度回来后再 zero-pad 回 32 维。
```

一句话版：

```text
算法变量在 32-D flow space，环境控制在 7-D action space，我显式维护这个边界。
```

### 8. FSDP 下显式梯度怎么处理？

```text
QAM 需要 grad_a Q 和 adjoint VJP。
直接用 autograd.grad 在 FSDP-wrapped model 里会比较容易触发限制。
所以我换成普通 backward，然后读取输入张量的 .grad。
数学上还是同一个 vector-Jacobian product，但工程上更符合 FSDP 的正常 backward 路径。
```

不用把代码背出来，但可以记住这句：

```text
我不是绕开梯度，而是把显式 autograd.grad 改成了 FSDP 更稳定的 backward 取 input grad。
```

### 9. rollout sync 为什么要改？

```text
rollout worker 初始化时已经加载了完整 SFT checkpoint，里面包括 frozen VLM。
训练过程中变化的其实只有 action expert / projection 这些 trainable actor 参数。
critic head 只给 actor worker 训练用，也不需要同步给 rollout。
所以我改成 partial sync，只发 trainable actor 参数，不发 frozen VLM，也不发 q-head。
```

这个回答能体现你理解系统：

```text
我不是为了绕过报错才少同步，而是重新定义了 Actor 到 Rollout 的同步契约。
```

### 10. 结果怎么讲才不夸张？

建议这样说：

```text
结果是有小幅提升，但我不会说是很大的突破。
SFT base 在 LIBERO Spatial 500 条 eval 上 success_once 大概 84.6%-84.8%，
QAM step 600 到 86.0%，step 900 到 87.0%。
success_at_end 上 step 600 是 77.2%，step 900 是 76.6%。
所以 QAM 没有把 policy 推崩，也确实比 SFT 高一点；
但 600 到 900 的差距可能有统计波动，说明它已经接近平台。
```

### 11. 为什么提升有限？

这题要主动分析，不要等面试官逼问：

```text
我觉得主要有几个原因。
第一，SFT baseline 已经很高，剩余空间小。
第二，LIBERO reward 比较稀疏，critic 很难学到特别细的动作修正。
第三，成功轨迹比较多，失败边界样本少，critic 对失败动作的排序不一定足够强。
第四，我冻结了 VLM，只训 action expert，这样稳定，但表达能力也受限。
```

再补实验日志：

```text
从日志看，后半程 actor grad 和 qam_velocity_delta_abs 在上升，但 eval 没继续明显涨。
所以我判断不是继续加训练时长就能解决，而是要让 actor 更新更保守，先把 critic 学稳。
```

### 12. 下一步怎么做？

```text
我下一轮的思路是更保守，不是更激进。
把 train_actor_steps 从 100 提到 200，让 critic 多 warm-up；
critic_actor_ratio 从 8 提到 12，让 actor 少更新一点；
actor lr 从 2e-6 降到 1e-6；
replay window 从 32 扩到 50。
同时每 100 step 做 500 条 eval，减少小样本噪声。
```

如果继续扩展：

```text
再往后我会看 critic 是否真的区分成功/失败，比如按成功轨迹和失败轨迹分组看 Q 排序，
而不是只看 critic loss。因为 critic loss 低不代表它给 actor 的梯度方向就是对的。
```

### 13. 你怎么关注到 QAM 的？

```text
我是从问题倒推方法的。
我当时面对的是 OpenPI 这种 flow VLA，直接 PPO 不太自然；
同时在线 RL 又不能把 SFT policy 破坏掉。
所以我去看 diffusion / flow policy 的 RL fine-tuning 方法。
QAM 吸引我的地方是，它不是硬构造 PPO logprob，而是直接用 critic gradient 和 adjoint matching
去更新 velocity field，而且保留 frozen behavior policy。
这和 OpenPI 冻结 VLM、只训 action expert 的实践很匹配。
```

## 项目介绍模板

你可以按这个顺序讲，逻辑会很顺：

1. **背景**：VLA 先 SFT，但 SFT 不直接优化任务成功率。
2. **问题**：OpenPI 是 flow policy，普通 PPO 不自然。
3. **方法**：QAM 用 critic gradient + adjoint matching 更新 velocity field。
4. **落地**：replay language、QAM forward type、QAM worker、FP32 q-head、partial sync。
5. **最大难点**：显式梯度 + FSDP + action space 映射 + frozen VLM sync。
6. **结果**：SFT 84.6%-84.8%，QAM 86%-87%，小幅提升无 collapse。
7. **反思**：提升有限，下一步更保守 actor、更强 critic 诊断。

## 最推荐背下来的完整回答

```text
我这个项目是把 QAM 接到 RLinf 的 OpenPI π0.5 VLA 训练栈里。
背景是 VLA 通常先做 SFT，SFT 能给一个很强的初始策略，但它优化的是动作拟合，
不是任务成功率。OpenPI 又是 flow policy，动作通过 velocity field 生成，
所以直接套 PPO/GRPO 并不是最自然。

QAM 的思路是，用 replay 学一个 critic Q(s,a)，然后把终端 action 上的 Q gradient，
通过 frozen SFT behavior policy f_beta 的 adjoint 反传到整条 flow trajectory 上。
actor 训练的不是最终 action，而是当前 velocity f_theta 相对 f_beta 的更新方向。
这样既能用 reward，又能尽量保留 SFT 的动作流形。

我落地时主要做了几件事：replay 保存 language，OpenPI 增加 QAM_ENCODE 和 QAM_VELOCITY，
worker 里维护 live f_theta、frozen f_beta 和 FP32 online/target q-head。
最大难点是系统适配。QAM 需要显式 grad_a Q 和 adjoint VJP，FSDP 下直接 autograd.grad 不稳定；
OpenPI 内部 action 是 32 维，但 LIBERO 环境 action 是 7 维；
同时 frozen VLM 不应该参与 rollout sync。
所以我把 q-head 移出 FSDP，用普通 backward 读 input grad，只同步 trainable actor 参数，
并显式处理 32-D flow action 和 7-D env action 的转换。

结果上，LIBERO Spatial 500 条 eval 里，SFT base success_once 大概 84.6%-84.8%，
QAM checkpoint 到 86%-87%。这个结果说明方法没有把 policy 推崩，也有小幅收益，
但提升不算大。我分析主要是 baseline 已经强、reward 稀疏、失败边界样本少，
critic signal 还不够细。所以后续我把 run2 改得更保守：更长 critic warm-up、
更低 actor lr、更大的 critic_actor_ratio，并用 500 条 eval 来选 checkpoint。
```


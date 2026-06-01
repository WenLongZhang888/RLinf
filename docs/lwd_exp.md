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

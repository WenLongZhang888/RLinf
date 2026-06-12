# Marvin 真机使用说明

本文记录当前 RLinf Marvin B 臂的硬件配置、USB 夹爪、相机、SpaceMouse、控制模式和 material 任务采集命令。

## 硬件配置

- 机械臂：`B`
- 机器人 IP：`192.168.71.190`
- USB 夹爪串口：`/dev/ttyUSB0`
- USB 夹爪波特率：`115200`
- 已验证相机：
  - `side_policy` / `side_classifier`: `335122271065`
- material 任务三相机配置：
  - `wrist_1`: `230322273834`
  - `wrist_2`: `230322274885`
  - `side_policy`: `335122271065`
  - `side_classifier`: `335122271065`，复用 `side_policy` 相机

## 环境变量

基础环境：

```bash
export ROBOT_IP=192.168.71.190
export MARVIN_ROBOT_IP=192.168.71.190
export MARVIN_ARM_ID=B
export MARVIN_CONTROL_MODE=position
```

USB 夹爪：

```bash
export MARVIN_GRIPPER_BACKEND=usb
export MARVIN_GRIPPER_SERIAL_PORT=/dev/ttyUSB0
export MARVIN_GRIPPER_BAUDRATE=115200
export MARVIN_GRIPPER_OPEN_HEX_B="01 10 00 02 00 02 04 00 00 00 00 72 76"
export MARVIN_GRIPPER_CLOSE_HEX_B="01 10 00 02 00 02 04 70 41 00 00 38 A2"
```

如果夹爪上电后需要初始化，可额外设置：

```bash
export MARVIN_GRIPPER_INIT_HEX="01 06 00 00 00 01 48 0A"
```

## USB 夹爪

当前夹爪是 USB-RS485 直连本机，不走机械臂末端工具通道。已验证命令：

```text
open:
01 10 00 02 00 02 04 00 00 00 00 72 76

close:
01 10 00 02 00 02 04 70 41 00 00 38 A2
```

单独测试夹爪打开、闭合、打开：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -c '
import os
import time

from rlinf.envs.realworld.marvin.marvin_usb_gripper import MarvinUsbGripper

arm = os.getenv("MARVIN_ARM_ID", "B").upper()
gripper = MarvinUsbGripper.from_env(arm)

try:
    print("open:", gripper.open())
    time.sleep(1.0)
    print("close:", gripper.close())
    time.sleep(1.0)
    print("open:", gripper.open())
finally:
    gripper.cleanup()
'
```

## 机器人连接与模式

测试 SDK 连接：

```bash
UV_CACHE_DIR=/tmp/uv-cache ROBOT_IP=192.168.71.190 uv run --no-sync python -c '
import os
from rlinf.envs.realworld.marvin.marvin_sdk import load_marvin_sdk_symbols

Robot, *_ = load_marvin_sdk_symbols()
robot = Robot()
print("connect:", robot.connect(robot_ip=os.environ["ROBOT_IP"]))
'
```

切到位置模式前，如果状态不干净，先下使能、清错、再切位置：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -c '
import os
import time

from rlinf.envs.realworld.marvin.marvin_sdk import (
    ARM_INDEX,
    ARM_STATE_POSITION,
    load_marvin_sdk_symbols,
)

arm = os.getenv("MARVIN_ARM_ID", "B").upper()
idx = ARM_INDEX[arm]
robot_ip = os.environ["ROBOT_IP"]

Robot, DCSS, *_ = load_marvin_sdk_symbols()
robot = Robot()
print("connect:", robot.connect(robot_ip=robot_ip))
dcss = DCSS()

def read(label):
    data = robot.subscribe(dcss)
    st = data["states"][idx]
    print(label, "cur_state:", st["cur_state"], "cmd_state:", st.get("cmd_state"), "err_code:", st.get("err_code"))
    return data

def wait_state(target, label, timeout=8.0):
    deadline = time.time() + timeout
    i = 0
    while time.time() < deadline:
        data = read(f"{label}[{i}]")
        if data["states"][idx]["cur_state"] == target:
            return True
        i += 1
        time.sleep(0.2)
    return False

print("down servo")
print("clear_set idle:", robot.clear_set())
print("set_state_idle:", robot.set_state(arm=arm, state=0))
print("send_cmd idle:", robot.send_cmd())
print("IDLE_REACHED:", wait_state(0, "idle", timeout=5.0))

print("clear_error:", robot.clear_error(arm))
time.sleep(0.5)

print("clear_set setup:", robot.clear_set())
print("set_vel_acc:", robot.set_vel_acc(arm=arm, velRatio=5, AccRatio=5))
print("send_cmd setup:", robot.send_cmd())
time.sleep(0.3)

print("clear_set mode:", robot.clear_set())
print("set_state_position:", robot.set_state(arm=arm, state=ARM_STATE_POSITION))
print("send_cmd mode:", robot.send_cmd())
print("POSITION_REACHED:", wait_state(ARM_STATE_POSITION, "position", timeout=10.0))

try:
    robot.release_robot()
except Exception:
    pass
'
```

## 相机测试

单相机读取：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -c '
from rlinf.envs.realworld.common.camera import CameraInfo, create_camera

serial = "335122271065"
camera = create_camera(CameraInfo(name="side_policy", serial_number=serial))

try:
    print("open camera:", serial)
    camera.open()
    for i in range(10):
        frame = camera.get_frame()
        print(f"frame[{i}] shape:", frame.shape, "dtype:", frame.dtype, "min:", frame.min(), "max:", frame.max())
finally:
    camera.close()
'
```

## SpaceMouse 测试

只读 SpaceMouse，不控制机械臂：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -c '
import time
import numpy as np

from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert
from rlinf.envs.realworld.marvin.wrappers import SPACEMOUSE_WIRELESS_REMAP

mouse = SpaceMouseExpert(axis_remap=SPACEMOUSE_WIRELESS_REMAP)
print("axis_remap:", SPACEMOUSE_WIRELESS_REMAP)
print("action = [tcp_x, tcp_y, tcp_z, rx, ry, rz]")

try:
    while True:
        action, buttons = mouse.get_action()
        print("action:", np.round(action, 3).tolist(), "buttons:", buttons)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    mouse.close()
'
```

当前 Marvin SpaceMouse 映射：

```text
action = [tcp_x, tcp_y, tcp_z, rx, ry, rz]
new_x = -raw_x
new_y = raw_z
new_z = raw_y
```

## Material 任务采集

新任务 ID：

```text
MarvinMaterialGrabAndPlaceEnv-v1
```

配置文件：

```text
examples/embodiment/config/marvin_realworld_material_collect_data.yaml
examples/embodiment/config/env/marvin_realworld_material_grab_and_place.yaml
```

启动采集：

```bash
UV_CACHE_DIR=/tmp/uv-cache bash examples/embodiment/collect_data.sh marvin_realworld_material_collect_data
```

默认行为：

- 保留 `tianji_hilserl` 的 material reset 流程。
- 启动或 reset 时会自动执行取放 reset 轨迹。
- 使用 USB 夹爪。
- 使用三相机配置。
- 默认启用二分类器。
- `MARVIN_CONTROL_MODE` 控制启动模式，默认建议先用 `position`。

如需临时关闭二分类器：

```bash
UV_CACHE_DIR=/tmp/uv-cache bash examples/embodiment/collect_data.sh marvin_realworld_material_collect_data \
  env.eval.override_cfg.classifier.enabled=False
```

如需切换阻抗模式：

```bash
export MARVIN_CONTROL_MODE=impedance
```

## Peg Insertion 任务

旧采集配置仍然可用：

```bash
UV_CACHE_DIR=/tmp/uv-cache bash examples/embodiment/collect_data.sh marvin_realworld_collect_data \
  runner.num_data_episodes=1 \
  env.eval.use_spacemouse=True \
  env.eval.no_gripper=False \
  +env.eval.override_cfg.controller_type=position \
  +env.eval.keyboard_reward_wrapper=single_stage \
  env.eval.max_episode_steps=300 \
  env.eval.data_collection.enabled=False
```

注意：`marvin_realworld_collect_data` 默认启动 `MarvinPegInsertionEnv-v1`，其 `reset()` 会主动移动到任务 reset pose。

## 注意事项

- 不要在普通脚本里直接 `MarvinController(...)`；它是 RLinf `Worker`，应通过 Ray/WorkerGroup 启动。硬件 smoke test 可直接用 `Marvin` SDK 或使用专门绕过 Worker 的测试脚本。
- material 任务默认会自动运动，启动前确认工作空间安全。
- 二分类器依赖 `tianji_hilserl` 的 `serl_launcher/jax` 环境和 classifier checkpoint；缺依赖时可先关闭 classifier。
- 如果 `MARVIN_GRIPPER_BACKEND=usb` 报 `pyserial is required`，安装 `pyserial` 或 Marvin extra。
- 如果夹爪无动作，检查 `/dev/ttyUSB0`、串口权限、USB-RS485 接线、波特率和 init 命令。

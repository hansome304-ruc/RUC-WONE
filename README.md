# RUC-WONE

比赛代码入口。目录分为三块：

```text
RUC-WONE/
├── server/                 # 机械臂 airbot_server、airbot_sdk、ZeroGrasp server bundle
├── manipulation/
│   ├── agentic/            # Agentic Grasp: perception + grasp skill
│   └── vla/                # VLA 占位，当前不启用
└── locomotion/             # 底盘移动控制 + 升降控制
```

## 1. 启动机械臂服务

比赛/Agentic Grasp 默认只需要两个执行臂：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh execution
```

检查服务：

```bash
bash server/check_airbot_servers.sh execution
tmux attach -t airbot_servers
```

默认端口：

| 服务 | CAN 口 | 端口 |
|---|---|---:|
| 左主臂 | `can_left_lead` | `50050` |
| 左执行臂 | `can_left` | `50051` |
| 右主臂 | `can_right_lead` | `50052` |
| 右执行臂 | `can_right` | `50053` |

常用模式：

```bash
bash server/start_airbot_servers.sh execution   # 只启动左/右执行臂，比赛默认
bash server/start_airbot_servers.sh all         # 启动主臂+执行臂，用于主从遥操作
bash server/start_airbot_servers.sh lead        # 只启动主臂
```

主从遥操作不会自动启动。需要四个 arm server 都 ready 后再手动运行：

```bash
bash server/start_airbot_servers.sh all
bash server/start_teleop_follow.sh
```

主从分离时，主臂机器只启动 lead server，机器人/执行臂机器启动 execution server 和 follow loop：

```bash
# 主臂机器
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh lead

# 机器人/执行臂机器
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh execution
LEAD_URL=192.168.x.x bash server/start_teleop_follow.sh
```

先在执行臂机器上确认能访问主臂端口：

```bash
AIRBOT_SERVER_HOST=192.168.x.x AIRBOT_SERVER_PORTS="50050 50052" \
  bash server/check_airbot_servers.sh
```

停止：

```bash
bash server/stop_airbot_servers.sh
```

## 2. 控制底盘

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/movebase.sh doctor
bash locomotion/scripts/movebase.sh state
bash locomotion/scripts/movebase.sh stop
```

前进约 `0.1m`：

```bash
bash locomotion/scripts/movebase.sh forward --speed 0.05 --duration 2.0 --yes
```

常用动作：

```bash
bash locomotion/scripts/movebase.sh backward --speed 0.05 --duration 1.0 --yes
bash locomotion/scripts/movebase.sh turn-left --yaw-speed 0.15 --duration 1.0 --yes
bash locomotion/scripts/movebase.sh turn-right --yaw-speed 0.15 --duration 1.0 --yes
```

参数：

- `--speed`: 线速度，单位 `m/s`
- `--yaw-speed`: 角速度，单位 `rad/s`
- `--duration`: 运动时间，单位 `s`
- `--max-linear`: 最大允许线速度，默认 `0.20`
- `--max-angular`: 最大允许角速度，默认 `0.50`
- `--ros-master-uri`: 底盘 ROS1 master，默认 `http://192.168.31.7:11311/`

升降/升降柱通过独立 USB 串口控制：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/lift.sh status --timeout 1
bash locomotion/scripts/lift.sh stop
bash locomotion/scripts/lift.sh goto --position 200 --yes
```

## 3. 运行 Agentic Grasp

先确认 ZeroGrasp perception server 已启动。默认机器人端访问：

```text
http://127.0.0.1:9100
```

在有 `zerograsp:latest` 镜像和完整 ZeroGrasp 模型文件的 GPU 服务器上启动：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_zerograsp_server.sh
bash server/check_zerograsp_server.sh
```

RUC-WONE 已经包含 ZeroGrasp 的服务端代码和 Docker 环境补丁：

```text
/home/ubuntu/RUC-WONE/server/zerograsp_server_bundle
```

注意：真实 `/grasp` 仍需要 GPU 服务器上有完整 ZeroGrasp checkout、`configs/demo.yaml` 和 checkpoint。这个 bundle 应同步到完整 repo 根目录后启动。

如果 ZeroGrasp 跑在 `zjlab` 远端 GPU 服务器，推荐目录是：
如果本机没有配置 `ssh zjlab` 别名，就把下面命令里的 `zjlab` 换成 `user@10.47.41.144`。

```text
/home/user/zyh/
├── RUC-WONE/
└── ZeroGrasp/
```

平时启动，使用已经下载好的 HuggingFace 缓存，不重新下载模型：

```bash
ssh zjlab "cd /home/user/zyh/RUC-WONE && git pull --ff-only && bash server/setup_zerograsp_overlay.sh --offline-cache --start --gpu 0"
```

第一次构建镜像，或 Dockerfile/依赖变了，再加 `--build`：

```bash
ssh zjlab "cd /home/user/zyh/RUC-WONE && bash server/setup_zerograsp_overlay.sh --offline-cache --build --start --gpu 0"
```

看日志，出现 `perception_server ready` 表示 warmup 完成：

```bash
ssh zjlab "docker logs -f perception-server"
```

机器人端把远端 `9100` 映射到本地：

```bash
autossh -M 0 -fN -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 9100:localhost:9100 user@10.47.41.144
```

```bash
cd /home/ubuntu/RUC-WONE/manipulation/agentic
conda activate dos-w1
cp .env.example .env
pip install -e .
```

编辑 `.env`，至少确认：

```text
PERCEPTION_URL=http://127.0.0.1:9100
LEFT_ARM_PORT=50051
RIGHT_ARM_PORT=50053
CALIB_DIR=/home/ubuntu/calib/result
```

测试感知服务：

```bash
python scripts/00_test_health.py
python scripts/01_test_segment.py --prompt "a green bottle" --save-rgb
```

抓取干跑，不动机械臂：

```bash
python scripts/run_scripted.py --prompt "a bottle" --arm right --strategy default --dry-run
```

真实抓取：

```bash
python scripts/run_scripted.py --prompt "a bottle" --arm right --strategy default --speed SLOW --record-video
```

调试输出默认写到 `debug_out/`：

- `segment_overlay.png`: 分割 mask 和 bbox
- `segment_overlay_rgb.png`: `--save-rgb` 保存的原始 RGB
- `sam_input.png`: 抓取流水线输入图
- `grasp_segment_overlay.png`: 抓取流水线中的分割结果
- `selected_grasp.png`: 选中抓取、预抓取点和抬升点
- `selected_grasp.json`: 抓取位姿、base 坐标和方向轴
- `grasp_run_*.mp4`: 加 `--record-video` 后保存的抓取过程视频

常用抓取参数：

- `--prompt`: 目标物体文本，例如 `"a bottle"`
- `--arm`: `left` 或 `right`
- `--strategy`: `default`、`low-bottle`、`side-bottle`
- `--endpoint`: `/grasp` 使用 ZeroGrasp，`/grasp_dummy` 使用几何回退
- `--dry-run`: 只规划不运动
- `--speed`: `SLOW`、`DEFAULT`、`FAST`
- `--lift-height`: 抓住后机械臂末端上抬距离，单位 `m`，默认 `0.10`

`manipulation/agentic/scripts` 根目录只保留比赛入口；标定、坐标系和可视化诊断脚本在 `scripts/debug/`。

## 4. 比赛前检查

```bash
cd /home/ubuntu/RUC-WONE
bash server/check_airbot_servers.sh execution
bash locomotion/scripts/movebase.sh doctor
bash server/check_zerograsp_server.sh
cd manipulation/agentic && python scripts/00_test_health.py
```

比赛默认看到机械臂 `50051`、`50053` ready，底盘 `doctor` 正常、感知 `/healthz` 正常后，再运行真实运动或抓取。只有做主从遥操作时才需要 4 个端口都 ready。

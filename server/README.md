# server

机械臂服务、本地 `airbot_sdk` 和 ZeroGrasp perception server bundle。

## 启动

比赛/Agentic Grasp：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh execution
```

默认 `execution` 只启动两个执行臂：

- `can_left:50051`
- `can_right:50053`

主从遥操作如果四条 CAN 都接在同一台机器，需要四个 `airbot_server`：

```bash
bash server/start_airbot_servers.sh all
bash server/start_teleop_follow.sh
```

`all` 模式端口：

- `can_left_lead:50050`
- `can_left:50051`
- `can_right_lead:50052`
- `can_right:50053`

## 检查与停止

```bash
bash server/check_airbot_servers.sh execution
tmux attach -t airbot_servers
bash server/stop_airbot_servers.sh
```

## 参数覆盖

```bash
LEFT_PORT=50151 RIGHT_PORT=50153 bash server/start_airbot_servers.sh execution
STOP_AIRBOT_DOCKERS=1 bash server/start_airbot_servers.sh execution
```

如果当前机器没有 `can_left_lead` / `can_right_lead`，不要用 `all`；使用 `execution` 即可。

## 主从分离遥操作

如果主臂接在另一台主机上，推荐直接复用 `airbot_server` 的网络能力，不需要另写 websocket。结构是：

```text
主臂机器: can_left_lead/can_right_lead -> airbot_server :50050/:50052
执行臂机器: can_left/can_right -> airbot_server :50051/:50053
执行臂机器运行 follow loop，读取远端主臂，伺服本机执行臂
```

主臂机器上：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh lead
bash server/check_airbot_servers.sh lead
```

执行臂/机器人机器上：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_airbot_servers.sh execution

# 把 192.168.x.x 换成主臂机器 IP
AIRBOT_SERVER_HOST=192.168.x.x AIRBOT_SERVER_PORTS="50050 50052" \
  bash server/check_airbot_servers.sh

LEAD_URL=192.168.x.x bash server/start_teleop_follow.sh
tmux attach -t airbot_teleop
```

如果左右主臂不在同一台机器，分别指定：

```bash
LEFT_LEAD_URL=192.168.x.10 RIGHT_LEAD_URL=192.168.x.11 \
  bash server/start_teleop_follow.sh
```

延迟要求：两台机器最好走有线局域网，保证能从执行臂机器访问主臂机器的 `50050`、`50052` 端口。跨公网或不稳定 Wi-Fi 不建议直接遥操作。

`server/airbot_api` 是机械臂 Python API 源码，可在需要时安装：

```bash
cd server/airbot_api
pip install -e .
```

## ZeroGrasp Perception Server

ZeroGrasp server 不在机械臂 `airbot_server` 里，它是单独的 FastAPI 服务，封装
SAM + GroundingDINO + ZeroGrasp。机器人端默认访问：

```text
http://127.0.0.1:9100
```

服务端代码已经整合在：

```text
/home/ubuntu/RUC-WONE/server/zerograsp_server_bundle
```

这个目录包含 FastAPI server、Dockerfile、依赖约束、启动脚本和测试脚本。真实 `/grasp` 还需要 GPU 服务器上有完整 ZeroGrasp checkout、`configs/demo.yaml` 和 checkpoint；所以推荐把这个 bundle 同步到完整 repo 根目录。

### 极速上手

如果 ZeroGrasp 跑在远端 GPU 服务器，先从机器人/本地机同步 bundle：

```bash
rsync -avzh --progress --exclude="__pycache__" --exclude=".git" \
  /home/ubuntu/RUC-WONE/server/zerograsp_server_bundle/ \
  user@10.42.115.70:~/zyh/ZeroGrasp/
```

然后在 GPU 服务器构建镜像并启动容器：

```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && ./docker/build.sh"
```

已有 `zerograsp:latest` 镜像时可以跳过 build，直接启动：

```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && DETACH=1 ./docker/run_server.sh 0"
```

查看日志，看到 `perception_server ready` 表示 warmup 完成：

```bash
ssh user@10.42.115.70 "docker logs -f perception-server"
```

机器人端建立隧道：

```bash
autossh -M 0 -fN -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 9100:localhost:9100 user@10.42.115.70
```

测试：

```bash
cd /home/ubuntu/RUC-WONE/manipulation/agentic
conda activate dos-w1
python scripts/00_test_health.py
python scripts/01_test_segment.py --prompt "a green bottle"
python scripts/run_scripted.py --dry-run --prompt "a bottle"
```

真实抓取：

```bash
python scripts/run_scripted.py --prompt "a bottle"
```

### 本机/GPU 服务器启动脚本

真实 `/grasp` 模型需要完整 ZeroGrasp checkout，里面应包含 `configs/demo.yaml` 和
`checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt`。如果当前在机器人本地运行
RUC-WONE 脚本，但模型 repo 在 GPU 服务器或其它路径，启动时指定完整 repo：

```bash
ZEROGRASP_BUNDLE_DIR=~/zyh/ZeroGrasp bash server/start_zerograsp_server.sh
```

如果这台机器/当前 GPU 服务器已经有 `zerograsp:latest` 镜像和完整模型文件：

```bash
cd /home/ubuntu/RUC-WONE
bash server/start_zerograsp_server.sh
bash server/check_zerograsp_server.sh
docker logs -f perception-server
```

停止：

```bash
bash server/stop_zerograsp_server.sh
```

只调客户端链路时可以用 stub 模式，它不需要 checkpoint/config，但仍然需要 Docker 镜像存在：

```bash
SERVER_STUB=1 bash server/start_zerograsp_server.sh
```

如果 `start_zerograsp_server.sh` 提示 `Docker image not found: zerograsp:latest`，需要先在 GPU 服务器的完整 ZeroGrasp checkout 里构建：

```bash
cd ~/zyh/ZeroGrasp
./docker/build.sh
```

机器人端 `.env` 里保持：

```text
PERCEPTION_URL=http://127.0.0.1:9100
```

如果 ZeroGrasp 跑在远端 GPU 服务器，先建立 SSH 隧道或把 `.env` 改成远端地址。

### 日常运维

修改 `perception_server/*.py` 后，代码目录由 Docker bind-mount 进容器，不需要重新 build，重启容器即可：

```bash
ssh user@10.42.115.70 "docker restart perception-server"
```

如果修改了 `requirements-server.txt` 或 `constraints.txt`，需要重新 build：

```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && ./docker/build.sh && docker restart perception-server"
```

启动 stub 模式，不占 GPU，只调客户端链路：

```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && docker rm -f perception-server && SERVER_STUB=1 DETACH=1 ./docker/run_server.sh 0"
```

### 接口契约

四个 endpoint，全部 JSON。图像数组用 base64 的原始字节，不是 PNG/JPEG。

| 接口 | 请求 | 响应 |
|---|---|---|
| `GET /healthz` | 无 | `{ok, ts, stub, model_loaded, cuda_mem_mb}` |
| `POST /segment` | `{rgb_b64, H, W, prompt? 或 bbox?}` | `{bbox, mask_b64, score}` |
| `POST /grasp` | `{rgb_b64, depth_b64, mask_b64, K, H, W, top_k}` | `{grasps: [{T_cam, width, score, depth}]}` |
| `POST /grasp_dummy` | 同 `/grasp` | 几何回退抓取 |

抓取位姿约定：

```text
T_cam[:3, 3]   = 抓取中心，相机坐标系，单位 m
T_cam[:3, 2]   = approach 方向，gripper 指向物体
T_cam[:3, 0:2] = 与 approach 垂直的 gripper 张合面
```

ZeroGrasp/GraspNet 原始约定里 approach 是旋转矩阵第 0 列；server 内部已在
`perception_server/grasp.py::_SWAP_GN_TO_CLIENT` 转成客户端期望的第 2 列。

### 环境变量

| Env | 默认 | 用途 |
|---|---|---|
| `SERVER_PORT` / `PORT` | `9100` | 容器外端口 |
| `SERVER_STUB` | `0` | `1` 表示不加载模型，返回假数据 |
| `SERVER_WARMUP` | `1` | `0` 表示不跑 warmup |
| `SERVER_REQUIRE_AUTH_TOKEN` | 空 | 设置后 `/segment`、`/grasp` 要 Bearer token |
| `ZEROGRASP_CHECKPOINT` | `checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt` | ZeroGrasp checkpoint |
| `ZEROGRASP_CONFIG` | `configs/demo.yaml` | ZeroGrasp config |
| `GD_MODEL` | `IDEA-Research/grounding-dino-tiny` | GroundingDINO 模型 |
| `SAM_MODEL` | `facebook/sam-vit-base` | SAM 模型 |
| `LOG_LEVEL` | `info` | uvicorn/server 日志级别 |

换 SAM 模型示例：

```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && docker rm -f perception-server && SAM_MODEL=facebook/sam-vit-large DETACH=1 ./docker/run_server.sh 0"
```

### 常见问题

| 现象 | 常见原因 | 处理 |
|---|---|---|
| `/healthz` 连不上 | 容器没启动或端口未转发 | `docker logs -f perception-server`，确认 SSH tunnel |
| `_ARRAY_API not found` / `numpy.dtype size changed` | numpy 2.x ABI 与 torch 2.2 不兼容 | `constraints.txt` 保持 `numpy<2` 后 rebuild |
| `Autotuner.__init__() got an unexpected keyword argument 'pre_hook'` | ocnn 版本过新但 triton 是 2.2 | Dockerfile 保持 `ocnn==2.2.4` |
| `OctreeFeatureExtractor.forward() missing ...` | submodule 是旧 API | `cd submodules/octree_feature_extractor && git checkout main` 后 rebuild |
| `PyTorch >= 2.4 is required but found 2.2.0` | transformers 5.x 不支持 torch 2.2 | `requirements-server.txt` 钉 `transformers<5` |
| HuggingFace 下载超时 | GPU 服务器网络问题 | `docker/run_server.sh` 已设 `HF_ENDPOINT=https://hf-mirror.com`，仍失败就检查远端网络 |
| `/segment` mask 全黑 | prompt 未检测到目标 | 用英文短语，或直接传 bbox |
| `/grasp` 返回 0 个抓取 | mask 太小、depth 全 0、碰撞过滤全删 | 看 `docker logs` 中 grasp/collision 相关日志 |

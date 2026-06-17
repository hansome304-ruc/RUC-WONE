# Perception Server (感知服务)

FastAPI 服务，封装了 **SAM + GroundingDINO + ZeroGrasp**，为机器人端（如 `agentic_grasp`）提供高效、常驻 GPU 的目标分割与抓取位姿预测服务。

---

## 💡 核心实现逻辑 (How it works)

本服务的设计目标是**低延迟、高内聚、易调试**，其核心逻辑可以概括为以下几点：

1. **单容器、模型常驻**：
   - 将 SAM（目标分割）、GroundingDINO（文本检测）与 ZeroGrasp（抓取预测）合并在同一个 Docker 容器和 Python 进程中。
   - 启动时一次性将所有模型加载至 GPU 显存，避免每次请求重复加载的开销，实现毫秒级/秒级快速响应。
2. **单 Worker 串行排队（防 OOM）**：
   - 内部使用 `asyncio.Lock` 保证 GPU 推理串行执行，即使客户端并发调用，也会排队处理，防止多模型并发抢占显存导致 GPU OOM。
3. **预热机制 (Warmup)**：
   - 容器启动生命周期中会自动跑一次 Demo 图像推理，提前把 Octree CUDA 内核进行 JIT (Just-In-Time) 编译，确保机器人第一次发送真实请求时不会遭遇长达数十秒的编译卡顿。
4. **统一坐标系**：
   - 服务端内部自动将 GraspNet 的坐标系（approach 为 x 轴）转换为机器人客户端期望的标准相机坐标系（approach 为 z 轴），简化客户端的控制逻辑。

---

## 🚀 极速上手：像跑脚本一样运行 (Quick Start)

只需以下几步，即可完成服务的部署、运行与测试。

### 第一步：在 GPU 服务器上启动服务

在本地开发机（有本 bundle 的机器）上，一键同步代码并启动服务：

```bash
# 1. 一键同步 RUC-WONE 内置 server bundle 到 GPU 服务器 (10.42.115.70)
rsync -avzh --progress --exclude="__pycache__" --exclude=".git" \
  /home/ubuntu/RUC-WONE/server/zerograsp_server_bundle/ \
  user@10.42.115.70:~/zyh/ZeroGrasp/

# 如 GPU 服务器还没有 zerograsp:latest 镜像，先构建
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && ./docker/build.sh"

# 2. 登录 GPU 服务器并启动容器（默认使用 GPU 0，端口 9100）
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && DETACH=1 ./docker/run_server.sh 0"
```

> 💡 **如何看服务是否就绪？**
> 运行以下命令查看日志，看到 `perception_server ready` 出现即表示 Warmup 完成，服务已就绪：
> ```bash
> ssh user@10.42.115.70 "docker logs -f perception-server"
> ```

---

### 第二步：在机器人端一键运行测试

在机器人本机上，建立端口转发并直接运行测试脚本：

```bash
# 1. 建立 SSH 隧道（将远端 9100 端口映射到本地 9100 端口）
autossh -M 0 -fN -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 9100:localhost:9100 user@10.42.115.70

# 2. 进入环境并运行测试脚本
cd /home/ubuntu/RUC-WONE/manipulation/agentic && conda activate dos-w1

# 运行心跳测试（应输出 ok: true, stub: false）
python scripts/00_test_health.py

# 运行分割测试（会在 debug_out/ 下保存分割效果图）
python scripts/01_test_segment.py --prompt "a green bottle"

# 运行抓取预测测试（Dry-run 模式，不实际控制机械臂）
python scripts/run_scripted.py --dry-run --prompt "a bottle"

# 运行真实抓取（控制机械臂进行抓取）
python scripts/run_scripted.py --prompt "a bottle"
```

---

## 🛠️ 日常运维与代码修改 (Operations)

日常开发中，你只需要记住以下 3 条最常用的命令：

### 1. 修改代码后热重启
由于代码目录是通过 Docker Bind-Mount 挂载的，修改 `perception_server/*.py` 后不需要重新 Build 镜像，只需重启容器重新加载模型即可：
```bash
ssh user@10.42.115.70 "docker restart perception-server"
```

### 2. 依赖变更后重新 Build
如果修改了 `requirements-server.txt` 或 `constraints.txt`，需要重新编译镜像并重启：
```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && ./docker/build.sh && docker restart perception-server"
```

### 3. 启动 Stub 模式（不占 GPU，纯调客户端逻辑）
如果只想调试机器人端的客户端代码，不想占用 GPU 资源，可以启动 Stub 虚拟服务：
```bash
ssh user@10.42.115.70 "cd ~/zyh/ZeroGrasp && docker rm -f perception-server && SERVER_STUB=1 DETACH=1 ./docker/run_server.sh 0"
```

---

## 📖 进阶参考 (Advanced Reference)

以下内容包含完整的接口契约、文件结构、环境变量和故障排查等细节，供深入开发和排坑时查阅。

<details>
<summary>🔍 点击展开：文件结构与接口契约</summary>

### 文件结构

```
GPU 服务器 ~/zyh/ZeroGrasp/                  # 完整 ZeroGrasp repo + 本 bundle 覆盖，bind-mount 进容器 /opt/app
├── docker/
│   ├── Dockerfile                          # 镜像配方
│   ├── build.sh                            # docker build
│   └── run_server.sh                       # ★ 启动容器
├── constraints.txt                          # 版本钉子: numpy<2, triton==2.2.0
├── requirements-server.txt                  # FastAPI + transformers
├── serve.sh                                 # 容器内 uvicorn entry
├── perception_server/                       # ★ 业务代码（大多改动只在这里）
│   ├── server.py                           # FastAPI app + 4 个 endpoints
│   ├── grasp.py                            # ZeroGrasp 包装 + 坐标系转换
│   ├── segment.py                          # SAM + GroundingDINO 包装
│   ├── schemas.py                          # Pydantic wire schemas
│   └── encoding.py                         # base64 ↔ ndarray
├── e2e_test.py                              # 容器内自测脚本
├── checkpoints/mirage_cvpr2025/mirage/      # ZeroGrasp ckpt (~214 MB)
│   └── epoch=1-step=80000.ckpt
└── （上游 ZeroGrasp 原仓库的其它文件）
```

### 接口契约

四个 endpoint，全部 JSON。图像数组用 base64 的**原始字节**（不是 PNG/JPEG）。

| 接口 | 请求 | 响应 | 典型耗时 |
|---|---|---|---|
| `GET  /healthz` | — | `{ok, ts, stub, model_loaded, cuda_mem_mb}` | <10 ms |
| `POST /segment` | `{rgb_b64, H, W, prompt? \| bbox?}` | `{bbox, mask_b64, score}` | ~0.6 s |
| `POST /grasp` | `{rgb_b64, depth_b64, mask_b64, K, H, W, top_k}` | `{grasps: [{T_cam, width, score, depth}, ...]}` | ~2.4 s |
| `POST /grasp_dummy` | 同 `/grasp` | 单 centroid 几何抓取 | ~40 ms |

### Grasp Pose 坐标系

```
T_cam[:3, 3]      = 抓取中心 (fingertip 中点)，相机坐标系下，单位 m
T_cam[:3, 2]      = approach 方向（gripper 指向物体的方向）
T_cam[:3, 0:2]    = 与 approach 垂直的平面（gripper 张合面）
```

GraspNet 自己的 R 第 0 列才是 approach，server 内部已经做了基变换 (`grasp.py::_SWAP_GN_TO_CLIENT`)。

</details>

<details>
<summary>⚙️ 点击展开：环境变量配置</summary>

`docker/run_server.sh` 支持通过环境变量进行灵活配置：

| Env | 默认 | 干什么 |
|---|---|---|
| `SERVER_PORT` | 9100 | 容器外端口 |
| `SERVER_STUB` | 0 | 1 = 不加载模型，返回假数据 |
| `SERVER_WARMUP` | 1 | 0 = 不跑 warmup（debug 时偶尔用） |
| `SERVER_REQUIRE_AUTH_TOKEN` | "" | 设了之后 `/segment` `/grasp` 要 `Authorization: Bearer <token>` |
| `ZEROGRASP_CHECKPOINT` | `checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt` | 想换 ckpt 时改这个 |
| `ZEROGRASP_CONFIG` | `configs/demo.yaml` | ZeroGrasp config 路径 |
| `GD_MODEL` | `IDEA-Research/grounding-dino-tiny` | HF model id |
| `SAM_MODEL` | `facebook/sam-vit-base` | HF model id |
| `LOG_LEVEL` | `info` | uvicorn / perception_server 日志级别 |

例：要换大一点的 SAM 模型：
```bash
docker rm -f perception-server
SAM_MODEL=facebook/sam-vit-large DETACH=1 ./docker/run_server.sh 0
```

</details>

<details>
<summary>🚨 点击展开：故障排查与已知问题</summary>

### 1. Server 起不来 / `/healthz` 503
运行 `docker logs --tail 100 perception-server` 查看日志。

| 关键字 | 原因 | 修法 |
|---|---|---|
| `_ARRAY_API not found` / `numpy.dtype size changed` | numpy 2.x ABI 跟 torch 2.2 不兼容 | `constraints.txt` 里要有 `numpy<2`，rebuild |
| `Autotuner.__init__() got an unexpected keyword argument 'pre_hook'` | ocnn 装到 2.3+，但当前 triton 是 2.2 | Dockerfile 里要有 `RUN pip install ... ocnn==2.2.4` 覆盖 |
| `OctreeFeatureExtractor.forward() missing 2 required positional arguments` | submodule 在 `d602008`（旧 API） | `cd submodules/octree_feature_extractor && git checkout main`，rebuild |
| `PyTorch >= 2.4 is required but found 2.2.0` | transformers 5.x 不支持 torch 2.2 | `requirements-server.txt` 钉 `transformers<5` |
| `HTTPSConnectionPool(host='huggingface.co'...)` | 远端服务器连接 HuggingFace 官方源超时/不通 | `docker/run_server.sh` 已经配置了 `HF_ENDPOINT="https://hf-mirror.com"` 镜像源。如果依然报错，请确认远端宿主机是否能正常访问 `hf-mirror.com`，或检查容器是否未正确注入环境变量。 |

### 2. `/segment` 返回 score=0 mask 全黑
GroundingDINO 没检测到目标。调 prompt：英文短语 + 形容词，结尾加 "."（server 会自动补）。也试试加 bbox 直接绕过 GD：
```python
client.segment(rgb, bbox=[x1, y1, x2, y2])
```

### 3. `/grasp` 返回 0 个抓取
- mask 是不是全黑或非常小（< 100 px）
- 物体的 depth 是否全 0（透明 / 反光 / 太近太远）
- collision detection 把所有候选都过滤了 → 看 `docker logs` 有没有 "Number of grasps before / after collision detection"

</details>

---

## 🔗 参考链接

- 上游 ZeroGrasp 仓库：<https://github.com/sh8/ZeroGrasp>
- 客户端 `agentic_grasp/perception_client.py` 定义的契约就是这个 server 实现的契约
- 历史决策、踩坑过程：见对话记录 [Cursor agent 开发会话](见 .cursor/projects/home-ubuntu/agent-transcripts/)

# medicine_agentic

固定工位双臂机器人“打包入库岗”的技能库与确定性工作流。

当前版本包含 P0 的安全基础设施与 **dry-run** 任务一规划器。位姿/TCP
采集命令只读机械臂反馈，不会发送运动；真实运动适配器仍保持锁定，直到现场
标定与低速验收完成。

- 三个任务的调用顺序；
- 每个技能的入口、出口与验收条件；
- 一次有限重试与失败停机；
- 后续 AIRBOT、RealSense、吸盘和 ACT 的适配边界。

## 当前技能

| 技能 | 首版实现 |
|---|---|
| `pick_carton` | 固定视觉 + 吸盘轨迹 |
| `place_carton` | 固定槽位轨迹 |
| `stabilize_carton` | 右夹爪固定/撑开盒口 |
| `pick_item` | 药板/说明书揭取，ACT候选 |
| `insert_item` | 药板/说明书插入，ACT候选 |
| `erect_carton` | 先固定双臂轨迹，必要时ACT |
| `close_carton` | 固定折小舌/主盖 + ACT插主舌 |
| `verify` | 视觉与I/O规则 |
| `safe_move` | 固定安全位和工位切换 |

详细入口与出口见 [docs/SKILL_CONTRACTS.md](docs/SKILL_CONTRACTS.md)。

## 在 dosw1 上运行

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/run_dry.py --workflow all
PYTHONPATH=src python -m unittest discover -s tests -v
```

P0 现场操作按 [docs/P0_ONSITE.md](docs/P0_ONSITE.md) 执行。最常用的三个入口：

```bash
# 全部为只读检查
PYTHONPATH=src python scripts/p0_doctor.py --live
PYTHONPATH=src python scripts/posectl.py doctor
PYTHONPATH=src python scripts/tcp_pivot.py --arm left doctor
```

## 独立打包控制台

打包项目使用自己的本地控制台，不依赖数据采集服务或商超遥操作网站。默认
只绑定 `127.0.0.1:8899`。网站可以同步录制相机和机械臂反馈，并提供一个
操作员确认的 Follow 遥操启动入口；它不提供自主运动、直接关节命令、
轨迹回放、吸盘或底盘控制接口：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/packaging_console.py
```

当前独立相机标准为前置 D435I 的 RGB 与 Depth 均
`1280x720 @ 30 FPS`，深度对齐到彩色图。8899 直接独占 RealSense；启动前
必须确认 8888、9999 等其他相机使用方已经停止：

```bash
PYTHONPATH=src python scripts/packaging_console.py --camera-mode realsense
```

遥操入口只在主臂 `50050/50052` 和执行臂 `50051/50053` 已稳定就绪时
调用现有 Follow 脚本；不会从网页自动重置 CAN 或重启机械臂容器。

详细的端口隔离、SSH 隧道和相机独占规则见
[docs/PACKAGING_CONSOLE.md](docs/PACKAGING_CONSOLE.md)。

本项目按固定工位双臂系统设计；底盘、移动、定位与导航完全不在系统边界内，
独立控制台也没有这些接口。

## 药盒六面参考库

药盒位置不再绑定固定格子。检测器可以在画面中提出任意位置的候选框，但候选
只有在“参考面身份、该面吸取许可、吸盘边缘余量”全部通过后，才会标记为
`graspable_2d=true`。这仍只是二维许可，不会令 `target_ready` 变为 true；
真实抓取还必须继续通过同步深度、表面平面、工作空间和试抬验证。

参考库按 SKU 和版本冻结，包含药盒六个面的 1–3 张图片。创建工具：

```bash
PYTHONPATH=src python scripts/reference_faces.py --help
```

完整的采图、`init → add → finalize → validate` 命令见
[reference_faces/README.md](reference_faces/README.md)。当前默认
`classical` 检测器只生成几何候选，不验证药盒面，因此会安全地保持
`graspable_2d=false`；接入视觉提示插件后也必须经过同一中央判定。

## P0 工具

| 命令 | 用途 | 会否运动 |
|---|---|---|
| `scripts/p0_doctor.py` | 检查外参、TCP、姿态、相机/双臂在线状态 | 否 |
| `scripts/posectl.py teach` | 遥操到位后按回车，稳定采样双臂姿态 | 否 |
| `scripts/tcp_pivot.py` | 左吸盘/右夹爪 TCP 枢轴标定 | 否 |
| `scripts/vacuum_audio.py` | 录制与判别气泵声音 | 否 |
| `scripts/task1_detect_box.py` | 检测药盒、吸取点与三维位置 | 否 |
| `scripts/p0_task1_plan.py` | 生成任务一完整动作计划并检查所有门禁 | 否 |

吸附成功的硬判据仍是“20 mm 试抬后药盒在视觉中跟随 TCP”。气泵声音在
独立留出集通过误报/漏报验收前只作辅助证据，不能单独放行运输。

模拟一次“第二块药板首次插入失败、回撤后成功”：

```bash
PYTHONPATH=src python scripts/run_dry.py \
  --workflow load \
  --fail-once insert_item:BLISTER:2
```

## 三个工作流

### 任务一：成品药盒装入大箱

```text
pick_carton(CLOSED_CARTON)
→ verify(carton_held)
→ safe_move(pack_preinsert)
→ place_carton(slot_id, orientation)
→ verify(slot_occupied)
```

### 任务二：装药板、说明书并盖盒

```text
stabilize_carton
→ [pick_item(BLISTER) → verify → insert_item(stage) → verify] × 3
→ pick_item(LEAFLET) → insert_item(LEAFLET) → verify
→ close_carton(LOADED_TOP)
→ verify(carton_closed)
```

`pick_item` 和 `insert_item` 之间使用固定安全运输轨迹，不让模型学习无接触长距离移动。

### 任务三：扁平纸盒展开并闭合底部

```text
pick_carton(FLAT_BLANK)
→ erect_carton
→ verify(carton_squared)
→ close_carton(EMPTY_BOTTOM)
→ verify(carton_closed)
```

## 按部就班的开发顺序

1. **工作流骨架（当前）**  
   dry-run、技能契约、事件记录、失败重试和单元测试全部通过。

2. **只读硬件预检**  
   接入现有 `manipulation/agentic` 的相机、标定和 AIRBOT 健康检查；仍禁止运动。

3. **任务一确定性动作**  
   先完成单盒吸取、真空验收和单槽放置，再扩展槽位表。目标是单盒连续50次。

4. **公共视觉与验收**  
   建立药盒、药板、说明书、盒口和盖舌的检测接口；所有技能必须返回证据。

5. **药板/说明书揭取**  
   先用规则吸点和固定轨迹；规则不足时再训练 `pick_item` ACT。

6. **药板/说明书插入**  
   固定运输到统一 `pre_insert`，只训练盒口附近的 `insert_item` ACT。

7. **闭盒**  
   固定轨迹折两个小舌和主盖，仅训练主插舌 `tuck_tongue`。

8. **展开纸盒**  
   先做吸盘与夹爪反向拉开的固定流程；不稳定时才增加 `erect_carton` ACT。

9. **整任务联调**  
   每个工位独立达到目标成功率后，才由工作流串联；模型不负责计数和任务切换。

任何真实运动适配器都必须默认 `dry_run=True`，并显式通过运动授权后才能执行。

## 任务一：先做药盒识别（过渡兼容方案）

以下旧识别脚本是早期联调保留的过渡兼容方案：现有 Web 控制台持续占用
RealSense 时，它读取该服务的 MJPEG 与深度接口，不会抢占相机，也不会连接
机械臂。新开发应优先使用上面的独立打包控制台；`127.0.0.1:8899` 自己管理
离线画面或 RealSense，既不代理也不依赖 `8765`/`8888`：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/task1_detect_box.py
```

输出位于 `artifacts/task1/latest/`：

- `capture.jpg`：本次原图；
- `detection_overlay.jpg`：上料区 ROI、药盒轮廓、吸取点和判定原因；
- `detection.json`：像素位置、五点深度、相机坐标、左臂基座坐标和拦截条件。

当前 `configs/task1_box.json` 的 ROI 仅是安全搜索区域，不再表达药盒格子或
固定中心点。比赛摆场后只需让搜索区域覆盖允许出现药盒的桌面范围；候选框和
吸取点必须由每帧图像计算。检测脚本是只读的；配置中的
`pick.tcp_calibrated` 仍为 `false`，在吸盘 TCP 实测完成前不允许接真实运动。

## 左吸盘 TCP pivot 标定（只读）

把吸盘标定尖端始终抵在同一个固定点上，通过遥操改变左臂法兰姿态。每个
姿态停止 Follow 后运行一次采集；脚本只调用 AIRBOT 状态读取接口，不切换
控制模式，也不包含任何运动命令：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1

PYTHONPATH=src python scripts/tcp_pivot.py doctor

PYTHONPATH=src python scripts/tcp_pivot.py capture --label pose_01
PYTHONPATH=src python scripts/tcp_pivot.py capture --label pose_02
# 至少采集 8 个姿态，建议 12–16 个，绕不同旋转轴改变 30° 以上

PYTHONPATH=src python scripts/tcp_pivot.py list
PYTHONPATH=src python scripts/tcp_pivot.py validate
PYTHONPATH=src python scripts/tcp_pivot.py solve --dry-run
PYTHONPATH=src python scripts/tcp_pivot.py solve
```

默认采样文件：

```text
artifacts/calibration/left_suction_tcp/pivot_samples.json
```

只有样本数、旋转激励、条件数、残差和 TCP 长度全部通过验收，才会原子写入：

```text
configs/calibration/left_suction_tcp.json
```

Pivot 法只能确定法兰到吸盘 TCP 的**平移**，不能确定吸盘坐标系绕自身轴的
旋转。正式结果对此会明确记录 `rotation_calibrated: false`。

## ACT 数据采集与训练交接

8899 控制台已包含 ACT 示教类型：从臂反馈作为 observation、主臂反馈作为
action，成功 episode 使用 `READY + checksums.sha256` 原子封存。8899 被关闭
期间可离线运行全部单元测试，不需要连接相机或机械臂，也不会绑定该端口。

```bash
# 仅在现场允许启动 8899 时执行
./scripts/start_packaging_console_8899.sh

# 校验与推送到 zjlab（不保存密码）
./scripts/validate_act_data.sh
./scripts/push_act_to_zjlab.sh
```

完整数据约定、网络方向和首次硬件验收步骤见
[docs/ACT_DATA_PIPELINE.md](docs/ACT_DATA_PIPELINE.md)。

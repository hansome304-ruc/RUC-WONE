# manipulation/agentic

Agentic Grasp 模块：RealSense RGB-D + 远端感知服务 + AIRBOT 双臂抓取。

## 准备

```bash
cd /home/ubuntu/RUC-WONE/manipulation/agentic
conda activate dos-w1
cp .env.example .env
pip install -e .
```

确认 `.env`：

```text
PERCEPTION_URL=http://127.0.0.1:9100
LEFT_ARM_PORT=50051
RIGHT_ARM_PORT=50053
CALIB_DIR=/home/ubuntu/calib/result
```

## 测试

```bash
python scripts/00_test_health.py
python scripts/01_test_segment.py --prompt "a bottle" --save-rgb
python scripts/run_scripted.py --prompt "a bottle" --arm right --dry-run
```

`scripts` 根目录只放比赛主流程入口；调试/标定工具在 `scripts/debug/`，未完成实验入口在 `scripts/experimental/`。

## Debug 输出

默认调试输出目录是 `debug_out/`：

| 阶段 | 命令 | 输出 |
|---|---|---|
| 服务心跳 | `python scripts/00_test_health.py` | 只打印 `/healthz` JSON，不产图 |
| 分割测试 | `python scripts/01_test_segment.py --prompt "a bottle" --save-rgb` | `segment_overlay.png`，可选 `segment_overlay_rgb.png` |
| 标定投影 | `python scripts/02_test_calib_overlay.py` | `calib_overlay.png` |
| 抓取干跑 | `python scripts/run_scripted.py --prompt "a bottle" --arm right --dry-run` | `sam_input.png`，`grasp_segment_overlay.png`，`selected_grasp.png`，`selected_grasp.json` |
| 抓取候选可视化 | `python scripts/debug/04_visualize_grasps.py --prompt "a bottle"` | `grasp_visualize_2d.png` |
| 真实抓取视频 | `python scripts/run_scripted.py --prompt "a bottle" --arm right --record-video` | `grasp_run_YYYYmmdd_HHMMSS.mp4` |

说明：

- `selected_grasp.png` 里会画出 mask、bbox、执行抓取点 `EXEC`、预抓取点 `PRE` 和抬升点 `LIFT`。
- `selected_grasp.json` 记录抓取位姿、抓取宽度、分数、base 坐标和各方向轴，适合复盘坐标问题。
- 图片文件默认会被下一次运行覆盖；视频文件带时间戳。

## 真实抓取

```bash
python scripts/run_scripted.py --prompt "a bottle" --arm right --strategy default --speed SLOW --record-video
```

参数：

- `--prompt`: 目标文本
- `--arm`: `left` 或 `right`
- `--strategy`: `default`、`low-bottle`、`side-bottle`
- `--endpoint`: `/grasp` 或 `/grasp_dummy`
- `--dry-run`: 只规划不执行
- `--record-video`: 保存抓取过程视频到 `debug_out`
- `--lift-height`: 抓住后机械臂末端上抬距离，单位 `m`

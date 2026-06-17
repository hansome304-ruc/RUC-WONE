# agentic scripts

比赛主入口只保留稳定脚本：

```bash
python scripts/00_test_health.py
python scripts/01_test_segment.py --prompt "a bottle" --save-rgb
python scripts/02_test_calib_overlay.py
python scripts/03_dummy_grasp.py --prompt "a bottle" --arm right --dry-run
python scripts/run_scripted.py --prompt "a bottle" --arm right --strategy default --speed SLOW --record-video
bash scripts/start_arms_only.sh
```

说明：

- `00_test_health.py`: 检查感知服务 `/healthz`
- `01_test_segment.py`: RealSense 拍图并测试分割
- `02_test_calib_overlay.py`: 标定坐标系投影检查
- `03_dummy_grasp.py`: 抓取流水线通用入口，可选 `/grasp_dummy` 或 `/grasp`
- `run_scripted.py`: 默认使用 `/grasp` 的真实抓取入口
- `start_arms_only.sh`: 只启动比赛用的左右执行臂 server

常见 debug 输出都在 `debug_out/`：

- `segment_overlay.png`: 分割 mask 和 bbox。
- `segment_overlay_rgb.png`: `01_test_segment.py --save-rgb` 保存的原始 RGB。
- `calib_overlay.png`: 标定坐标系投影检查。
- `sam_input.png`: 抓取流水线送入 SAM/GroundingDINO 的原图。
- `grasp_segment_overlay.png`: 抓取流水线里的分割结果，即使后续没有抓取候选也会保存。
- `selected_grasp.png`: 最终选中的抓取、预抓取和抬升点。
- `selected_grasp.json`: 抓取位姿、base 坐标、方向轴和分数。
- `grasp_run_*.mp4`: 加 `--record-video` 后保存的抓取过程视频。

调试脚本在 `scripts/debug/`，未完成实验入口在 `scripts/experimental/`。

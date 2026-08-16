# P0任务一：纯规划状态机

该状态机只读取：

1. `posectl`维护的姿态文件；
2. `task1_detect_box.py`已经生成的`detection.json`；
3. `task1_box.json`中的相机外参与真实 TCP 标定文件。

它不会打开相机，不会导入机械臂驱动，也不会设置吸盘。成功结果
`DRY_RUN_READY`只表示门禁通过且规划步骤完整，不表示实体任务完成。

## 使用

先离线或通过现有只读识别脚本生成检测报告，再运行：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/p0_task1_plan.py \
  --config configs/p0_task1_plan.json
```

运行日志位于：

```text
artifacts/p0_task1_runs/<run_id>/
├── events.jsonl
└── summary.json
```

默认门禁会阻止当前未完成配置：

- `configs/calibration/left_suction_tcp.json`必须存在，并且
  `calibrated=true`、`usable_for_motion=true`、平移有限；若带内容哈希还会
  验证哈希；
- 相机到左臂基座外参必须存在且为有效刚体变换；
- P0要求的八个双臂姿态必须存在；
- 姿态必须为`validated`、`stable=true`和`collision_free=true`；
- 检测报告必须成功、未过期、具有有效三维位置且目标可达；
- `dry_run`必须为`true`；
- 运动与吸盘命令必须为`false`；
- `robot_adapter`只能为`none`或`disabled`。

20 mm 试抬的视觉跟随判据始终必选。声学未通过独立留出集前只记录，不会
替代视觉；启用为硬门禁后则要求声音和视觉同时通过。

该CLI没有`--execute`选项。以后接入真实执行器时，应建立独立模块和独立
人工授权流程，不应向本规划器增加绕过门禁的开关。

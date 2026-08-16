# P0 现场操作

所有命令均在：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
```

真实运动当前默认锁定。以下工具只读机械臂反馈、相机或麦克风，不会切换
控制模式，不会控制气泵，也不会移动机械臂。

## 1. 启动检查

先用现有 Web 控制台启动 follower AIRBOT 服务并确认 Follow 已停止，再运行：

```bash
PYTHONPATH=src python scripts/p0_doctor.py --live
```

输出会明确列出未完成项。相机外参文件通过只代表文件和刚体矩阵有效；实际
投影仍以现有 `episode_0004` 验证结果为准，不能用 TCP 偏移去修改相机外参。

## 2. 标定左吸盘 TCP

用吸盘中心的临时标定尖端（或可重复定位的杯口中心）始终顶住同一个固定点。
每次只改变手腕方向，覆盖至少 8 个差异明显的方向，每到一个方向运行一次：

```bash
PYTHONPATH=src python scripts/tcp_pivot.py --arm left capture
```

全部采完：

```bash
PYTHONPATH=src python scripts/tcp_pivot.py --arm left list
PYTHONPATH=src python scripts/tcp_pivot.py --arm left solve --dry-run
PYTHONPATH=src python scripts/tcp_pivot.py --arm left solve
```

只有样本数、姿态激励、条件数、RMS 与最大残差全部达标，结果才会写入
`configs/calibration/left_suction_tcp.json`。建议 RMS 不超过 2 mm。

右夹爪闭合中心用同样方式标定：

```bash
PYTHONPATH=src python scripts/tcp_pivot.py --arm right capture
PYTHONPATH=src python scripts/tcp_pivot.py --arm right solve --dry-run
PYTHONPATH=src python scripts/tcp_pivot.py --arm right solve
```

右夹爪可先把现有约 130 mm 的法兰到指尖中心距离作为机械初值，但未通过
上述实测前不会被标记为可运动。

## 3. 一次录完固定姿态

建议先完成两个 TCP，再运行：

```bash
PYTHONPATH=src python scripts/posectl.py doctor
PYTHONPATH=src python scripts/posectl.py teach --manual-validate
```

每一步的流程相同：遥操到提示姿态，停止 Follow，确认双臂静止，按回车。
脚本会采 20 组双臂反馈，拒绝仍在运动、接近软限位或左右采样时间差过大的
姿态。`--manual-validate` 会要求现场人员在当前姿态输入
`VALIDATE <姿态名>`，记录工具/桌面/箱体的人工间隙检查；它不冒充自动
全连杆碰撞检测。

需要记录的姿态：

1. `home`
2. `task1_observe`
3. `safe_transport_empty`
4. `safe_transport_carton`
5. `pre_pick_carton`
6. `pre_place_carton`
7. `place_slot_0_contact`
8. `post_place`
9. `recovery_high`
10. `pre_pick_blister`
11. `pre_insert`

中断后重新运行 `teach` 会自动跳过已存在的姿态。单独补录：

```bash
PYTHONPATH=src python scripts/posectl.py capture pre_insert
PYTHONPATH=src python scripts/posectl.py approve pre_insert
```

## 4. 气泵声音（可选辅助信号）

dosw1 已发现 UGREEN Camera 2K 的 USB 麦克风，默认设备为
`plughw:U2K,0`。工具只录音，不控制气泵。

先各采至少 10 条训练样本：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py record --label empty --count 10
PYTHONPATH=src python scripts/vacuum_audio.py record --label sealed --count 10
PYTHONPATH=src python scripts/vacuum_audio.py calibrate
```

再把另一批各 10 条录到独立目录并验收：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py record \
  --label empty --count 10 --out-dir artifacts/vacuum_audio/holdout
PYTHONPATH=src python scripts/vacuum_audio.py record \
  --label sealed --count 10 --out-dir artifacts/vacuum_audio/holdout
PYTHONPATH=src python scripts/vacuum_audio.py evaluate \
  --samples-dir artifacts/vacuum_audio/holdout
```

默认要求空吸误报为 0。即使声学模型通过，任务一仍必须执行 20 mm 试抬并
用视觉确认药盒随吸盘上升。

## 5. 药盒检测与动作计划

确保 Web 控制台相机在线，在真实上料区放一个药盒：

```bash
PYTHONPATH=src python scripts/task1_detect_box.py
PYTHONPATH=src python scripts/p0_task1_plan.py
```

第一条生成原图、叠加图和三维检测 JSON；第二条检查外参、真实 TCP 文件、
固定姿态、检测新鲜度与工作区，并生成以下 dry-run 状态机：

```text
观察 → 动态预抓取 → 垂直接触 → 开吸盘
→ 试抬20 mm → 视觉/声音融合确认 → 完整抬升
→ 安全运输 → 箱内固定点释放 → 垂直撤离 → 视觉确认
```

`ready=true` 仍只表示“动作计划门禁通过”，不会发送任何运动。

## 6. 药板抓取

P0 先记录右夹爪 TCP 与 `pre_pick_blister`。药板贴桌且很薄，因此默认路线是
固定移动到 `pre_pick_blister`，只让 ACT 学最后约 50–80 mm 的斜向揭取、
闭爪和 20 mm 试抬；长距离运输仍使用固定姿态。

采集示范时，每段从同一 `pre_pick_blister` 开始，成功标准是单片药板被夹住
并完成 20 mm 试抬。先采 50–80 条成功示范，覆盖位置、角度、外露边缘和
轻微堆叠变化。训练/部署 ACT 前，任务一的吸盘闭环可以独立验收。

# debug scripts

这里放标定、坐标系、可视化和硬件诊断脚本。它们不是比赛主流程入口。

常用脚本：

```bash
python scripts/debug/04_visualize_grasps.py --prompt "a bottle"
python scripts/debug/06_test_physical_base_motion.py --arm right --axis z --distance 0.03
python scripts/debug/10_get_current_poses.py
python scripts/debug/12_list_cameras.py
```

`04_visualize_grasps.py` 会保存 `debug_out/grasp_visualize_2d.png`，用于检查 ZeroGrasp 返回的多个候选抓取是否投影到物体上。

使用原则：

- 只在调试标定、相机、坐标轴、抓取候选时运行。
- 会真实移动机械臂的脚本先用很小的 `--distance`，并保持急停可触达。
- 比赛 README 不依赖这里的脚本。

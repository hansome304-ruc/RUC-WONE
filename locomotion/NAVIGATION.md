# 底盘建图与定点移动探索

当前 RUC-WONE 已经实现的是底盘低层速度控制：

- `/cmd_vel`: 发速度
- `/odom`: 读里程计
- `doctor`: 读厂商状态 topic，例如 `/map_info_s`、`/localization_state`

是否能建图、定位、定点导航，取决于底盘 ROS1 master 是否暴露雷达、地图和导航接口。先做只读探测。

## 1. 连接检查

接上底盘网线后，确认本机有 `192.168.31.x` 地址：

```bash
ip addr show enp2s0
ping -c 1 -W 1 192.168.31.7
```

如果没有自动拿到地址，可以临时手动设置一个同网段地址：

```bash
sudo ip addr add 192.168.31.243/24 dev enp2s0
sudo ip link set enp2s0 up
```

## 2. 探测底盘 ROS topic

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/movebase.sh doctor
bash locomotion/scripts/movebase.sh state
bash locomotion/scripts/base_probe.sh
```

保存完整 JSON 方便后续分析：

```bash
bash locomotion/scripts/base_probe.sh --json > /tmp/base_probe.json
```

重点看：

- `laser_candidates`: 是否有 `/scan`、`/laser_scan`、`/lidar` 等雷达 topic
- `map_candidates`: 是否有 `/map` 或厂商地图 topic
- `localization_candidates`: 是否有 AMCL/localization/pose topic
- `navigation_candidates`: 是否有 move_base/nav/goal/path/costmap topic
- `interesting_services`: 是否有 build_map/load_map/save_map/navigation goal 相关服务

## 3. 三种路线

### A. 厂商自带导航接口

如果 `base_probe` 发现了导航 goal topic/service，这是最推荐的路线：

1. 用厂商接口建图或加载地图。
2. 用 localization 状态确认定位成功。
3. 封装一个 `goto` 命令发布目标点。

优点是避障、局部规划和恢复行为由底盘原生系统处理，比赛最稳。

### B. 自己跑 SLAM/Nav2

如果只有 `/scan`、`/odom`、`/tf`，没有厂商导航接口，就需要自己部署：

1. 桥接 ROS1 底盘数据到 ROS2，或在 ROS1 环境运行建图。
2. 使用 SLAM 建 `/map`。
3. 保存地图。
4. 使用定位和 Nav2/move_base 做定点移动。

这台机器目前只有 ROS2 `rviz2`，没有 `slam_toolbox`、`nav2`、`ros1_bridge`，所以这条路线需要额外安装环境。

### C. 里程计点到点移动

如果只是短距离实验，可以不用地图，直接基于 `/odom` 做点到点闭环：

1. 记录当前 `/odom` 作为原点。
2. 给定相对目标，例如前方 `0.5m`、旋转 `90deg`。
3. 用 `/cmd_vel` 做 P 控制靠近目标。

这不等于导航：没有全局地图，不保证避障，里程计会漂移。只适合小范围调试。

## 4. 建议顺序

先不要急着装 SLAM。按这个顺序判断：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/base_probe.sh --json > /tmp/base_probe.json
```

把 `/tmp/base_probe.json` 里的 topic/service 看清楚后，再决定：

- 有厂商 goal/nav service：封装厂商定点导航。
- 有 `/scan` 但没有导航：补 SLAM/Nav2 或 ROS1 move_base 环境。
- 只有 `/cmd_vel`/`/odom`：先做短距离 odom waypoint，不做真正避障导航。

## 5. 当前底盘已经发现的接口

2026-06-17 已探测到：

- 雷达：`/scan`、`/scan_filtered`
- 地图：`/map`
- 定位：`/localization_state`、`/pose_info`
- 导航入口：`/move_base_simple/goal`
- 规划/避障：`/move_base/global_costmap/*`、`/move_base/local_costmap/*`

这说明底盘已经有 ROS1 `move_base` 导航栈。优先使用厂商已有导航，不需要自己从零写避障。

当前风险：

```text
current_map_name = 0203.yaml.txt
confidence = 0
initialed = -1
```

这表示底盘加载了旧地图，但当前没有完成定位。定位成功前不要发导航目标。

## 6. 定点移动步骤

### 6.1 安全准备

1. 清空底盘前方和两侧障碍。
2. 急停保持可触达。
3. 先确认低层运动正常：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/movebase.sh doctor
bash locomotion/scripts/movebase.sh state
```

### 6.2 确认定位状态

```bash
bash locomotion/scripts/movebase.sh doctor
```

如果看到：

```text
confidence = 0
initialed = -1
```

先做初始化或重新建图，不要发目标点。

### 6.3 设置初始位姿

如果你知道机器人在当前地图 `/map` 中的位置，先发初始位姿：

```bash
bash locomotion/scripts/nav.sh set-initial --x 1.0 --y 2.0 --yaw 0.0
```

角度也可以用度数：

```bash
bash locomotion/scripts/nav.sh set-initial --x 1.0 --y 2.0 --yaw 90 --deg
```

然后再次检查：

```bash
bash locomotion/scripts/movebase.sh doctor
```

需要看到定位状态变好，例如 `initialed` 不再是 `-1`，`confidence` 不再是 `0`。

### 6.4 发定点导航目标

确认定位成功后，才发送目标点：

```bash
bash locomotion/scripts/nav.sh goto --x 2.0 --y 1.0 --yaw 0.0 --yes
```

注意：`goto` 会触发底盘自主运动，所以必须显式加 `--yes`。

### 6.5 停止

如果需要停止底盘：

```bash
bash locomotion/scripts/movebase.sh stop
```

## 7. 重新建图

如果当前场景和 `0203.yaml.txt` 不一致，需要先建当前场景地图。现在 ROS topic 里能看到 `/mapping_state`、`/map_change_call`、`/save_3d_map` 等厂商接口线索，但还没有确认“开始建图/保存地图”的完整命令格式。

建议顺序：

1. 先问厂商确认建图接口或 Web 控制台入口。
2. 如果厂商提供 topic/service 格式，把它封装到 `locomotion/scripts/nav.sh`。
3. 如果厂商没有接口文档，再考虑自己部署 ROS1/ROS2 SLAM。

## 8. 没有厂商建图入口时怎么办

有三条替代路线，按稳定性排序：

### 路线 A：短期使用里程计相对移动

这条路不需要地图。适合短距离、小范围、有人看护的比赛动作编排。

前进 `0.5m`：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/odom_nav.sh relative --dx 0.5 --dy 0.0 --dyaw 0.0 --yes
```

左转 `90deg`：

```bash
bash locomotion/scripts/odom_nav.sh relative --dx 0.0 --dy 0.0 --dyaw 90 --deg --yes
```

先前进 `0.4m`，再最终朝向 `30deg`：

```bash
bash locomotion/scripts/odom_nav.sh relative --dx 0.4 --dy 0.0 --dyaw 30 --deg --yes
```

限制：

- 没有地图。
- 没有真正避障。
- 里程计会漂移。
- 适合几十厘米到一两米的短动作，不适合复杂场景自主导航。

### 路线 B：外部 SLAM 建图

底盘已经暴露 `/scan`、`/odom`、`/tf`，理论上可以在外部机器跑 SLAM：

1. 从底盘 ROS1 master 读取 `/scan`、`/odom`、`/tf`。
2. 用 ROS1 `gmapping` / `hector_slam` / `cartographer`，或 ROS2 `slam_toolbox` 建图。
3. 保存 `.pgm + .yaml` 地图。
4. 再用定位和 `move_base`/Nav2 发目标点。

当前这台机器是 Ubuntu 22.04 + ROS2 Humble，只有 `rviz2`，没有 `slam_toolbox`、`nav2`、`ros1_bridge`，所以这条路线需要额外安装或用 Docker 环境。

### 路线 C：使用厂商已有旧地图，但手动初始化

如果当前场地和 `0203.yaml.txt` 很接近，可以手动发 `/initialpose` 让定位收敛，然后试小目标：

```bash
bash locomotion/scripts/nav.sh set-initial --x 1.0 --y 2.0 --yaw 0.0
bash locomotion/scripts/movebase.sh doctor
```

只有定位状态变好后，才发：

```bash
bash locomotion/scripts/nav.sh goto --x 1.5 --y 2.0 --yaw 0.0 --yes
```

如果地图不匹配，这条路不要用。

## 9. SDK 探索结论

目前本机已检查：

- `airbot_py-5.1.6-py3-none-any.whl`
- `/home/ubuntu/out_dexmal/mobile_base`
- `/home/ubuntu/dos_w1/base_cmd.py`
- RUC-WONE 内 `airbot_base` 兼容层

结论：

- `airbot_py` 主要是机械臂 SDK，没有发现底盘建图/导航 API。
- `dos_w1/base_cmd.py` 只是 ROS2 `/cmd_vel` 速度发布器。
- 旧的 `mobile_base/airbot_base.py` 现在只是兼容导入。
- `AIRBOTBase.build_map/load_map/move_to_pose` 等名字存在，但只是 `NotImplementedError` 占位。

所以当前更可靠的方向是直接利用底盘 ROS1 master 暴露出来的 topic/param/service。

## 10. 避障距离相关参数

底盘当前 `move_base` 参数里能读到这些关键项：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `/move_base/local_costmap/footprint` | `[[0.35,-0.305],[0.35,0.305],[-0.35,0.305],[-0.35,-0.305]]` | 机器人外形，约 `0.70m x 0.61m` |
| `/move_base/local_costmap/footprint_padding` | `0.01` | 外形额外膨胀 |
| `/move_base/local_costmap/inflation_layer/inflation_radius` | `0.8` | 局部代价地图障碍膨胀半径 |
| `/move_base/global_costmap/inflation_layer/inflation_radius` | `0.8` | 全局代价地图障碍膨胀半径 |
| `/move_base/local_costmap/laser_layer/laser_scan_sensor/obstacle_range` | `2.9` | 雷达障碍物标记距离 |
| `/move_base/local_costmap/laser_layer/laser_scan_sensor/raytrace_range` | `3.0` | 雷达清除距离 |
| `/move_base/TrajectoryPlannerROS/stop_path_obs_dis` | `1.5` | 路径障碍停止距离，厂商扩展参数名 |
| `/move_base/tracking_body_padding` | `0.015` | 跟踪/车体 padding |

可以认为底盘“避障距离”主要由 `inflation_radius`、`footprint_padding`、`stop_path_obs_dis` 这几类参数影响。

注意：

- 当前服务列表没有发现常见的 dynamic reconfigure `set_parameters` 接口。
- 只用 `rosparam set`/XMLRPC `setParam` 改参数，运行中的 `move_base` 不一定立即生效。
- 真要改，建议先做只读记录，再小幅调整，并准备重启底盘导航节点或整机。比赛现场不要大改。

## 11. 外部 SLAM 环境建议

底盘是 ROS1 Noetic 生态，已经暴露：

```text
/scan
/scan_filtered
/odom
/tf
```

理论上可以外部建图。推荐两种做法：

### 方案 A：ROS1 Noetic Docker

优点是和底盘 ROS1 生态最接近。建议镜像内安装：

```text
ros-noetic-slam-gmapping
ros-noetic-map-server
ros-noetic-rviz
```

运行时用 host 网络，并设置：

```bash
export ROS_MASTER_URI=http://192.168.31.7:11311
export ROS_IP=192.168.31.243
```

注意不要直接在底盘 ROS master 上发布同名 `/map` 或 `map->odom` TF，以免干扰底盘自己的 `localization2d/move_base`。更稳的方式是独立 ROS master + topic relay，或者 remap 外部 SLAM 输出 topic。

### 方案 B：ROS2 Humble + slam_toolbox

这台机器当前只有 ROS2 `rviz2`，还没有：

```text
slam_toolbox
nav2
ros1_bridge
```

如果走 ROS2，需要安装这些包，并把 ROS1 `/scan`、`/odom`、`/tf` 桥接到 ROS2。工程量比 ROS1 Docker 更大，但后续 Nav2 生态更现代。

### 推荐

起步阶段先不要碰底盘原生导航节点。优先：

1. 用 `base_probe.sh` 固定记录 topic/param。
2. 用 `odom_nav.sh` 做短距离动作。
3. 另开 Docker/独立 ROS master 做 SLAM 实验。
4. SLAM 稳定后，再考虑把地图和导航接回 RUC-WONE。

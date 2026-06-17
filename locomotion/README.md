# locomotion

底盘控制模块。默认连接底盘 ROS1 master：

```text
http://192.168.31.7:11311/
```

## 状态与停止

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/base_probe.sh
bash locomotion/scripts/movebase.sh doctor
bash locomotion/scripts/movebase.sh state
bash locomotion/scripts/movebase.sh stop
```

## 移动

```bash
bash locomotion/scripts/movebase.sh forward --speed 0.05 --duration 2.0 --yes
bash locomotion/scripts/movebase.sh backward --speed 0.05 --duration 1.0 --yes
bash locomotion/scripts/movebase.sh turn-left --yaw-speed 0.15 --duration 1.0 --yes
bash locomotion/scripts/movebase.sh turn-right --yaw-speed 0.15 --duration 1.0 --yes
```

`位移 ≈ speed × duration`。例如 `0.05m/s × 2s ≈ 0.1m`。

`left/right` 和 `strafe-left/strafe-right` 是横移命令，发送 `linear.y`。如果底盘不是全向底盘，它会忽略这个方向；右转/左转请用 `turn-right` / `turn-left`。

## 原始速度

```bash
bash locomotion/scripts/movebase.sh raw --x 0.03 --y 0.0 --yaw 0.0 --duration 1.0 --yes
```

参数：

- `--x`: 前后速度，`m/s`
- `--y`: 左右速度，`m/s`
- `--yaw`: 旋转速度，`rad/s`
- `--max-linear`: 线速度保护上限
- `--max-angular`: 角速度保护上限

## 升降

升降通过独立 USB 串口控制，不走底盘 ROS master。默认使用 Prolific 控制盒：

```text
/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_EKDMo151406-if00-port0
```

读取状态：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/lift.sh status --timeout 1
```

运动控制：

```bash
bash locomotion/scripts/lift.sh stop
bash locomotion/scripts/lift.sh up --yes
bash locomotion/scripts/lift.sh down --yes
bash locomotion/scripts/lift.sh goto --position 200 --yes
```

更多说明见：

```text
/home/ubuntu/RUC-WONE/locomotion/lift/
```

## 建图与定点移动

底盘建图/导航需要先确认雷达、地图和导航 topic 是否存在：

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/base_probe.sh --json > /tmp/base_probe.json
```

详细流程见：

```text
/home/ubuntu/RUC-WONE/locomotion/NAVIGATION.md
```

如果已经完成定位，可发送 ROS1 `move_base` 目标点：

```bash
bash locomotion/scripts/nav.sh set-initial --x 1.0 --y 2.0 --yaw 0.0
bash locomotion/scripts/nav.sh goto --x 2.0 --y 1.0 --yaw 0.0 --yes
```

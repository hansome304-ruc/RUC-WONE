# lift

升降/升降柱控制模块，封装 `/home/ubuntu/hq_v2/SJJ_control/sjj_cli.py` 的 TYC 串口协议。

## 硬件接口

默认串口：

```text
/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_EKDMo151406-if00-port0
```

当前映射通常是：

```text
/dev/ttyUSB1
```

协议：

- 反馈包：`8E 8E <百位> <十/个位> FF`
- 控制包：`CE CE <命令> <百位> <十/个位> FF`
- 命令 `0`: 停止
- 命令 `1`: 上升到上极限
- 命令 `2`: 下降到下极限
- 命令 `3`: 指定位置

## 常用命令

```bash
cd /home/ubuntu/RUC-WONE
bash locomotion/scripts/lift.sh list
bash locomotion/scripts/lift.sh status --timeout 1
bash locomotion/scripts/lift.sh stop
```

真实运动需要 `--yes`：

```bash
bash locomotion/scripts/lift.sh up --yes
bash locomotion/scripts/lift.sh down --yes
bash locomotion/scripts/lift.sh goto --position 200 --yes
```

参数：

- `--position`: 指定位置，单位 `mm`，默认安全范围 `0..300`
- `--timeout`: 读取状态/原始数据的等待时间，单位 `s`
- `--check-timeout`: 运动前读取反馈的等待时间，默认 `1.0s`
- `--port`: 手动指定串口
- `--skip-feedback-check`: 跳过运动前反馈检查，不建议比赛时使用
- `--yes`: 非停止运动命令必须显式添加

## Python 调用

```python
from locomotion.lift import Lift

lift = Lift()
packets = lift.read_status(timeout_s=1.0)
print(packets[-1].position_mm)
lift.goto(200)
```

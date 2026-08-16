# 独立打包控制台

`packaging_console` 是药品打包任务的独立控制台。它提供相机画面、
相机档案、系统状态、药盒二维检测，以及相机与机械臂反馈的同步轨迹录制。
录制器只调用机械臂状态读取接口，不切换控制模式、不控制吸盘，也不执行任务。
网页另有操作员确认的 Follow 启停入口；该入口只复用已有主从遥操脚本，
不接受浏览器传入主机、端口或命令，也不会启动 8888/9999。
机器人是固定工位双臂系统；底盘、移动、定位、建图和导航完全不在本项目的
系统边界内，也不存在对应的网页或 API 接口。

## 网络与端口隔离

控制台默认且强制只监听本机回环地址：

```text
http://127.0.0.1:8899
```

端口约定：

| 端口 | 用途 |
|---|---|
| `8765` | 现有服务保留，打包控制台不得占用 |
| `8766` | 现有服务保留，打包控制台不得占用 |
| `8888` | 现有 Web 控制台保留 |
| `8899` | 独立打包控制台专属 |
| `9999` | 现有底盘服务保留，打包控制台不得占用 |

服务会拒绝 `0.0.0.0`、局域网地址和主机名等非 loopback 绑定。不要通过修改
配置把 8899 暴露到局域网或公网。

从开发机访问 dosw1 时使用 SSH 隧道：

```bash
ssh -N -L 8899:127.0.0.1:8899 dosw1
```

然后在开发机浏览器打开：

```text
http://127.0.0.1:8899
```

## 离线模式

离线模式使用固定图片，不访问 RealSense，适合开发页面、检测器和接口测试：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/packaging_console.py --camera-mode offline
```

默认离线图片由 `configs/packaging_console.json` 中的
`camera.offline_image` 指定。离线帧会统一缩放为 `1280×720`。

## RealSense 实时模式

实时模式直接打开配置中指定序列号的 RealSense：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/packaging_console.py --camera-mode realsense
```

默认相机档案：

```text
color: 1280×720 @ 30 FPS, bgr8
depth: 1280×720 @ 30 FPS, z16
depth aligned to color
```

配置文件不保存由其他分辨率按比例缩放得到的猜测内参。实时模式启动后从
RealSense SDK 当前活动的 `1280×720` 档位读取 Color/Depth 内参、
Depth→Color 外参、对齐后 Depth 内参和 depth scale；在独立验证完成前，
页面继续显示“配置未批准”。

`rs.align(color)` 后的 Depth 已经处于 Color 像素网格和光学坐标系。反投影
对齐后的深度时必须使用 `aligned_depth_intrinsics`，不能再次应用
`depth_to_color_extrinsics`。

该档案始终返回 `profile_approved: false`。即使画面和深度可用，它也不能作为
机械臂运动授权。

## RealSense 独占约束

RealSense 设备通常只能被一个进程稳定占用。启动实时模式前必须停止所有会
直接打开同一台相机的进程，例如：

- 现有 Web 控制台中的直接 RealSense 采集；
- 数据录制客户端；
- 标定、预览或调试脚本；
- 另一个 `packaging_console --camera-mode realsense` 实例。

不得让独立打包控制台和其他进程同时占用同一台 RealSense。发生冲突时，服务
会继续启动，但相机会显示 `error`，取帧接口返回 `503`。先停止冲突进程并确认
设备已释放，再重启本控制台；不要通过反复重连掩盖设备占用问题。

## 只读接口

允许的接口只有：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 服务状态与只读安全声明 |
| `GET` | `/api/status` | 相机、任务预览和安全门状态 |
| `GET` | `/api/camera/profile` | 相机档案 |
| `GET` | `/api/camera/frame.jpg` | 当前 JPEG 帧 |
| `GET` | `/api/camera/frame.jpg?overlay=<detection_id>` | 与该次检测严格绑定的叠加图 |
| `GET` | `/api/recordings/status` | 当前录制状态与计数 |
| `GET` | `/api/recordings` | 最近保存的录制列表 |
| `POST` | `/api/detect` | 对当前帧执行二维药盒检测 |
| `POST` | `/api/recordings/start` | 开始只读轨迹录制 |
| `POST` | `/api/recordings/stop` | 停止并原子保存当前录制 |

`/api/detect` 只产生候选框、二维吸取点和叠加图。它不产生深度目标、基座坐标、
运动命令或吸盘命令。

每次检测返回唯一 `detection.id` 和相应 `overlay_url`。叠加图按检测 ID 缓存，
不会把另一次检测的坐标画到当前画面；仅保留最近 8 次，过期 ID 返回 `404`。

录制类型 `calibration_left`、`calibration_right` 和
`projection_validation` 会同时保存左右臂反馈，以兼容现有标定转换脚本；
普通轨迹可以选择仅左臂、仅右臂或双臂。每次成功录制包含：

```text
meta.json
actions/left_arm.jsonl       # 选中左臂时
actions/right_arm.jsonl      # 选中右臂时
sensors/cam_front_rgb.mp4
sensors/cam_front_rgb.mp4.tsf
sensors/cam_front_frames.jsonl
```

停止时先在隐藏的 `.inprogress` 目录完成视频和时间戳封装，再原子改名为最终
episode 目录。最终目录名包含经过路径安全处理的记录名称，并保留时间戳与
随机后缀避免重名覆盖。首版不提供轨迹回放、删除或机械臂写入接口。录制期间
网页只显示录制线程已经写入视频的最新帧（约 5 FPS），不会额外向 RealSense
取帧；二维检测按钮保持禁用。

不存在以下类别的接口：

- 机械臂移动、回零或姿态执行；
- 吸盘、真空泵或电磁阀开关；
- 工作流启动、暂停或恢复；
- 轨迹回放或执行；
- 底盘移动、速度控制、定位、建图或导航；
- 配置写入和标定部署。

所有未知 `/api/*` 路径返回 `404`，`PUT`和`DELETE`返回`405`。静态文件服务
会将路径限制在配置的 `static_dir` 内，编码后的 `..` 路径同样返回`403`。

## 常用检查

服务启动后可以执行：

```bash
curl -s http://127.0.0.1:8899/api/health
curl -s http://127.0.0.1:8899/api/status
curl -s http://127.0.0.1:8899/api/camera/profile
curl -s http://127.0.0.1:8899/api/recordings/status
curl -s http://127.0.0.1:8899/api/teleop/status
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -d '{}' \
  http://127.0.0.1:8899/api/detect
```

也可以直接在网页选择录制类型，填写标签，点击“开始录制”；遥操完成后点击
“停止并保存”。保存目录由 `trajectory_recorder.output_dir` 配置，默认是
`recordings/trajectories/`。新 episode 的目录格式是
`trajectory_时间_记录名称_随机后缀/`，保存完成后可在当前记录或最近记录中
一键复制完整路径。

遥操按钮读取原控制台保存在
`web_console/runtime/arm_services.json` 的主臂配置，后台依次调用已有的
`start_teleop_follow.sh`、`check_teleop_follow.sh` 与
`stop_teleop_follow.sh`。它不会自动重置 CAN，也不会启动或重建机械臂
容器；主臂 `50050/50052` 和执行臂 `50051/50053` 任一端点未稳定就绪时
会直接拒绝启动。启动前应先让主从臂姿态接近、清空工作区并准备物理急停。
建议先启动遥操，再开始录制。

健康接口必须包含：

```json
{
  "safety": {
    "mode": "operator_teleop",
    "dry_run": false,
    "motion_api": true,
    "teleop_enable_api": true,
    "autonomous_motion_api": false,
    "direct_joint_command_api": false,
    "suction_api": false,
    "chassis_api": false,
    "navigation_api": false
  }
}
```

`motion_api=true` 明确表示 Follow 启动后执行臂可能运动；
`autonomous_motion_api=false` 和 `direct_joint_command_api=false` 表示没有
自主运动或浏览器直接关节命令；`teleop_enable_api=true` 只允许操作员确认
后启停既有 Follow 生命周期。
如果服务出现浏览器可指定的任意命令、吸盘、底盘或导航写接口，应视为安全
回归并停止使用。

## 运行测试

在项目根目录用 `unittest` 的发现模式运行全部测试：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python -m unittest discover -s tests -v
```

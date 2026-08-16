# RUC-WONE 独立图传系统

本目录维护机器人三路 D435i 相机的唯一采集、RGB-D 共享、H.264 编码和 WebRTC 分发。
图传与底盘、机械臂、升降杆、吸盘及急停控制相互独立；视频故障不得阻塞控制链路。

## 数据流

```text
front/left/right camera workers
  ├─ latest aligned RGB-D → /dev/shm/ruc-video/<camera>/
  │                         ├─ 8877 depth/status API
  │                         └─ 8899 Agentic shared-memory reader
  └─ latest color frame → FFmpeg VAAPI H.264
                          → RTSP 127.0.0.1:8554
                          → MediaMTX
                          → WebRTC 8889/TCP + 8189/UDP
                          → 9999 Teleop
```

8888 仍负责机器人控制和安全 API，只把兼容相机请求代理到 8877，不直接打开相机。

## 相机配置

配置文件：`/home/ubuntu/RUC-WONE/video_gateway/config.json`

| 名称 | 序列号 | 彩色采集 | 彩色发布 | 深度 |
|---|---|---|---|---|
| front | D435i `420222072569` | 1280×720@30 | 1280×720@15 | 1280×720@30 |
| left | D435i `347622072392` | 1280×720@30 | 1280×720@15 | 480×270@30 |
| right | D435i `347622071407` | 1280×720@30 | 1280×720@15 | 480×270@30 |

三路彩色画面保持 1280×720，禁止为了带宽调整服务分辨率。三台 D435i 共享同一条 5Gbps Hub 上行，需要持续留意 USB `-71` 错误。

## 端口

| 端口 | 范围 | 用途 |
|---|---|---|
| 9999/TCP | 操作端 | Teleop 页面和控制代理 |
| 8889/TCP | 已授权操作端 | WebRTC/WHEP 信令 |
| 8189/UDP | 已授权操作端 | WebRTC 媒体数据 |
| 8877/TCP | 仅 127.0.0.1 | 深度、状态及兼容帧 API |
| 8554/TCP | 仅 127.0.0.1 | Camera worker 到 MediaMTX 的 RTSP 发布 |
| 9997/TCP | 仅 127.0.0.1 | MediaMTX 管理 API |
| 9998/TCP | 仅 127.0.0.1 | MediaMTX 指标 |

视频逻辑路径只有 `/front`、`/left`、`/right`。

## systemd 用户服务

```bash
systemctl --user status ruc-mediamtx.service
systemctl --user status ruc-video-api.service
systemctl --user status 'ruc-video-camera@*.service'
systemctl --user status ruc-video-sync.service
systemctl --user status ruc-agentic-console.service
```

整套图传重启：

```bash
systemctl --user restart ruc-mediamtx.service ruc-video-api.service
systemctl --user restart ruc-video-camera@front.service
systemctl --user restart ruc-video-camera@left.service
systemctl --user restart ruc-video-camera@right.service
systemctl --user restart ruc-video-sync.service
```

`ruc-video.target` 在开机时只拉起 front、left、right 三个相机实例。

## 健康检查

```bash
curl -s http://127.0.0.1:8877/healthz | python -m json.tool
curl -s http://127.0.0.1:9997/v3/paths/list | python -m json.tool
curl -s http://127.0.0.1:9998/metrics
find /dev/shm/ruc-video -maxdepth 2 -type f -ls
```

## 码率策略

当前使用固定码率，`encoder.adaptive_bitrate=false`，目标总码率约 8.0 Mbps：Front 2.8 Mbps、Left 2.6 Mbps、Right 2.6 Mbps。

9999 切换主画面时仍调用 `POST /api/video/main` 记录当前主相机，但不会改变码率、分辨率、帧率或重启 FFmpeg。每路编码队列最多保留一帧；链路来不及发送时丢弃过期画面，不累计延迟。

## 8899 注意事项

8899 必须使用配置中的 `mode: shared_memory`。启动时禁止传入 `--camera-mode realsense`，否则会绕过共享内存并重新独占 Front。正确启动命令由 `ruc-agentic-console.service` 维护。

## 三台 D435i 硬件复位

复位程序只操作以下三个固定 USB 设备和对应采集服务：

| 相机 | USB 物理端口 | VID:PID |
|---|---|---|
| Front D435i | `2-2.2` | `8086:0b3a` |
| Right D435i | `2-2.3` | `8086:0b3a` |
| Left D435i | `2-2.4` | `8086:0b3a` |

```bash
sudo systemctl --no-block start ruc-video-reset-all.service
systemctl status ruc-video-reset-all.service
cat /run/ruc-video-reset-all-status.json
curl -s http://127.0.0.1:8877/healthz | python -m json.tool
```

后端有互斥锁和 30 秒冷却时间，避免连续复位 USB 总线。权限只允许 `ubuntu` 免密码启动固定 systemd 单元，网页进程不能传入设备名或任意 root 命令。

## 已知硬件风险

- 三台 D435i 位于同一条 Hub 链路，曾在相同时间产生 USB `-71` 错误。
- `Device or resource busy` 通常是相机重复打开，不代表 USB 带宽不足。
- `-71` 通常指向 USB 物理链路、供电、Hub 级联或等时传输问题，软件只能隔离和恢复，不能修复物理层。
- 防火墙只允许既有操作端访问 8889/TCP 和 8189/UDP，不要把 WebRTC 媒体端口开放到未授权网段。

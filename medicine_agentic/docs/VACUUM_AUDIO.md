# 吸盘声学吸附验证（只读、可选）

气泵空吸和形成密封后的声音可能不同，因此声音可以作为 P0 `20 mm`
试抬前后的附加证据。本模块默认只调用麦克风录音并读取 WAV，不控制气泵、
不连接机械臂，也不发送任何运动。

声学判断不是安全级真空传感器。未通过独立留出集时只记录为诊断信息，
视觉试抬独立决定是否抓起；通过验收并显式启用为硬门禁后才采用：

```text
声学明确 sealed
AND 20 mm 试抬后药盒随机械臂移动
AND 原位置药盒消失
→ carton_held
```

在声学硬门禁模式下，`empty` 和 `uncertain` 都不得授权进入
`safe_transport`。未启用声学硬门禁时，声音不能否决也不能放行，仍由视觉
20 mm 跟随结果决定。

## 硬件与采样前提

- Ubuntu 能通过 ALSA `arecord` 读取一个固定麦克风。
- 麦克风建议固定在气泵、排气口或真空管附近 5–15 cm，不能在标定后移动。
- 尽量关闭麦克风 AGC、降噪和回声消除，使用 16 kHz、单声道 PCM。
- 机械臂和人说话都会污染声音；录音窗口内保持机械臂静止、现场安静。
- `empty` 不是“气泵关闭”，而是“气泵开启、吸盘悬空且未密封”。
- `sealed` 是“同一气泵开启、吸盘稳定吸住代表性药盒”。
- 气泵、管路、消音器、麦克风位置或环境发生变化后必须重新标定。

先查看录音设备：

```bash
cd ~/RUC-WONE/medicine_agentic
conda activate dos-w1
PYTHONPATH=src python scripts/vacuum_audio.py devices
```

## 1. 录制标定样本

建议至少 10 组交替采样，而不是先一次性录完一种状态。每次先由人手动建立
状态并稳定 1 秒，再按 Enter。CLI 不会切换气泵。

```bash
PYTHONPATH=src python scripts/vacuum_audio.py record \
  --label empty --count 1 --seconds 3

PYTHONPATH=src python scripts/vacuum_audio.py record \
  --label sealed --count 1 --seconds 3
```

重复以上两条命令至少 10 次。默认样本目录为：

```text
artifacts/vacuum_audio/samples/
├── empty/
└── sealed/
```

每个 WAV 旁边会保存一份 JSON，包含标签定义、录音设备、时长、削波率和特征。

## 2. 标定阈值

```bash
PYTHONPATH=src python scripts/vacuum_audio.py calibrate \
  --samples-dir artifacts/vacuum_audio/samples \
  --model-out configs/vacuum_audio_model.json \
  --report-out artifacts/vacuum_audio/calibration_report.json
```

模型使用帧级中位数/IQR、频谱质心、平坦度、频带能量比例、主峰比例和频谱
变化等特征。标定时空吸误报的默认代价高于漏检，并在阈值两侧保留不确定区间。

训练集报告只用于发现明显问题，不代表真实泛化能力。

## 3. 独立评估

隔一段时间、换不同药盒再录一套样本，存入例如：

```text
artifacts/vacuum_audio/holdout/empty/
artifacts/vacuum_audio/holdout/sealed/
```

不要把 holdout 样本重新用于标定：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py evaluate \
  --model configs/vacuum_audio_model.json \
  --samples-dir artifacts/vacuum_audio/holdout \
  --report-out artifacts/vacuum_audio/holdout_report.json
```

默认验收门槛：

- 空吸误报率 `0%`；
- 保守漏检率不超过 `5%`，其中 `uncertain` 也按吸附未通过计算；
- 总不确定率不超过 `10%`。

建议 holdout 至少包含 20 段空吸和 20 段成功吸附。任何空吸被判为
`sealed`，都不应启用声学验证进入自动运输。

## 4. 只读在线验证

外部系统先建立气泵状态，工具只录两秒并判断：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py verify \
  --model configs/vacuum_audio_model.json \
  --seconds 2
```

退出码：

- `0`：明确 `sealed`
- `2`：明确 `empty`
- `3`：`uncertain`
- `1`：录音、模型或数据错误

保存本次验证音频便于审计：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py verify \
  --model configs/vacuum_audio_model.json \
  --seconds 2 \
  --output-wav artifacts/vacuum_audio/latest_verification.wav
```

也可以分析已有 WAV：

```bash
PYTHONPATH=src python scripts/vacuum_audio.py predict \
  --model configs/vacuum_audio_model.json \
  --input example.wav
```

## 推荐接入方式

不要让声音模块直接调用泵或机器人。工作流只读取其判定：

1. 外部吸盘控制器执行吸附。
2. 机械臂静止，采集 1–2 秒稳定声音。
3. 声学输出必须连续多窗为 `sealed`。
4. 机械臂垂直试抬 20 mm。
5. 视觉确认药盒跟随和原位置消失。
6. 声学与视觉同时通过后，工作流才进入 `safe_transport`。

若现场已有真空压力开关，应以压力开关为主，声音只做冗余诊断。

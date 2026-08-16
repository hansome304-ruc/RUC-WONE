# 药盒六面参考图库

这个目录只保存经过人工确认的药盒参考图，不保存训练集。每个药盒 SKU
和版本独立成库：

```text
reference_faces/
└── medicine_carton_001/
    └── 1.0.0/
        ├── manifest.json
        └── images/
            ├── front_large/
            ├── back_large/
            ├── long_side_a/
            ├── long_side_b/
            ├── short_side_a/
            └── short_side_b/
```

六个面必须按实物固定标记。`front_large`、`back_large` 默认允许作为吸盘
抓取面；四个侧面默认只用于识别，不允许抓取。若需要改变抓取面，请在
`init` 时明确使用 `--pick-face`，不要直接手改已经 `ready` 的版本。

## 推荐采图

- 每个面准备 1–3 张图；第一张是正视、清晰、无手遮挡的基准图。
- 建议至少再补一张由机器人 D435 在实际工位拍摄的图。
- 固定曝光和白平衡；单个面占画面约 60%–80%，四角完整可见。
- 裁图完成后再执行 `add`。加入图库后不要再次编辑文件，否则哈希校验会失败。
- 用卡尺将长、宽、高各测 3 次，取中位数，单位为 mm。

任务一首轮只验证向上大面时，可以先加入 `front_large` 和 `back_large`
两张图并 `finalize`；未提供图片的四个侧面不会被验证，也绝不会获准吸取。
后续要识别侧面时，新建 `1.0.1` 版本再补齐四个侧面，不修改已经冻结的版本。

## 工作流

以下命令在项目根目录执行。把尺寸和图片路径替换为真实值：

```bash
python scripts/reference_faces.py init \
  --root reference_faces \
  --sku medicine_carton_001 \
  --version 1.0.0 \
  --length-mm 120.0 \
  --width-mm 70.0 \
  --height-mm 20.0

python scripts/reference_faces.py add \
  --manifest reference_faces/medicine_carton_001/1.0.0/manifest.json \
  --face front_large \
  --image /绝对路径/front_large.jpg
```

任务一首轮先对两个大面执行 `add` 即可；要覆盖任意朝向时，再在新版本中
对六个面分别执行。每个面最多加入 3 张。采集过程中可检查草稿结构：

```bash
python scripts/reference_faces.py validate \
  --manifest reference_faces/medicine_carton_001/1.0.0/manifest.json \
  --allow-draft
```

所有允许吸取的面都有图片后即可冻结版本：

```bash
python scripts/reference_faces.py finalize \
  --manifest reference_faces/medicine_carton_001/1.0.0/manifest.json

python scripts/reference_faces.py validate \
  --manifest reference_faces/medicine_carton_001/1.0.0/manifest.json
```

`finalize` 会逐张解码图片，检查路径、像素尺寸和 SHA-256，然后写入
manifest 自身的规范化 SHA-256。每个 `pick_allowed=true` 的面必须有
1–3 张图片；没有图片的非吸取面保持不可识别、不可吸取。`ready` 版本不可
再 `add`；需要换图或补面时新建 `1.0.1` 等版本。任何未知字段、路径穿越、
符号链接、坏图或哈希不一致都会让加载失败，不会降级为“尽量使用”。

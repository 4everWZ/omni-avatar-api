# Ditto 单张参考图校准

`POST /v1/avatar/generations` 在一次登记中绑定调用方的参考图、裁剪参数与手工标注。每次提交生成独立任务及形象。修改标注时，对同一资产创建新任务；同槽随后处理的形象独立选择自动或手工路径。

## 输入

`infra.version` 为 1。`infra.portrait.calibration` 可含以下任一字段或两者。坐标原点在 **640×720 上传资产规范化结果**的左上角，x 向右、y 向下，以像素计。客户端先按 EXIF 方向处理，等比缩放、居中填充后再标点，或直接上传完成标注的 640×720 图。

| 字段 | 契约 | 接入位置 |
|---|---|---|
| `face_box` | `[x_min,y_min,x_max,y_max]`，两个边长至少 1，边界可到 640、720 | 直接向原 landmark106 提供选定的人脸框。随后仍运行 203 点、两次原生裁剪及指定的 crop 参数。 |
| `eyes` | `left` 和 `right` 按看图者左右。每眼含外眼角 `outer`、内眼角 `inner`、上下眼睑中点 `upper/lower`、瞳孔中心 `pupil`，各为图内 `[x,y]`。 | 使用实际 512 裁剪的仿射变换换算标点。用中央睑距近似三组睑距之和，计算开度；以眼角中点及外眼角方向计算二维视线，深度为 0。输出固定版 Ditto 所需的 float32 `[1,2]` 与 `[1,6]`。 |

不传校准时走原人脸检测与 478 点眼部测量。只传脸框时仍测眼，仅传双眼时仍自动找脸。标注不参与外观提取、运动提取或后续音频驱动、生成与回贴。非法结构或越界坐标在排队前返回 400。登记回执保留实际提交的标注，按原任务所有权及删除规则读写。[固定版 Source2Info](https://github.com/antgroup/ditto-talkinghead/blob/c3e47eee2e626500017a0556b470d6d4182f85e8/core/atomic_components/source2info.py) · [眼部特征定义](https://github.com/antgroup/ditto-talkinghead/blob/c3e47eee2e626500017a0556b470d6d4182f85e8/core/utils/eye_info.py)

## 两份原稿的隔离验收

下面两份文件是根据原设定稿右上角头像标注的初值。它们配套使用现有 640×720 上传图，或使用下表区域从 1448×1086 原稿裁切后，等比居中填充至 640×720，背景色 `#ece7df`。裁区按 Pillow 的左、上、右、下顺序，右、下边界不含在内。

| 原稿 | 原稿裁区 | 登记选项 |
|---|---|---|
| `original.png` 常规版 | `[910,32,1184,295]` | [手工脸框](examples/ditto-original.json)，保留原眼部测量 |
| `Q_original.png` Q 版 | `[922,32,1202,273]` | [手工脸框与双眼标点](examples/ditto-q.json) |

在独立候选 worker 和独立数据目录内运行。各上传一次头像至 `POST /v1/avatar/assets`，记录返回的资产 ID。以下示例用 PowerShell 生成请求，`$base`、`$headers` 指向隔离网关及测试身份，`$assetId` 对应同一行的资产。

```powershell
$marks = Get-Content docs/examples/ditto-q.json -Raw | ConvertFrom-Json
$body = @{
    driver = 'ditto'
    reference_asset_id = $assetId
    infra = @{ version = 1; portrait = @{ calibration = $marks } }
}
$job = Invoke-RestMethod "$base/v1/avatar/generations" -Method Post -Headers $headers -ContentType application/json -Body ($body | ConvertTo-Json -Depth 10)
Invoke-RestMethod "$base/v1/avatar/generations/$($job.id)" -Headers $headers
```

1. 两份资产先各测自动登记；常规版再用配套脸框。Q 版依次仅传配套脸框、再传脸框加双眼。每次登记分别保留状态、实际标注和失败阶段。
2. 就绪形象按现有接口创建 audio 会话，两版播放相同的含停顿语音。记录接收端 WHEP 音视频、实际生成帧计数、音画偏差；人工核对五官、张合口和停顿，以及是否保持各自原稿身份。
3. 同槽再跑既有真人自动参考，核对无标注登记仍可生成真实帧。控制路径测试、本次标点与历史 GPU 帧率各自独立；候选实测前，两个标注文件均仅为初始值。

## 读取失败阶段

`GET /v1/avatar/generations/{id}` 保留 `state=failed` 和 `error=reference_preparation_failed`。本版 Ditto 准备错误附带 `diagnostic`，例如 `{"stage":"face_detection","reason":"no_face"}`。不同身份查询仍返回 404。

| stage | 对应处理 |
|---|---|
| `face_detection` | 原检测器无框时返回 `no_face`。检查头像区域或提供本图的框。 |
| `landmark106`、`landmark203`、`crop` | 原模型求点或几何裁剪。复核参考区域、脸框与现有裁剪选项。 |
| `eye_mesh` | 原 478 点或本图手工标点的眼部特征计算。框已通过后可只调整双眼。 |
| `motion_features`、`appearance_features` | 原生运动或外观提取。 |
| `registration`、`warmup`、`persist` | 参考图装载／特征汇总、槽位初次预热、特征文件保存。 |

缺失或非有限的标志点特征可返回 `invalid_features`，阶段内其他异常返回 `stage_failed`。原始异常留在 worker 日志中。旧作业、取消和重启沿用原状态。

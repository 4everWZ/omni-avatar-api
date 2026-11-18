# Repository tools

## 部署入口

`python tools/compose.py -- <Docker Compose 参数>` 默认选择分卡工作站。`--portrait`、`--vpn`、`--media` 开启可选层；`--target` 选择硬件／评估配置，`--env-file` 指定环境文件，`--dry-run` 查看命令。入口选项写在 `--` 之前，Docker 参数放在之后。模型目录和构建上下文以仓库根目录解析。

## 工作站实机验证

验证器调用真实网关并消费 WHEP 媒体，依赖使用 `services/avatar/requirements.txt`，隔离运行方式见[开发指南](../docs/development.md)。`.env` 配置主应用及独立的额外应用令牌；报告保存指标与验证结果。共享语音和长文输入由 `evaluations/fixtures/` 维护。

```sh
python -m tools.qualification.workstation --env .env --avatar-only --seconds 60 --report tmp/avatar-check.json
python -m tools.qualification.inference --env .env --seconds 1200 --report tmp/inference-functional.json
python -m tools.qualification.concurrency --env .env --report tmp/llm-concurrency.json
python -m tools.qualification.portrait --env .env --reference /path/to/portrait.png --seconds 120 --report tmp/portrait-functional.json
```

| 工具 | 验证内容 |
| --- | --- |
| `qualification.workstation --avatar-only` | Ava 2D 音画、跨调用方隔离与幂等释放。 |
| `qualification.inference` | Omni 8192 与 LLM 262144 模型配置、Omni 约 7.9K 多轮检索与超限 400、Responses 约 22K 历史及归属续接、JSON Schema、function 输出与夹具结果回填、增量 SSE、后台取消及独立准入；随后进行 1200 秒原生双工／Ava 2D 与每三十秒后台请求。 |
| `qualification.concurrency` | 四路输出区间重叠、约 28K 输入与三路短请求混跑、worker 独立续接、token 压力 429、短 Chat 准入及取消释放。 |
| `qualification.portrait` | 参考图上传、异步准备与归属、原生音频绑定、真实神经帧、共享音画时间线、取消恢复及释放。 |
| `qualification.thinking` | 默认 CoT 完整答案、显式覆盖、思考预算、非法值、JSON Schema 和工具续接。完整答案夹具显式使用 4096 总输出额度。 |

`concurrency --engine-metrics http://<引擎私网地址>:8000/metrics` 可采样实际运行序列并要求达到 4；采样通过引擎私网进行。`portrait --seconds 90 --concurrent-llm 4` 在原生通话期间执行四路各 256-token SSE，检查输出区间重叠、25 fps 真实生成、采样偏差小于 160 ms 及取消恢复。

`qualification.workstation_capacity --mps-percent 20 --avatar-id <id>` 测两路 Chat、Chat 与双工混跑、两个绑定 Ditto 的双工会话，以及双头像运行期间单路／四路 Qwen。更改 MPS 上限后使用相同输入重复执行。

`qualification.images --reference <portrait.png> --anime-reference <character.png>` 在工作站生成真人参考、动漫参考和文生图，保存 PNG、各阶段耗时、100 ms 间隔的整卡与主机内存峰值和 Responses 续接结果。按相同种子和输入进行目视对比。

`portrait`、`throughput` 和 `omni_throughput` 支持 `--avatar-id <已有形象ID>` 或 `--reference <PNG>`。准备新形象时使用清楚的单人正面图片。神经帧计数来自 `renderer.generated_frames`，WHEP 解码帧另行统计；诊断图片保存在报告旁。

## TPS 与混跑

```sh
python -m tools.qualification.throughput --env .env --avatar-id <id> --seconds 90 --report tmp/throughput.json
python -m tools.qualification.omni_throughput --env .env --avatar-id <id> --seconds 90 --sliding-window-mode basic --report tmp/omni-throughput.json
python -m tools.qualification.thinking --env .env --report tmp/thinking.json
```

`throughput` 测量后台空闲单路／四路，以及原生语音和 25 fps 头像期间的单路／四路。空闲单路至少 45 TPS、通话四路每路至少 15 TPS 为对应负载的回归检查线。`omni_throughput` 测量前台空闲单路、两个同时提交请求，以及原生通话、头像、四路 LLM 共驻期间的额外一路 Omni Chat。报告记录输出区间是否实际重叠。

两个工具默认 `--thinking enabled`，温度 0、相同短提示、每路 256-token 总输出，先预热 32 tokens；`--thinking disabled` 供显式策略对照。端到端 TPS 按引擎 usage 的思考＋正文 tokens 计量，包含预填充、传输与等待。报告分别保留思考／正文首增量、增量数、输出区间及重叠情况；256-token 夹具覆盖总生成吞吐，完整答复由 `thinking` 夹具验证。`thinking_token_budget=-1` 采用上游语义，由总输出和上下文额度约束思考；非法负值使用 -2。

全栈通过条件同时覆盖 25 fps、检查期最大采样偏差小于 160 ms、原生取消恢复和幂等释放。原生语音按音频块、答复内容及媒体时间线判定。实时窗口通过 `autostart=0` 后的首条会话更新配置，默认 `basic`；`context` 可用于策略对照。当前 TPS 和原始记录见[工作站指南](../docs/deployment/workstation.md#四路并发与计算预算)。

## 采集与验证范围

报告记录持续时间、音画帧、非静音帧、请求数、隔离及释放。`metrics_collector_host` 标明资源采集位置；工作站 GPU 数据应在工作站采集。可使用现有 Avatar 镜像、host 网络和 `--gpus all`，以读取 `.env` 的宿主 UID 运行，挂载代码与单独可写的报告目录。

自动检查覆盖 API、媒体解码、时间线和资源生命周期。浏览器实物播放、麦克风／屏幕权限、主观口型与持续满载 QoS 由[后续验收计划](../docs/plans/avatar-workstation.md)维护。当前功能检查周期约半小时。

## 模型诊断

`tools/diagnostics/` 用于模型与硬件边界定位。`minicpmo_text` 从同一 BF16 快照严格加载 `llm.*` 文本骨干，使用 Transformers SDPA、BF16 KV 和贪心解码执行短问答与固定长文检索；报告以各 case 的 `correct` 表示答案判定。该探针覆盖直接文本骨干，服务接口由上述资格验证器覆盖。

```sh
python -m tools.diagnostics.minicpmo_text --checkpoint /models/<revision> --report tmp/text-reference.json
```

探针按层使用两张 GPU，运行于固定引擎镜像的 PyTorch、Transformers、Accelerate 与 safetensors 环境。运行前排空请求并停止占用 GPU 的引擎，挂载仓库与只读模型目录，结束后恢复 Compose 服务。

## TensorRT 帧对照

`tools.qualification.portrait_frame --prepared <本机准备特征.pkl> --report <可写报告路径>` 在挂载仓库的独立 TensorRT 头像镜像中运行。`--root` 指定固定 Ditto 资产根目录，`--iterations` 默认 100，`--backends` 默认依次比较 `cuda-graph tensorrt`。报告目录与 `/cache/tensorrt` 按运行 UID 配置读写权限。

该工具使用同一真实准备特征和三组固定种子关键点，保存 CUDA Event 均值／p50／p95、像素 MAE／最大差异／PSNR，以及对照 PNG。`state=completed` 表示测量完成，画面质量按数值和人工检查判定。帧延迟测量时为 GPU 0 保留独立运行空间；共驻 TPS 由 `throughput` 与 `omni_throughput` 在恢复完整服务后测量。[构建与预编译](../services/portrait/README.md#tensorrt-frame-backend) · [当前对照数据](../docs/deployment/workstation.md#tensorrt-fp16-对照)。

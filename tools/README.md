# Repository tools

## 部署入口

`python tools/compose.py -- <Docker Compose 参数>` 默认选择当前分卡工作站。`--portrait`、`--vpn`、`--media` 开启可选层；`--target` 选择历史／评估硬件配置，`--env-file` 指定环境文件，`--dry-run` 查看命令。选项写在 `--` 之前，Docker 参数放在之后。入口固定仓库根目录解析模型目录和构建上下文。

## 工作站实机验证

`tools.qualification.workstation` 使用真实网关和 WHEP，不启动模拟引擎。依赖与 `services/avatar/requirements.txt` 相同。`.env` 需配置主应用及一个独立的额外应用令牌；凭据不写进报告。

```sh
python -m tools.qualification.workstation --env .env --avatar-only --seconds 60 --report tmp/avatar-check.json
python -m tools.qualification.inference --env .env --seconds 1200 --report tmp/inference-functional.json
```

历史单引擎 `tools.qualification.workstation` 完整模式检查模型身份、40,960 窗口、普通／流式推理、JSON Schema、视觉颜色和接近窗口上限的长文编号检索（重复记录、多样库存两种文档），再运行一路原生双工及 Ava，并由第二调用方每三十秒请求普通推理。该默认流量属于轻量后台请求，不是持续满载；完整负载强度须按实际调用方需求确定。原生输入、普通请求和媒体接收各自运行，普通请求耗时不阻塞音频输入。前九十秒没有原生语音输出即失败；结束时还需在字幕中找到语音问题的正确答案。

报告包括实际时长、解码音画帧及非静音帧、普通请求数、调用方隔离和幂等释放结果。GPU／主机指标来自运行验证器的机器，`metrics_collector_host` 标明采集位置；远端客户端运行的 GPU 数字不代表工作站。解码并消费媒体也不代表真实扬声器和显示器已经播放。

在工作站运行可以收集其 GPU 和内存指标。也可使用现有 Avatar 镜像，挂载代码及单独可写的报告目录，以读取 `.env` 的宿主 UID 运行，使用 host 网络和 `--gpus all`；不需要修改宿主 Python。

完整模式还用持续 SSE 占据普通准入，检查另一请求排队到期返回 429，再断开 SSE 并确认容量释放和后续推理可用。该工具不替代浏览器对麦克风、屏幕采集、重连、静默、重叠语音、字幕同步、打断和渲染故障降级的验收。当前按用户要求运行约半小时功能检查，不要求两小时；短报告不表示长时间压力测试通过。

`tools.diagnostics.minicpmo_text` 是长文定位工具：只读取同一官方 BF16 快照的 `llm.*` 权重，严格加载 Qwen3 文本骨干，用 Transformers SDPA、BF16 KV 和贪心解码运行短问答及相同的 39,932-token 检索。它按层分布到两张 GPU；使用现有引擎镜像中的 PyTorch、Transformers、Accelerate 与 safetensors，不需修改宿主依赖。运行前停止占 GPU 的引擎，挂载仓库与只读模型目录，用 `--gpus all`，结束后恢复原 Compose 服务。入口为 `python -m tools.diagnostics.minicpmo_text --checkpoint /models/<revision> --report tmp/text-reference.json`。它不经过 Omni 的多模态编码器、服务调度或输出解析；`state=completed` 只表示完成测试，必须检查各 case 的 `correct`。该结果不替代 Omni API 或原生双工验收。

`tools.qualification.inference` 是当前分卡主线验证器：先检查 Omni 8192、LLM 262144 的模型配置、Omni 约 7.9K-token 多轮早期事实与超限 400、Responses 归属续接、约 22K-token 多轮历史、JSON Schema、function 输出与测试夹具回填、增量 SSE、后台取消和独立准入，再进行 1200 秒原生双工／Ava 2D 与每三十秒后台 Responses 请求。它不实现 Agent。已通过的实机结果见 [功能报告](../evaluations/results/inference-functional-workstation.json)；完整 262K 窗口质量不在该通过结论内。

`tools.qualification.portrait --reference <单人PNG> --seconds 120 --report tmp/portrait-functional.json` 验证参考图上传、异步准备、跨调用方隔离、真实神经视频、原生语音绑定及后台 Responses 共驻。它按测试时长检查 25 fps 的实际生成总量、帧进度及共享音画时间线采样偏差、取消后的代次隔离与恢复和幂等释放。输出诊断帧保存在报告旁；`renderer.generated_frames` 来自神经帧接收计数，不以 WHEP 的重复画面凑数。短功能测试不证明持续满载 QoS 或物理扬声器播放。

共享语音和长文输入位于 `evaluations/fixtures/`，验收与诊断工具共同使用该层，彼此不导入流程实现。

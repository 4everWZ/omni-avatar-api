# 工作站实机验证

`qualify_workstation.py` 使用真实网关和 WHEP，不启动模拟引擎。依赖与 `avatar/requirements.txt` 相同。`.env` 需配置主应用及一个独立的额外应用令牌；凭据不写进报告。

```sh
python tools/qualify_workstation.py --env .env --avatar-only --seconds 60 --report tmp/avatar-check.json
python tools/qualify_workstation.py --env .env --seconds 7200 --report tmp/workstation-qualification.json
```

完整模式先检查模型身份、40,960 窗口、普通／流式推理、JSON Schema、视觉颜色和接近窗口上限的长文编号检索（重复记录、多样库存两种文档），再运行一路原生双工及 Ava，并由第二调用方每三十秒请求普通推理。该默认流量属于轻量后台请求，不是持续满载；完整负载强度须按实际调用方需求确定。原生输入、普通请求和媒体接收各自运行，普通请求耗时不阻塞音频输入。前九十秒没有原生语音输出即失败；结束时还需在字幕中找到语音问题的正确答案。

报告包括实际时长、解码音画帧及非静音帧、普通请求数、调用方隔离和幂等释放结果。GPU／主机指标来自运行验证器的机器，`metrics_collector_host` 标明采集位置；远端客户端运行的 GPU 数字不代表工作站。解码并消费媒体也不代表真实扬声器和显示器已经播放。

在工作站运行可以收集其 GPU 和内存指标。也可使用现有 Avatar 镜像，挂载代码及单独可写的报告目录，以读取 `.env` 的宿主 UID 运行，使用 host 网络和 `--gpus all`；不需要修改宿主 Python。

完整模式还用持续 SSE 占据普通准入，检查另一请求排队到期返回 429，再断开 SSE 并确认容量释放和后续推理可用。该工具不替代浏览器对麦克风、屏幕采集、重连、静默、重叠语音、字幕同步、打断和渲染故障降级的验收。报告中 `scope=avatar_only` 或时长不足 7200 秒，均不能作为两小时 Omni 混合负载通过的证据。

`reference_minicpmo_text.py` 是长文定位工具：只读取同一官方 BF16 快照的 `llm.*` 权重，严格加载 Qwen3 文本骨干，用 Transformers SDPA、BF16 KV 和贪心解码运行短问答及相同的 39,932-token 检索。它按层分布到两张 GPU；使用现有引擎镜像中的 PyTorch、Transformers、Accelerate 与 safetensors，不需修改宿主依赖。运行前停止占 GPU 的引擎，挂载仓库与只读模型目录，用 `--gpus all`，结束后恢复原 Compose 服务。入口为 `python3 tools/reference_minicpmo_text.py --checkpoint /models/<revision> --report tmp/text-reference.json`。它不经过 Omni 的多模态编码器、服务调度或输出解析；`state=completed` 只表示完成测试，必须检查各 case 的 `correct`。该结果不替代 Omni API 或原生双工验收。

# 工作站实机验证

`qualify_workstation.py` 使用真实网关和 WHEP，不启动模拟引擎。依赖与 `avatar/requirements.txt` 相同。`.env` 需配置主应用及一个独立的额外应用令牌；凭据不写进报告。

```sh
python tools/qualify_workstation.py --env .env --avatar-only --seconds 60 --report tmp/avatar-check.json
python tools/qualify_workstation.py --env .env --seconds 7200 --report tmp/workstation-qualification.json
```

完整模式先检查模型身份、40,960 窗口、普通／流式推理、JSON Schema、视觉颜色和接近窗口上限的长文编号检索，再运行一路原生双工及 Ava，并由第二调用方每三十秒请求普通推理。原生输入、普通请求和媒体接收各自运行，普通请求耗时不阻塞音频输入。前九十秒没有原生语音输出即失败；结束时还需在字幕中找到语音问题的正确答案。

报告包括实际时长、解码音画帧及非静音帧、普通请求数、调用方隔离和幂等释放结果。GPU／主机指标来自运行验证器的机器，`metrics_collector_host` 标明采集位置；远端客户端运行的 GPU 数字不代表工作站。解码并消费媒体也不代表真实扬声器和显示器已经播放。

在工作站运行可以收集其 GPU 和内存指标。也可使用现有 Avatar 镜像，挂载代码及单独可写的报告目录，以读取 `.env` 的宿主 UID 运行，使用 host 网络和 `--gpus all`；不需要修改宿主 Python。

完整模式还用持续 SSE 占据普通准入，检查另一请求排队到期返回 429，再断开 SSE 并确认容量释放和后续推理可用。该工具不替代浏览器对麦克风、屏幕采集、重连、静默、重叠语音、字幕同步、打断和渲染故障降级的验收。报告中 `scope=avatar_only` 或时长不足 7200 秒，均不能作为两小时 Omni 混合负载通过的证据。

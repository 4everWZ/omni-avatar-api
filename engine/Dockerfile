FROM vllm/vllm-omni:v0.32.0-aarch64@sha256:6da195b87ca635601559ea11a6dd175758b0fa68cbada3a4529e01f36bc92df1

COPY vocoder.lock /opt/tuber-omni/vocoder.lock
RUN uv pip install --python "$(command -v python3)" --no-deps --require-hashes --no-cache-dir \
        -r /opt/tuber-omni/vocoder.lock \
    && python3 -c 'import cosyvoice2, hyperpyyaml, onnxruntime, s3tokenizer, torchaudio'

COPY model.py minicpmo-orin.yaml /opt/tuber-omni/

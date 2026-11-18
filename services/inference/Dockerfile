ARG OMNI_BASE_IMAGE=vllm/vllm-omni:v0.32.0-aarch64@sha256:6da195b87ca635601559ea11a6dd175758b0fa68cbada3a4529e01f36bc92df1
FROM ${OMNI_BASE_IMAGE}

COPY vocoder.lock /opt/tuber-omni/vocoder.lock
RUN env -u UV_CONSTRAINT -u UV_OVERRIDE uv --no-config pip install \
        --python "$(command -v python3)" --no-deps --require-hashes --no-cache-dir \
        -r /opt/tuber-omni/vocoder.lock \
    && python3 -c 'import cosyvoice2, hyperpyyaml, onnxruntime, s3tokenizer, torchaudio'

COPY model.py /opt/tuber-omni/
COPY configs/ /opt/tuber-omni/configs/
COPY patches/ /opt/tuber-omni/patches/
RUN python3 /opt/tuber-omni/patches/restore_chat_constraints.py \
    && python3 /opt/tuber-omni/patches/patch_minicpmo_int4_window.py \
    && python3 /opt/tuber-omni/patches/patch_minicpmo_awq_stages.py \
    && python3 /opt/tuber-omni/patches/patch_chat_prompt_tokens.py \
    && python3 /opt/tuber-omni/patches/patch_reasoning_budget.py \
    && python3 /opt/tuber-omni/patches/patch_minicpmo_duplex_idle.py

ARG PORTRAIT_BASE_IMAGE=vllm/vllm-omni:v0.32.0-x86_64@sha256:6b44ee504a0dc3a21eacf72e7dc5da318b7bcc8ed0e349dc928e67278414430c
ARG PORTRAIT_RUNTIME=cuda-graph
FROM ${PORTRAIT_BASE_IMAGE} AS cuda-graph
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/portrait-env
COPY requirements.txt /opt/portrait/requirements.txt
RUN /opt/portrait-env/bin/python -m pip install --no-cache-dir -r /opt/portrait/requirements.txt
ENV PATH="/opt/portrait-env/bin:${PATH}" HOME=/tmp
WORKDIR /srv
RUN install -d -o 10001 -g 10001 /run/omni-portrait /run/omni-gpu /data /cache
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
FROM cuda-graph AS tensorrt
COPY requirements-tensorrt.txt /opt/portrait/requirements-tensorrt.txt
RUN /opt/portrait-env/bin/python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    -r /opt/portrait/requirements-tensorrt.txt

FROM ${PORTRAIT_RUNTIME} AS worker
RUN python -c "import torch, onnxruntime, cv2, mediapipe; assert not torch.__file__.startswith('/opt/portrait-env/')"
COPY *.py /srv/portrait/
USER 10001:10001
CMD ["python", "-u", "-m", "portrait"]

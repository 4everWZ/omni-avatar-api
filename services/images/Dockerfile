ARG IMAGE_BASE_IMAGE=vllm/vllm-omni:v0.32.0-x86_64@sha256:6b44ee504a0dc3a21eacf72e7dc5da318b7bcc8ed0e349dc928e67278414430c
FROM ${IMAGE_BASE_IMAGE}
COPY requirements.txt /srv/images-requirements.txt
RUN env -u UV_CONSTRAINT -u UV_OVERRIDE uv --no-config pip install --system --no-deps --no-cache-dir -r /srv/images-requirements.txt \
    && python3 -c 'import aiohttp, PIL, pynvml; from diffusers import Flux2KleinPipeline'
WORKDIR /srv
RUN install -d -o 10001 -g 10001 /run/omni-gpu
COPY *.py /srv/images/
USER 10001:10001
ENV HOME=/tmp
ENTRYPOINT ["python3", "-u", "-m", "images"]

ARG PORTRAIT_BASE_IMAGE=omni-api-engine:0.32.0-workstation
FROM ${PORTRAIT_BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/portrait-env
COPY requirements.txt /opt/portrait/requirements.txt
RUN /opt/portrait-env/bin/python -m pip install --no-cache-dir -r /opt/portrait/requirements.txt
ENV PATH="/opt/portrait-env/bin:${PATH}" HOME=/tmp
WORKDIR /srv
RUN install -d -o 10001 -g 10001 /run/omni-portrait /run/omni-gpu /data /cache
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
COPY *.py /srv/portrait/
USER 10001:10001
CMD ["python", "-u", "-m", "portrait"]

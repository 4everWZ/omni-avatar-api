FROM omni-api-engine:0.32.0-workstation
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/portrait-env
COPY requirements.txt /opt/portrait/requirements.txt
RUN /opt/portrait-env/bin/python -m pip install --no-cache-dir -r /opt/portrait/requirements.txt
ENV PATH="/opt/portrait-env/bin:${PATH}" HOME=/tmp
WORKDIR /srv/portrait
RUN install -d -o 10001 -g 10001 /run/omni-portrait /data /cache
ENV TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton
COPY *.py ./
USER 10001:10001
CMD ["python", "-u", "service.py"]

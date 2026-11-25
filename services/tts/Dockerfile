FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

RUN apt-get update && apt-get install -y --no-install-recommends \
        sox \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt /srv/tts-requirements.txt
RUN pip install --no-cache-dir -r /srv/tts-requirements.txt \
    && useradd -u 10001 -m -s /usr/sbin/nologin tts \
    && install -d -o 10001 -g 10001 /models

RUN install -d -m 0755 /srv/tts
COPY --chmod=0644 *.py /srv/tts/
USER 10001:10001
ENV HOME=/tmp
EXPOSE 8000
ENTRYPOINT ["python3", "-u", "-m", "tts"]
CMD ["serve"]

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt /srv/asr-requirements.txt
RUN pip install --no-cache-dir -r /srv/asr-requirements.txt \
    && useradd -u 10001 -m -s /usr/sbin/nologin asr \
    && install -d -o 10001 -g 10001 /models/asr

COPY *.py /srv/asr/
USER 10001:10001
ENV HOME=/tmp
ENTRYPOINT ["python3", "-u", "-m", "asr"]

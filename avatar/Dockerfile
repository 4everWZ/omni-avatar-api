FROM python:3.12-slim-bookworm
WORKDIR /srv/avatar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN install -d -o 10001 -g 10001 /run/omni-avatar
COPY *.py ./
COPY avatars.json ./
COPY assets ./assets
USER 10001:10001
CMD ["python", "-u", "server.py"]

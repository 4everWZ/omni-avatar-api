FROM python:3.12-slim-bookworm
WORKDIR /srv
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN install -d -m 0755 /srv/gateway
COPY --chmod=0644 *.py /srv/gateway/
COPY web /srv/gateway/web
USER 10001:10001
EXPOSE 8000
CMD ["python", "-u", "-m", "gateway"]

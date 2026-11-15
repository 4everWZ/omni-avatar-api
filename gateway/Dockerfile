FROM python:3.12-slim-bookworm
WORKDIR /srv/gateway
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway.py .
USER 10001:10001
EXPOSE 8000
CMD ["python", "-u", "gateway.py"]

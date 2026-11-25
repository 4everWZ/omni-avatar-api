FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

WORKDIR /srv
COPY requirements.txt /srv/omni-requirements.txt
RUN pip install --no-cache-dir -r /srv/omni-requirements.txt \
    && useradd -u 10001 -m -s /usr/sbin/nologin omni \
    && install -d -o 10001 -g 10001 /models

COPY *.py /srv/omni/
USER 10001:10001
ENV HOME=/tmp
ENTRYPOINT ["python3", "-u", "-m", "omni"]
CMD ["serve"]

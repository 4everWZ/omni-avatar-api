FROM python:3.12-slim-bookworm
WORKDIR /srv
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN install -d -o 10001 -g 10001 /run/omni-avatar
COPY *.py /srv/avatar/
COPY avatars.json /srv/avatar/
COPY assets /srv/avatar/assets
USER 10001:10001
CMD ["python", "-u", "-m", "avatar"]

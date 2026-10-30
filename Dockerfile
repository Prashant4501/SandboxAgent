FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 runner && \
    pip install --no-cache-dir pytest

USER runner
WORKDIR /app

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --retries 10 --default-timeout 120 -i https://pypi.org/simple -r /app/requirements.txt

COPY app /app/app
COPY pages /app/pages
COPY assets /app/assets

CMD ["/bin/sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

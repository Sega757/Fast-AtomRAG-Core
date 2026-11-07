# Single stage: all runtime deps ship as manylinux wheels, no compiler needed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ATOMRAG_DB_PATH=/data/atoms.db

WORKDIR /app

COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . \
    && useradd --uid 10001 --system --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app /data

USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status == 200 else 1)"]

# One worker by design: the in-memory index is owned by a single process (docs/ADR.md, ADR-001).
CMD ["uvicorn", "fast_atomrag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "10"]

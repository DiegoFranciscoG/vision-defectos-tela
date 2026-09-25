# Inference API: FastAPI + ONNX Runtime on CPU. No PyTorch in this image.
# Multi-stage, pinned versions, non-root user and health check.

FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.14.7-slim-trixie AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.14.7-slim-trixie AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_DIR=/app/models \
    MODEL_CACHE_DIR=/app/.cache/models \
    MODEL_REGISTRY=/app/models/registry.json
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Only the registry goes into the image. The ONNX files come from a read-only mount (local
# `docker compose`) or are downloaded from the GitHub Release on start, and are loaded only if
# their SHA-256 matches the registry.
COPY models/registry.json ./models/registry.json
RUN mkdir -p /app/.cache/models /app/data && chown -R app:app /app/.cache /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\", timeout=4)"]
CMD ["sh", "-c", "fabric-inspection init-db && fabric-inspection seed-demo && exec fabric-inspection serve"]

# Streamlit demo (thin client of the API). Dependencies installed with pinned hashes.

FROM python:3.12.14-slim-trixie AS builder
WORKDIR /build
COPY demo/requirements.lock ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock

FROM python:3.12.14-slim-trixie AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY demo/app.py demo/api_client.py ./demo/
COPY demo/samples ./demo/samples
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

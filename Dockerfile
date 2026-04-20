# syntax=docker/dockerfile:1

FROM ghcr.io/meta-pytorch/openenv-base:latest

LABEL org.opencontainers.image.title="OpenEnv Clinical Trial Designer"
LABEL org.opencontainers.image.source="https://github.com/suyashkumar102/openenv-clinical-trialv2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pydantic==2.7.1 \
        pydantic-settings==2.2.1

# Copy source and install local package
COPY server/ ./server/
RUN pip install --no-cache-dir --no-deps .

# Health check: GET /ping must respond within 30s
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/ping')"

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT}"]

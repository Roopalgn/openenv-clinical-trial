# syntax=docker/dockerfile:1
# NOTE: ghcr.io/meta-pytorch/openenv-base:0.2.1 is not publicly available.
# Using python:3.11-slim as the fallback base image per task 32 instructions.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="OpenEnv Clinical Trial Designer"
LABEL org.opencontainers.image.source="https://github.com/suyashkumar102/openenv-clinical-trialv2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_PATH=./logs \
    CURRICULUM_START_TIER=0

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip==24.0 \
 && pip install --no-cache-dir \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pydantic==2.7.1 \
        pydantic-settings==2.2.1 \
        scipy==1.13.0 \
        numpy==1.26.4 \
        matplotlib==3.8.4 \
        pandas==2.2.2

# Copy source and install local package
COPY server/ ./server/
COPY models.py ./
RUN pip install --no-cache-dir --no-deps .

# Copy entrypoint script and make executable
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Create non-root user and transfer ownership
RUN useradd --no-create-home --shell /bin/false appuser \
 && chown -R appuser:appuser /app

USER appuser

# Health check: GET /ping must respond within 30s
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/ping')"

ENTRYPOINT ["./entrypoint.sh"]

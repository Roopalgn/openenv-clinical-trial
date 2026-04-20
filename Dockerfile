# syntax=docker/dockerfile:1

# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11.9-slim

# Metadata
LABEL org.opencontainers.image.title="OpenEnv Clinical Trial Designer"
LABEL org.opencontainers.image.source="https://github.com/your-org/openenv-clinical-trial"

# Prevent .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Create a non-root user (while still root)
RUN addgroup --system appgroup \
 && adduser --system --ingroup appgroup --no-create-home appuser

# Install Python dependencies (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip==24.0 \
 && pip install --no-cache-dir --no-build-isolation \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pydantic==2.7.1 \
        pydantic-settings==2.2.1

# Copy application source and install the local package
COPY environment/ ./environment/
RUN pip install --no-cache-dir --no-deps .

# Drop to non-root user
USER appuser

EXPOSE 8000

# Health check: GET /ping must respond within 30 s of container start
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/ping')"

CMD ["sh", "-c", "uvicorn environment.app:app --host ${HOST} --port ${PORT}"]

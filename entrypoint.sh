#!/bin/sh
set -e

# Apply defaults for optional vars
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_PATH:=./logs}"
: "${CURRICULUM_START_TIER:=0}"

export HOST PORT LOG_PATH CURRICULUM_START_TIER

# Validate PORT is a number
case "$PORT" in
  ''|*[!0-9]*)
    echo "[entrypoint] ERROR: PORT must be a positive integer, got: '${PORT}'" >&2
    exit 1
    ;;
esac

# Validate CURRICULUM_START_TIER is a number in range 0-4
case "$CURRICULUM_START_TIER" in
  ''|*[!0-9]*)
    echo "[entrypoint] ERROR: CURRICULUM_START_TIER must be an integer 0-4, got: '${CURRICULUM_START_TIER}'" >&2
    exit 1
    ;;
esac

echo "[entrypoint] Starting OpenEnv Clinical Trial Designer"
echo "[entrypoint] HOST=${HOST} PORT=${PORT} LOG_PATH=${LOG_PATH} CURRICULUM_START_TIER=${CURRICULUM_START_TIER}"

exec uvicorn server.app:app --host "$HOST" --port "$PORT"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
exec uv run uvicorn pointremote.main:app --host 0.0.0.0 --port "${POINTREMOTE_PORT:-8765}"

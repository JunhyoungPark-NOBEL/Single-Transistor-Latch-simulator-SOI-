#!/usr/bin/env bash
# Local development: FastAPI backend (:8000, auto-reload on server/ edits) + Vite dev server (:5173, proxies /api).
#   scripts/dev.sh                 # both
#   scripts/dev.sh api             # backend only
#   scripts/dev.sh web             # frontend only (expects the API on $API_PORT)
# Env: API_PORT (8000), WEB_PORT (5173), STL_WORKERS (2 here), STL_CACHE_DIR, STL_PREWARM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
export STL_WORKERS="${STL_WORKERS:-2}"
MODE="${1:-all}"
PY="${PYTHON:-python3}"

run_api() {
  exec "$PY" -m uvicorn server.main:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir server
}

run_web() {
  cd "$ROOT/web"
  [ -d node_modules ] || npm ci --no-audit --no-fund
  STL_API="http://127.0.0.1:$API_PORT" exec npx vite --port "$WEB_PORT" --strictPort
}

case "$MODE" in
  api) run_api ;;
  web) run_web ;;
  all)
    "$PY" -m uvicorn server.main:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir server &
    API_PID=$!
    trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
    echo "[dev] API  http://127.0.0.1:$API_PORT  (pid $API_PID, $STL_WORKERS workers)"
    echo "[dev] Web  http://127.0.0.1:$WEB_PORT"
    ( run_web )
    ;;
  *) echo "usage: $0 [all|api|web]" >&2; exit 2 ;;
esac

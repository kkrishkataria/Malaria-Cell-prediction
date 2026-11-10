#!/usr/bin/env bash
set -e

cleanup() {
  kill "$api_pid" "$web_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 -m uvicorn app.main:app --reload --port 8000 &
api_pid=$!
(cd frontend && npm run dev -- --host 127.0.0.1) &
web_pid=$!
wait "$api_pid" "$web_pid"

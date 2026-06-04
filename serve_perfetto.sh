#!/usr/bin/env bash
# Serve a PyTorch-profiler chrome trace for the Perfetto UI.
#
# The PyTorch profiler (train_gpt2.py with PROFILE=1) writes its chrome trace to
# logs/<run_id>/trace.json. This starts a trace_processor RPC server on
# 127.0.0.1:$PORT, which https://ui.perfetto.dev auto-detects and uses.
#
# Usage:
#   ./serve_perfetto.sh [path/to/trace.json]   # default: newest logs/*/trace.json
#   PERFETTO_PORT=9002 ./serve_perfetto.sh      # override port
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PERFETTO_PORT:-9001}"
TRACE="${1:-}"

if [ -z "$TRACE" ]; then
  # wait for a trace to show up (e.g. while a PROFILE=1 run is compiling/running)
  until TRACE=$(ls -t logs/*/trace.json 2>/dev/null | head -1) && [ -n "$TRACE" ]; do
    echo "[serve_perfetto] no logs/*/trace.json yet — run train_gpt2.py with PROFILE=1. retrying..."
    sleep 3
  done
fi

echo "[serve_perfetto] serving: $TRACE"
echo "[serve_perfetto] open https://ui.perfetto.dev  (it connects to localhost:$PORT)"
echo "[serve_perfetto] if you're remote: ssh -L $PORT:localhost:$PORT <this-host>"
exec trace_processor server http --port "$PORT" "$TRACE"

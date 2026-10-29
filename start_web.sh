#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export PYTHONUTF8=1
SOPG_PY="${SOPG_PY:-$(pwd)/.venv/bin/python}"
PY="$SOPG_PY"

if [ ! -x "$PY" ]; then
  echo "Python venv not found: $PY"
  echo "See README.md 'Install' to rebuild it, or set SOPG_PY to your python binary."
  exit 1
fi

if [ ! -f "models/hand_landmarker.task" ]; then
  echo "models/hand_landmarker.task is missing. See models/README.md to download it."
  exit 1
fi

echo "Starting SOP-Guard web monitor (open the URL below in a browser)..."
"$PY" web_server.py --port 8765 "$@"

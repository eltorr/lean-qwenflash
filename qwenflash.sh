#!/bin/bash
# QwenFlash on Linux: the manager (dashboard, OpenAI API, model lifecycle, setup) with its one dependency (psutil).
#   ./qwenflash.sh                 start the service and print the dashboard URL
#   ./qwenflash.sh detect | setup [STEP] | status | plan | open      see: ./qwenflash.sh --help
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv-manager/bin/python ]; then
  echo "creating the manager's Python environment (.venv-manager)..."
  python3 -m venv .venv-manager
  .venv-manager/bin/pip install -q psutil
fi
exec .venv-manager/bin/python -m qwenflash "$@"

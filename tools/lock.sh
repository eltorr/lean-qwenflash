#!/usr/bin/env bash
# Regenerates the hash-pinned package lists from their inputs (needs uv: pip install uv):
#   setup/requirements.in      -> setup/requirements.txt      the engine (Linux / WSL, Python 3.12, CUDA 12.8)
#   qwenflash/requirements.in  -> qwenflash/requirements.txt  the manager (Windows and Linux)
#   tools/scan-requirements.in -> tools/scan-requirements.txt the scanners (tools/scan.sh)
# Installs then fetch exactly these files: setup/20_env.sh and QwenFlash.cmd use pip --require-hashes.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
UV=${UV:-uv}
TORCH_INDEX=https://download.pytorch.org/whl/cu128
# torch comes from PyTorch's index and the rest from PyPI; the hashes make mixing them safe
"$UV" pip compile setup/requirements.in -o setup/requirements.txt --generate-hashes --no-header \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
"$UV" pip compile qwenflash/requirements.in -o qwenflash/requirements.txt --generate-hashes --no-header --universal \
  --python-version 3.10
"$UV" pip compile tools/scan-requirements.in -o tools/scan-requirements.txt --generate-hashes --no-header --universal \
  --python-version 3.10

# shellcheck shell=bash disable=SC2034  # sourced by the setup steps, which use its variables
# Shared by the setup steps. They run the same way inside WSL2 (as root, started by the Windows manager) and on
# native Linux (as your user; steps that install packages use sudo).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${1:-$(cd "$SETUP_DIR/.." && pwd)}           # engine checkout
IN_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IN_WSL=1
SUDO=""
if [ "$(id -u)" != 0 ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo"; else SUDO="NEEDS_PASSWORD"; fi
fi
CUDA_VER=12.8
CUDA_HOME_DEFAULT=/usr/local/cuda-$CUDA_VER

need_root() {
  if [ "$SUDO" = NEEDS_PASSWORD ]; then
    echo "This step installs system packages and needs sudo. Run it in a terminal:"
    echo "  sudo bash $SETUP_DIR/$(basename "$0") $R"
    exit 10
  fi
}

cuda_env() {
  if [ -f /etc/profile.d/cuda.sh ]; then source /etc/profile.d/cuda.sh; fi
  if [ -z "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]; then
    export CUDA_HOME=$CUDA_HOME_DEFAULT PATH=$CUDA_HOME_DEFAULT/bin:$PATH
  fi
}

gpu_cc() { nvidia-smi -i "${QWEN_GPU:-0}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' '; }

site_packages() { ls -d "$R"/.venv/lib/python3*/site-packages 2>/dev/null | head -1; }

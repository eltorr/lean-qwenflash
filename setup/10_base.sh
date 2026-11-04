#!/bin/bash
# Build tools and the CUDA 12.8 toolkit. Under WSL the Windows NVIDIA driver provides the GPU, so only the toolkit
# is installed (NVIDIA's WSL repository, which has no driver packages). On native Linux the NVIDIA driver must
# already be installed: it needs a reboot and, with Secure Boot, key enrollment, so this step only checks it.
source "$(dirname "$0")/lib.sh"
need_root

if ! nvidia-smi >/dev/null 2>&1; then
  if [ "$IN_WSL" = 1 ]; then
    echo "nvidia-smi does not work inside WSL. Install or update the NVIDIA driver on Windows, then run: wsl --shutdown"
  else
    echo "No working NVIDIA driver. Install one first, then reboot. On Ubuntu:"
    echo "  sudo ubuntu-drivers install     (or: sudo apt install nvidia-driver-570-open)"
  fi
  exit 11
fi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)"

. /etc/os-release
if ! command -v apt-get >/dev/null; then
  echo "This step automates Ubuntu and Debian ($PRETTY_NAME detected). Install these yourself, then continue:"
  echo "  build tools (gcc/g++, make), git, python3 + venv + headers, ninja, curl; CUDA toolkit $CUDA_VER"
  exit 12
fi
echo "installing build tools..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential git python3-venv python3-dev ninja-build wget curl bc >/dev/null

if [ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]; then
  echo "CUDA $CUDA_VER already installed"
else
  if [ "$IN_WSL" = 1 ]; then repo=wsl-ubuntu
  elif [ "$ID" = ubuntu ]; then repo=ubuntu${VERSION_ID//./}
  elif [ "$ID" = debian ]; then repo=debian${VERSION_ID%%.*}
  else echo "no CUDA repository mapping for $PRETTY_NAME; install CUDA $CUDA_VER manually"; exit 12; fi
  echo "installing CUDA $CUDA_VER toolkit from NVIDIA's $repo repository (several GB)..."
  tmp=$(mktemp -d)
  wget -q -O "$tmp/keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i "$tmp/keyring.deb" >/dev/null
  rm -rf "$tmp"
  $SUDO apt-get update -qq
  # the toolkit only: never pull a driver package (WSL must not have one; native keeps the one you installed)
  $SUDO apt-get install -y -qq "cuda-toolkit-${CUDA_VER/./-}" >/dev/null
fi

extra=""
[ "$IN_WSL" = 1 ] && extra=":/usr/lib/wsl/lib"
$SUDO tee /etc/profile.d/cuda.sh >/dev/null <<EOF
export CUDA_HOME=$CUDA_HOME_DEFAULT
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64$extra:\${LD_LIBRARY_PATH:-}
EOF
cuda_env
nvcc --version | tail -1
python3 --version

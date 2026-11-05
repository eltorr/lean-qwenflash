#!/bin/bash
# Prints key=value lines describing the engine side (read by the QwenFlash manager). File checks only: importing
# torch from a cold WSL page cache takes minutes, so nothing heavy is imported here.
R=${1:-$(cd "$(dirname "$0")/.." && pwd)}
c() { if bash -c "$2" >/dev/null 2>&1; then echo "$1=1"; else echo "$1=0"; fi; }
grep -qi microsoft /proc/version 2>/dev/null && echo in_wsl=1 || echo in_wsl=0
[ -d /run/systemd/system ] && echo systemd=1 || echo systemd=0
. /etc/os-release 2>/dev/null && echo "os=$PRETTY_NAME"
echo "kernel=$(uname -r)"
if [ "$(id -u)" = 0 ] || sudo -n true 2>/dev/null; then echo sudo=1; else echo sudo=0; fi
c driver "nvidia-smi -L"
echo "gpu=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
CC=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
echo "cc=$CC"
# the toolkit setup installs (CUDA_VER in lib.sh); another version counts as missing
CUDA_VER=$(sed -n "s/^CUDA_VER=\([0-9.]*\).*/\1/p" "$(dirname "$0")/lib.sh")
echo "cuda_needed=$CUDA_VER"
NVCC=/usr/local/cuda-$CUDA_VER/bin/nvcc
[ -x "$NVCC" ] || NVCC=""
if [ -n "$NVCC" ]; then echo "cuda_version=$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"; fi
# installed, and what run.sh loads (/etc/profile.d/cuda.sh, written by the setup step) names it
if [ -n "$NVCC" ] && grep -q "cuda-$CUDA_VER" /etc/profile.d/cuda.sh 2>/dev/null; then echo cuda=1; else echo cuda=0; fi
c engine "test -d $R/.git && test -x $R/run.sh && test -f $R/bin/qwenflash-backend && test -f $R/sizing.py"
if [ -d "$R/.git" ]; then
  echo "engine_head=$(git -C "$R" rev-parse HEAD 2>/dev/null)"
  [ -n "$(git -C "$R" status --porcelain --untracked-files=no 2>/dev/null)" ] && echo engine_dirty=1 || echo engine_dirty=0
fi
SP=$(ls -d "$R"/.venv/lib/python3*/site-packages 2>/dev/null | head -1)
c env "test -x $R/.venv/bin/python && test -f $SP/torch/version.py && grep -q +cu${CUDA_VER/./} $SP/torch/version.py && test -d $SP/fastapi"
[ -f "$SP/torch/version.py" ] && echo "torch=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py")"
BA=$(cat "$R/.qwenflash/build-arch" 2>/dev/null)
BT=$(cat "$R/.qwenflash/build-torch" 2>/dev/null)
TV=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py" 2>/dev/null)
echo "build_arch=$BA"
c build "test -f $R/qwen38-flash-offload/build/nv/qwen38_nv.so && test -f $R/qwen38-flash-offload/build/ec/qwen38_ec.so && test -f $R/qwen38-flash-offload/build/nv2/qwen38_nv2.so && { ls $SP/exllamav3_ext*.so || ls $R/exllamav3/exllamav3_ext*.so; } && { [ -z '$BA' ] || [ '$BA' = '$CC' ]; } && [ '$BT' = '$TV' ]"
model_ok() { test -f "$R/$1/config.json" && ! ls "$R/$1/"*.http-part >/dev/null 2>&1 && test -f "$R/store/$2.bin" && test -f "$R/store/$2.json"; }
export -f model_ok; export R
c m305 "model_ok model qwen"
c m405 "model_ok model-4.05 qwen405"
for d in model model-4.05; do [ -d "$R/$d" ] && echo "partial_${d//[.-]/}=$(du -sb "$R/$d" 2>/dev/null | cut -f1)"; done
echo "disk_free_gb=$(df -B1000000000 --output=avail "$R" 2>/dev/null | tail -1 | tr -d ' ')"
echo "fs=$(stat -f -c %T "$R" 2>/dev/null)"
echo "mem_total_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1e6 }' /proc/meminfo)"

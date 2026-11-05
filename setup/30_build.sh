#!/bin/bash
# Compile exllamav3 and the offload kernels for the GPU in this machine (its compute capability, e.g. 8.6 for an
# RTX 3090, 8.9 for a 4090, 12.0 for a 5090). Re-run after changing GPUs or PyTorch.
source "$(dirname "$0")/lib.sh"
cd "$R" || exit 1
cuda_env
source .venv/bin/activate
CC=$(gpu_cc)
[ -n "$CC" ] || { echo "no GPU visible (nvidia-smi)"; exit 11; }
case "$CC" in
  [0-6].*|7.0|7.2) echo "compute capability $CC is too old: this engine needs Turing (7.5) or newer"; exit 13 ;;
  7.5) echo "note: Turing (7.5) runs the EXL3 kernels' sm_75 fallbacks; this combination is untested" ;;
esac
MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="$CC" MAX_JOBS
SP=$(site_packages)
# A copy of exllamav3's module installed by an earlier build is found before the new one; built for another PyTorch,
# it fails to load. Remove it before anything imports exllamav3.
DEST="$SP/exllamav3_ext$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
rm -f "$DEST"
echo "building exllamav3 for sm_${CC/./} (a few minutes)..."
# its dependencies are pinned in setup/requirements.txt; build with this environment's tools, nothing downloaded
pip install -q --no-deps --no-build-isolation -e exllamav3
echo "compiling offload extensions..."
rm -rf qwen38-flash-offload/build
python compile_local.py
# exllamav3 JIT-builds its extension into torch's cache when the pip build above didn't leave one in exllamav3/, and
# then re-runs ninja's up-to-date check on every start (thousands of header stats; ~140 s on a cold WSL disk). Install
# that module, taken from this PyTorch's own cache folder only.
if ! ls exllamav3/exllamav3_ext*.so >/dev/null 2>&1; then
  CACHE=$(python -c 'from torch.utils.cpp_extension import _get_build_directory as d; print(d("exllamav3_ext", False))' \
          2>/dev/null || true)
  if [ -n "$CACHE" ] && [ -f "$CACHE/exllamav3_ext.so" ]; then
    cp "$CACHE/exllamav3_ext.so" "$DEST"
    echo "installed prebuilt exllamav3_ext into $SP"
  fi
fi
python -c 'import torch, exllamav3.ext as e; print("exllamav3 extension:", e.exllamav3_ext.__file__)'
mkdir -p .qwenflash
echo "$CC" > .qwenflash/build-arch
# the kernels match this PyTorch: a new one (setup/requirements.txt) means building again
sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py" > .qwenflash/build-torch
echo "built for sm_${CC/./}"

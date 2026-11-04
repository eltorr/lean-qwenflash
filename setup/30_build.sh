#!/bin/bash
# Compile exllamav3 and the offload kernels for the GPU in this machine (its compute capability, e.g. 8.6 for an
# RTX 3090, 8.9 for a 4090, 12.0 for a 5090). Re-run after changing GPUs.
source "$(dirname "$0")/lib.sh"
cd "$R"
cuda_env
source .venv/bin/activate
CC=$(gpu_cc)
[ -n "$CC" ] || { echo "no GPU visible (nvidia-smi)"; exit 11; }
case "$CC" in
  [0-6].*|7.0|7.2) echo "compute capability $CC is too old: this engine needs Turing (7.5) or newer"; exit 13 ;;
  7.5) echo "note: Turing (7.5) runs the EXL3 kernels' sm_75 fallbacks; this combination is untested" ;;
esac
export TORCH_CUDA_ARCH_LIST="$CC" MAX_JOBS=$(nproc)
echo "building exllamav3 for sm_${CC/./} (a few minutes)..."
pip install -q -e exllamav3
echo "compiling offload extensions..."
rm -rf qwen38-flash-offload/build
python compile_local.py
# exllamav3 JIT-builds its extension into torch's cache and, unless an importable copy exists, re-runs ninja's
# up-to-date check on every start (thousands of header stats; ~140 s on a cold WSL disk). Install the built module.
SP=$(site_packages)
SO=$(find ~/.cache/torch_extensions -path '*exllamav3_ext/exllamav3_ext.so' -newer exllamav3 2>/dev/null | head -1)
[ -n "$SO" ] || SO=$(find ~/.cache/torch_extensions -path '*exllamav3_ext/exllamav3_ext.so' | head -1)
if [ -n "$SO" ]; then
  cp "$SO" "$SP/exllamav3_ext$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
  echo "installed prebuilt exllamav3_ext into $SP"
fi
python -c 'import torch, exllamav3.ext as e; print("exllamav3 extension:", e.exllamav3_ext.__file__)'
mkdir -p .qwenflash
echo "$CC" > .qwenflash/build-arch
echo "built for sm_${CC/./}"

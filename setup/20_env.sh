#!/bin/bash
# Engine code and its Python environment.
#   20_env.sh ENGINE_DIR [SOURCE_REPO]
# With SOURCE_REPO (the Windows manager passes its own checkout), ENGINE_DIR becomes a clone of it that is updated
# to the source's current commit. Uncommitted changes in ENGINE_DIR are stashed first, never lost.
# Without it (native Linux), ENGINE_DIR is the checkout itself.
source "$(dirname "$0")/lib.sh"
SRC=${2:-}
BRANCH=qwenflash

if [ -n "$SRC" ] && [ "$(readlink -f "$SRC")" != "$(readlink -f "$R")" ]; then
  # git refuses repositories owned by another user (the Windows checkout, seen through /mnt) unless trusted
  git config --global --get-all safe.directory | grep -qxF "$SRC" || git config --global --add safe.directory "$SRC"
  want=$(git -C "$SRC" rev-parse "$BRANCH")
  if [ ! -d "$R/.git" ]; then
    echo "cloning the engine from $SRC into $R..."
    git clone -q --branch "$BRANCH" "$SRC" "$R"
  else
    cd "$R" || exit 1
    if [ "$(git rev-parse HEAD)" = "$want" ]; then
      echo "engine code is current (${want:0:7})"
    else
      if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
        git stash push -u -q -m "qwenflash setup: local changes before sync $(date '+%F %T')"
        echo "stashed local changes in $R (git stash list)"
      fi
      git remote get-url qwenflash >/dev/null 2>&1 || git remote add qwenflash "$SRC"
      git remote set-url qwenflash "$SRC"
      git fetch -q qwenflash "$BRANCH"
      git checkout -q -B "$BRANCH" FETCH_HEAD
      echo "engine code updated to $(git log --oneline -1)"
    fi
  fi
fi

cd "$R" || exit 1
cuda_env
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
# file checks only: importing torch from a cold WSL page cache takes minutes
have_env() {
  grep -q +cu12 "$1/torch/version.py" 2>/dev/null || return 1
  for m in fastapi uvicorn safetensors jinja2 numpy ninja; do [ -e "$1/$m" ] || return 1; done
}
SP=$(site_packages)
if [ -n "$SP" ] && have_env "$SP"; then
  echo "Python environment ready: torch $(sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py")"
  exit 0
fi
pip install -q --upgrade pip
echo "installing PyTorch (CUDA 12.8 build)..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q fastapi uvicorn safetensors jinja2 huggingface_hub numpy ninja
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0),
      "| sm_%d%d" % torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda"); (x @ x).sum().item(); print("GPU matmul OK")
PY

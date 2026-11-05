"""Paths, settings and their defaults.

Settings and logs live outside the checkout, so updating or resetting the repo never loses them:
  Windows  %LOCALAPPDATA%\\QwenFlash
  Linux    $XDG_STATE_HOME/qwenflash (default ~/.local/state/qwenflash)
QWENFLASH_HOME overrides both.
"""
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GB = 1e9
GIB = 1024 ** 3
SERVED_NAME = "qwen3.8-flash-next"
IS_WINDOWS = sys.platform == "win32"


def state_dir():
    if os.environ.get("QWENFLASH_HOME"):
        d = Path(os.environ["QWENFLASH_HOME"])
    elif IS_WINDOWS:
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "QwenFlash"
    else:
        d = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "qwenflash"
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


STATE = state_dir()
CONFIG_PATH = STATE / "config.json"
LOG_DIR = STATE / "logs"

DEFAULTS = {
    # where the engine lives. Windows: inside this WSL distro; Linux: this checkout ("" = the repo itself)
    "distro": "Ubuntu-24.04",
    "wsl_location": "",                 # where a new distro is installed ("" = suggest the drive with most space)
    "repo_dir": "",                     # engine checkout as the engine sees it ("" = platform default)
    "quant": "3.05",
    "context": 262144,
    "listen_host": "127.0.0.1",
    "listen_port": 30000,
    "backend_port": 30001,
    "dashboard_token": "",              # required by the dashboard/control API when listen_host is not loopback
    "auto_switch": True,
    "wake_on_request": True,
    "preload_on_launch": False,
    "preload_when_free": False,
    "upgrade_when_idle": True,
    "idle_unload_min": 0,
    "wake_override_min": 5,
    "pressure_response": "lean",        # lean | pause
    "gpu_busy_action": "lean",          # lean | pause
    "ext_util_pct": 50,
    "ext_vram_lean_gb": 0.0,            # other apps' VRAM that sends the model to lean (0 = a quarter of the card)
    "vram_min_free_gb": 0.8,
    "ram_lean_gb": 6.0,
    "ram_pause_gb": 3.0,
    "ram_reserve_gb": 6.0,              # RAM always left to the host OS and other apps
    "lean_vram_gb": 0.0,                # VRAM the model keeps when other apps need the GPU (0 = automatic)
    "lean_ram_gb": 8,
    "full_vram_headroom_gb": 0.0,       # VRAM left free in full mode (0 = ~10% of the card, 1.2-3.2 GB)
    "max_ram_cache_gib": 0,             # 0 = automatic (enough to hold every expert that is not in VRAM)
    "auto_recover": True,               # restart a crashed model, stepping its VRAM budget down after an OOM
    "watchlist": [],
    "util_ignore": ["dwm.exe", "csrss.exe", "system", "explorer.exe", "xorg", "gnome-shell", "kwin_x11",
                    "kwin_wayland"],
    "moderate_after_s": 10,
    "severe_after_s": 3,
    "recover_after_s": 30,
    "min_dwell_s": 60,
    "severe_max_wait_s": 120,
    "busy_wait_s": 90,
    "start_timeout_s": 600,
    "poll_s": 1.5,
    "ui_browser": "",
    # Optional Ollama-compatible front door, for replacing an Ollama install (docs/manager.md): set ollama_alias_port
    # to 11434 and QwenFlash answers on Ollama's usual port, passing anything it doesn't serve to an Ollama moved
    # to ollama_port. 0 (the default) leaves it off.
    "ollama_port": 11435,
    "ollama_alias_port": 0,
    "qwenflash_aliases": [],
}
# settings that only take effect when the service restarts
SERVICE_KEYS = {"listen_port", "backend_port", "listen_host", "repo_dir", "distro", "ollama_port", "ollama_alias_port"}
RENAMED = {"windows_ram_reserve_gb": "ram_reserve_gb"}

logging.basicConfig(filename=LOG_DIR / "manager.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwenflash")


def migrate_legacy():
    """First run of the integrated manager: adopt settings from the old standalone switcher next to the repo."""
    old = REPO / "config.json"
    if CONFIG_PATH.exists() or not old.exists():
        return
    try:
        data = json.loads(old.read_text())
    except ValueError:
        return
    for a, b in RENAMED.items():
        if a in data:
            data[b] = data.pop(a)
    # the old switcher hard-coded 20 GB as the minimum for full; automatic sizing replaces it
    data.pop("full_min_vram_gb", None)
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
    for name in ("switcher.log",):
        src = REPO / "logs" / name
        if src.exists():
            shutil.copy2(src, LOG_DIR / "switcher-legacy.log")
    log.info("migrated settings from %s", old)


def load_config():
    migrate_legacy()
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            for a, b in RENAMED.items():
                if a in data and b not in data:
                    data[b] = data.pop(a)
            cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
        except Exception as e:  # noqa: BLE001 - a bad config must not stop the service
            log.warning("config.json unreadable (%s); using defaults", e)
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, CONFIG_PATH)


def coerce(key, value):
    d = DEFAULTS[key]
    if isinstance(d, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
    if isinstance(d, int):
        return int(float(value))
    if isinstance(d, float):
        return float(value)
    if isinstance(d, list):
        if isinstance(value, str):
            value = [x.strip() for x in re.split(r"[,\n]", value)]
        return [str(x).strip() for x in value if str(x).strip()]
    return str(value)


def load_json(name, default):
    p = STATE / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def save_json(name, data):
    p = STATE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)

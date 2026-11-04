# QwenFlash manager

The manager runs next to the engine and makes it a service:

- **One endpoint:** an OpenAI-compatible API on `http://127.0.0.1:30000/v1`, plus an Ollama-style front door on
  port 11434. See [api.md](api.md).
- **The model loads when called and makes room for other apps.** It sizes the model to the free VRAM and RAM,
  scales it back or pauses it when games or other GPU apps need the card, and re-saturates when they're done.
- **A local web dashboard** to watch and control all of it: state, speed, GPU/RAM use, setup and logs.
- **Setup for both platforms:** native Linux, or Windows with the engine inside WSL2. It detects which, checks the
  hardware, and runs each step.

The manager itself is plain Python (only `psutil`, plus `pywin32` on Windows) and never touches the GPU. The model
server it controls is the engine in this repo, started through `run.sh`.

## Start it

| | Windows (engine in WSL2) | Linux |
|---|---|---|
| Start the service and open the dashboard | double-click `QwenFlash.cmd` (or the desktop shortcut) | `./qwenflash.sh open` |
| Run in the foreground | `.venv\Scripts\python -m qwenflash serve` | `./qwenflash.sh` |
| Start at login | Settings: *Start the service when I sign in to Windows* | Settings: *Start at login* (a systemd user service) |

The first start creates the manager's small Python environment (`.venv` on Windows, `.venv-manager` on Linux).
Then open the **Setup** tab. It shows what was detected and recommends settings for the machine. Run the steps
top to bottom; each one can be re-run safely.

The same is available from a terminal, which helps on headless servers:

```
qwenflash detect          platform, GPU (VRAM, architecture, PCIe link), CPU, RAM, WSL, engine, recommendations
qwenflash setup           list the setup steps and their status
qwenflash setup build     run one step here, with its output
qwenflash plan            this machine's preset, and the VRAM/RAM plan at each quant (sizing.py)
qwenflash status          what the running service is doing (add --json for everything)
```

On Windows, run these as `.venv\Scripts\python -m qwenflash ...`; on Linux, `./qwenflash.sh ...`.

## Setup

| Step | Windows + WSL2 | Linux |
|---|---|---|
| WSL 2 | installs or updates WSL (asks for administrator approval; a first install needs a restart) | - |
| WSL memory limit | writes `%USERPROFILE%\.wslconfig` so WSL may use most of the RAM (its default is half); keeps your other settings | - |
| Ubuntu 24.04 | installs the distro on the drive you choose (default: the fixed drive with most free space), with systemd | - |
| NVIDIA driver | - (the Windows driver serves WSL) | checked only: installing one needs a reboot, so it shows the commands |
| Build tools + CUDA 12.8 | inside WSL, from NVIDIA's WSL repository | from NVIDIA's repository for your Ubuntu/Debian release; uses `sudo` (if it needs a password, the step prints the command to run) |
| Engine + Python environment | makes the engine directory inside WSL a clone of this checkout and updates it to this checkout's commit (local changes there are stashed, never lost) | uses this checkout |
| Compile GPU kernels | for the GPU's own architecture (sm_75 Turing to sm_120 Blackwell) | same |
| Models | 3.05 bpw (~85 GB download + ~47 GB store) and/or 4.05 bpw (~107 GB + ~62 GB); resumable | same |

**Engine sync (Windows):** the manager runs from this checkout on Windows, while the engine runs from a clone inside
WSL.
- The header shows both commits.
- After you commit a change here, *Engine + Python environment* shows **Update available**; running it brings the
  WSL clone to the same commit.
- Only committed code is synced.

**Recommendations:** the Setup tab compares the machine with what the model needs. For example:
- the baseline for this card, RAM and platform: quant, context, RAM cache and the manager's RAM thresholds (the
  [presets](../README.md#presets-a-baseline-from-vram-and-ram)). A new install starts from it; later it is offered
  when your settings differ
- the context window that fits this GPU
- whether every expert fits in VRAM + RAM, or some will be read from NVMe
- a PCIe link below Gen4 x16
- WSL limited to half the RAM
- a filesystem without O_DIRECT
- several GPUs (the engine uses GPU 0; see [research/multi-gpu.md](research/multi-gpu.md))

Recommendations that suggest a setting have an *Apply* button.

## How it sizes and switches the model

The model's VRAM use has two parts:
- **A fixed part:** about 6.2 GB (3.05 bpw) or 6.6 GB (4.05) plus the KV cache: 9.0-9.4 GB at 256K context.
- **An expert cache** that takes whatever budget is left.

[`sizing.py`](../sizing.py) turns a VRAM budget into a plan: cache size, prefill chunk (smaller on small cards),
and whether prefill staging fits. The manager and `run.sh` both use it.

| Profile | VRAM budget | RAM cache | When |
|---|---|---|---|
| **Full** | everything free, minus headroom (~10% of the card, 1.2-3.2 GB; 0.6 GB minimum on a card without a display) | enough for every expert that isn't in VRAM, shrunk to what the host can spare | nothing else needs the GPU |
| **Lean** | about half the card (16 GB on a 32 GB card), never less than the context needs | `lean_ram_gb` (8 GiB) | other apps need VRAM or RAM (moderate pressure) |
| **Paused** | 0 | 0 | a watched program runs, or VRAM/RAM is critically short (heavy pressure) |

**Pressure:**
- **VRAM:** other apps use more than a quarter of the card, or grow into the model's free memory.
- **RAM:** free RAM drops below `ram_lean_gb`.
- **GPU busy:** another app keeps the GPU at or above `ext_util_pct`.

Thresholds scale with the card. Each is a setting, where 0 means automatic.

**Switching:**
- **Scaling back** waits ~10 s of sustained pressure and happens between requests.
- **Pausing** happens ~3 s into heavy pressure, after the request in progress (or after 2 minutes at most).
- **Re-saturating:** a paused model wakes on the next request. A lean model returns to full once the GPU has been
  clear for 30 s, between requests, even with an agent sending requests non-stop.
- **Your choice wins:**
  - *Start lean* from the dashboard stays lean until you start full.
  - Calling the model under pressure wakes it lean and holds off auto-switching for 5 minutes.

### Robustness

- **The model survives manager restarts.** The engine runs detached: as the transient systemd service
  `qwenflash-backend` where systemd runs (WSL, most Linux machines), otherwise in its own session. A new manager
  adopts it instead of restarting it, and *Quit, keep model* leaves it running on purpose.
- **Crashes recover.** The manager notices a dead server within ~10 s and restarts it after 10 s, 30 s and 90 s.
  After three failures in 10 minutes it stops and shows the error.
- **It learns from out-of-memory crashes.** If the server ran out of VRAM, the next start uses a budget 2 GB
  smaller. If it ran out of RAM, the RAM cache shrinks by 4 GiB. These learned limits apply per GPU, quant and
  context; *Settings > Forget learned limits* clears them.
- **It measures RAM use.** 90 s after each start, the manager measures the RAM the model uses beyond its expert cache
  and sizes later starts with that figure. RAM dips in the first 90 s after a start (page cache being handed back,
  slow under WSL) don't count as pressure.

### Speed

The dashboard shows the generation speed, prompt speed and time to first token of recent requests, measured as
they pass through the manager. `GET /_switcher/api/metrics` lists the last 100.

## Where things are

| Path | What |
|---|---|
| `qwenflash/` | the manager: `service.py` (lifecycle, API), `host.py` (Windows+WSL2 / Linux), `hardware.py`, `setup.py`, `nvml.py`, `ui/` |
| `sizing.py` | the VRAM/RAM model shared by the manager, `run.sh` and the engine |
| `run.sh` | launches the engine on any NVIDIA GPU, sized from the hardware unless told otherwise |
| `bin/qwenflash-backend` | starts/stops the engine detached; its log is `.qwenflash/backend.log` in the engine checkout |
| `setup/` | the Linux-side steps, shared by both platforms |
| `bench/` | `vram_ballast.py` (hold VRAM: emulate a smaller card, or play "another app") and `vram_matrix.py` |
| settings and logs | Windows `%LOCALAPPDATA%\QwenFlash`, Linux `~/.local/state/qwenflash` (`config.json`, `learned.json`, `logs/`) |

## Remote dashboards

The dashboard and control API listen on `127.0.0.1` only. If you set `listen_host` to another address (for
example, a headless Linux server), the manager creates a `dashboard_token` in `config.json`; open the dashboard
once with `?token=<token>` and the browser remembers it. The OpenAI API itself does not check keys, like Ollama.
Over SSH, a tunnel (`ssh -L 30000:127.0.0.1:30000 host`) keeps everything on loopback instead.

## Replacing Ollama on a tailnet (Windows example)

This PC serves the tailnet in place of Ollama.

- **Address:** QwenFlash answers on Ollama's port, 11434, through its front door. Other devices keep using
  `http://<tailnet-host>:11434/v1`.
- **Exposure:** `tailscale serve --bg --tcp=11434 tcp://127.0.0.1:11434`. Check it with `tailscale serve status`;
  turn it off with `tailscale serve --tcp=11434 off`.
  - It uses TCP mode because HTTP mode answers only requests whose Host header is the machine's own name, and
    returns 404 for the raw tailnet IP or a proxy that rewrites the hostname.
- **Old clients keep working:** chat requests for any model name are answered by QwenFlash, including Ollama
  names such as `qwen3.8`.
- **Ollama-only features return 404:** embeddings, images, Ollama's `/api/*`, and Anthropic `/v1/messages`.
  `GET /` answers `QwenFlash is running` for health checks.
- **Running Ollama alongside:** start it with `OLLAMA_HOST=127.0.0.1:11435`. The front door then passes
  embeddings, `/api/*` and requests for its installed models through to it.
- **The dashboard never leaves the machine:** the front door doesn't serve it or the control API.

# QwenFlash manager

The manager runs the model as a service on a PC that is also used for other things, such as gaming, and adapts the
model's resources to them:

- **One endpoint:** an OpenAI-compatible API on `http://127.0.0.1:30000/v1` ([api.md](api.md)).
- **Shares the GPU:**
  - loads the model when a client calls it;
  - sizes it to the free VRAM and RAM;
  - scales it back or pauses it when games or other GPU apps need the card;
  - comes back to full when they're done.
- **A local dashboard** shows the state, speed, GPU and RAM use, setup and logs.
- **Setup** for native Linux, or Windows with the engine in WSL2.

The manager is plain Python (`psutil`, plus `pywin32` on Windows) and never touches the GPU itself.

## Start it

| | Windows (engine in WSL2) | Linux |
|---|---|---|
| Start and open the dashboard | double-click `QwenFlash.cmd` | `./qwenflash.sh open` |
| Run in the foreground | `.venv\Scripts\python -m qwenflash serve` | `./qwenflash.sh` |
| Start at login | Settings: *Start the service when I sign in to Windows* | Settings: *Start at login* |

From a terminal, which helps on headless machines:

```
qwenflash detect          platform, GPU, CPU, RAM, WSL, engine and recommendations
qwenflash setup           the setup steps and their status
qwenflash setup STEP      run one step: wsl, wslconfig, distro, driver, base, engine, build, m305, m405
qwenflash plan            this machine's preset, and what fits at each quant
qwenflash status          what the service is doing (--json for everything)
```

On Windows, run these as `.venv\Scripts\python -m qwenflash ...`; on Linux, as `./qwenflash.sh ...`.

## Setup

| Step | Windows + WSL2 | Linux |
|---|---|---|
| WSL 2 | installs WSL; needs administrator approval, and a first install needs a restart | - |
| WSL memory limit | lets WSL use most of the RAM (`.wslconfig`) | - |
| Ubuntu 24.04 | installs the distro on the drive you pick, with systemd | - |
| NVIDIA driver | - | checks it, and shows the install commands if it's missing |
| Build tools + CUDA 12.8 | inside WSL | uses `sudo`, or prints the command to run |
| Engine + Python environment | a clone of this checkout inside WSL, at the same commit | this checkout |
| Compile GPU kernels | for the local GPU (Turing to Blackwell) | same |
| Models | 3.05 bpw (~85 GB download + 47 GB store) or 4.05 bpw (~107 + 62 GB); downloads resume | same |

On Windows, when you commit a change here, *Engine + Python environment* shows **Update available**. Running it
again syncs the WSL clone; only committed code is synced.

**Model versions:**
- **Source:** [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) on
  Hugging Face.
- **Pinned:** each quant is pinned to one upload (`PINNED` in `download_http.py`), so every install gets the same
  files, even after newer uploads appear. A newer upload is only used once it has been tested and the pin changed.
- **Changing the pin:** running a model step again skips files that are already there. So after a pin change,
  delete that model's folder and its store file (`store/qwen*.bin`, `.json`), then run the step again.

## How it shares the GPU

| Profile | VRAM | RAM cache | When |
|---|---|---|---|
| **Full** | all that's free, minus ~10% headroom | every expert that isn't in VRAM, as far as RAM allows | nothing else needs the GPU |
| **Lean** | about half the card | 8 GiB | a game or another app needs VRAM or RAM |
| **Paused** | 0 | 0 | a watched program runs, or VRAM or RAM is critically short |

- **Scaling back:** happens between requests, after about 10 s of pressure.
- **Pausing:** happens about 3 s into heavy pressure, once the current request finishes (2 minutes at most).
- **Coming back:**
  - A paused model wakes on the next request.
  - A lean model returns to full after the GPU has been clear for 30 s.
- **Your choice wins:** *Start lean* stays lean until you start full. A request made under pressure wakes the model
  lean and holds off auto-switching for 5 minutes.
- **Thresholds:** all are in Settings, scaled to the card; 0 means automatic.

[`sizing.py`](../sizing.py) turns a VRAM budget into a plan: the expert cache, the prefill chunk, and whether
staging fits.

**Recovery:**

- **Model kept running:** the model runs detached, so restarting the manager doesn't stop it.
- **Crashes:** a crashed model is restarted after 10 s, 30 s and 90 s. After three failures in 10 minutes, the
  manager stops and shows the error.
- **Out of memory:**
  - VRAM: the next start uses 2 GB less.
  - RAM: the RAM cache shrinks by 4 GiB.
  - *Settings > Forget learned limits* clears these.

## Where things are

| Path | What |
|---|---|
| `qwenflash/` | the manager: `service.py`, `host.py` (Windows + WSL2 or Linux), `hardware.py`, `setup.py`, `ui/` |
| `sizing.py` | the VRAM and RAM model, shared by the manager and `run.sh` |
| `run.sh` | starts the engine, sized from the hardware |
| `bin/qwenflash-backend` | runs the engine detached (a systemd unit where available) |
| `setup/` | the install steps, shared by both platforms |
| settings and logs | Windows `%LOCALAPPDATA%\QwenFlash`, Linux `~/.local/state/qwenflash` |

## Other machines

The dashboard and API listen on `127.0.0.1` only.

- **Dashboard over SSH:** `ssh -L 30000:127.0.0.1:30000 host`.
- **API for other devices:**
  1. Turn on the API-only port: set `"ollama_alias_port": 11434` in `config.json` and restart the service. This
     port serves the OpenAI API but never the dashboard.
  2. Expose the port, for example on a tailnet:

     ```
     tailscale serve --bg --tcp=11434 tcp://127.0.0.1:11434
     ```

  3. Clients use `http://<tailnet-host>:11434/v1`.
- **Dashboard without SSH:** set `listen_host` in `config.json`. This needs the `dashboard_token` the manager then
  creates, given once as `?token=...`.

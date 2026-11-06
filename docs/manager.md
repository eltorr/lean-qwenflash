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
qwenflash bench           benchmark this machine (see Benchmarks)
```

On Windows, run these as `.venv\Scripts\python -m qwenflash ...`; on Linux, as `./qwenflash.sh ...`.

## Setup

| Step | Windows + WSL2 | Linux |
|---|---|---|
| WSL 2 | installs WSL; needs administrator approval, and a first install needs a restart | - |
| WSL memory limit | lets WSL use most of the RAM (`.wslconfig`) | - |
| Ubuntu 24.04 | installs the distro on the drive you pick, with systemd | - |
| NVIDIA driver | - | checks for driver 595 or newer (CUDA 13.2), and otherwise shows the install commands |
| Build tools + CUDA 13.2 | inside WSL | Ubuntu 22.04+ or Debian 12+; uses `sudo`, or prints the command to run |
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
| **Full** | all that's free, minus ~10% headroom (at most 2.4 GB) | with the VRAM cache, 90% of the experts, as far as RAM allows | nothing else needs the GPU |
| **Lean** | about half the card | 8 GiB | a game or another app needs VRAM or RAM |
| **Paused** | 0 | 0 | a watched program runs, or VRAM or RAM is critically short |

- **Scaling back:** happens between requests, after about 10 s of pressure.
- **Pausing:** happens about 3 s into heavy pressure, once the current request finishes (2 minutes at most).
- **Coming back:**
  - A paused model wakes on the next request.
  - A lean model returns to full after the GPU has been clear for 30 s.
- **Your choice wins:** *Start lean* stays lean until you start full. A request made under pressure wakes the model
  lean and holds off auto-switching for 5 minutes.
- **The RAM cache:**
  - It holds 90% of the experts together with the VRAM cache. The rest are rarely picked: at 4.05 bpw on a 32 GB
    card, 34 and 40 GiB ran at the same speed, and the last 6 GiB only cost the PC RAM.
  - A restart reuses the RAM the stopped model held. Under WSL2, Windows only sees freed memory again after a few
    minutes, and sizing from its figure gave restarts a third of the cache.
  - If full runs short of RAM and drops to lean, the next full starts get a smaller RAM cache for an hour, so
    full -> lean -> full doesn't repeat.
  - A full model whose RAM cache was sized while RAM was short grows it, with a restart, once 8 GiB more fits.
- **Short prompts** (up to 1,638 tokens, or each new turn of a conversation that adds no more) skip prefill
  staging, which costs about 5.5 s per prompt whatever its size.
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

## Benchmarks

The **Benchmark** tab and `qwenflash bench` measure this machine: they start the model in each configuration, time
answers and prompt reading, and record the results. The model stops for the run, requests get an error meanwhile,
and the model comes back afterwards. Only the PC itself can start one.

```
qwenflash bench                      this machine's settings on the whole card
qwenflash bench quants               3.05 and 4.05 bpw
qwenflash bench lean                 the lean profile
qwenflash bench 4.05:22:262144       4.05 bpw held to 22 GB of VRAM (a 24 GB card), 256K context
qwenflash bench matrix               8-32 GB cards emulated on a 32 GB one: the README's measured table (~1 h;
                                     needs 64 GB of RAM: every run gets the same 38 GiB RAM cache)
  --workload quick                   ~2 min a run instead of 3-6
  --out research/NAME/results        also write the results there
qwenflash bench --resume             continue the last benchmark that was cut short
qwenflash bench --list | --table FOLDER | --sizing FOLDER
```

- **What a run measures,** the way the original's `benchmark.py` does, so results compare with MaxDevv's:
  - thinking off and greedy decoding, with his prompts; every answer ends on its own (the output is never capped);
  - a warm-up, then three trials of one answer (a jet engine in three paragraphs, about 250 tokens) and documents of
    1K, 8K, 32K and 64K tokens, each read and answered in one sentence. Each trial's text differs from the start, so
    nothing comes from the prompt cache;
  - his retrieval test: three access codes at 10%, 50% and 90% of a document of up to 100K tokens;
  - startup time, VRAM, and where the experts came from (VRAM, RAM or NVMe).
- **The PC comes first.** When a game or another app needs the GPU (an app on the watchlist, another app's GPU load,
  or its VRAM crowding the card) or RAM nearly runs out, the run steps aside: its server stops, and the run starts
  again from the beginning once the PC has been free for `recover_after_s`. A run interrupted four times is recorded
  as interrupted, and the benchmark moves on.
- **Resuming:** a benchmark cut short (cancelled, the manager restarted, the PC rebooted) continues with
  `qwenflash bench --resume` or the tab's **Resume** button. The runs left go into the same folder, with the same
  workload.
- **A run spec** is `QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB]`. A VRAM figure holds the engine to that much, as on
  a smaller card. It can't emulate a smaller card's bandwidth, so real smaller cards are slower.
- **Results:** one JSON file per run in `bench/` in the settings folder, saved after every step, with the machine and its stack (GPU, driver,
  CUDA, PyTorch, commit). Each run shows the change since the last run of the same configuration, which makes a quick
  before-and-after check of a change.
- Without the service running, `qwenflash bench` runs in the terminal, and leaves the model stopped.

The measured runs behind the README are listed in [research/](../research/README.md).

## Where things are

| Path | What |
|---|---|
| `qwenflash/` | the manager: `service.py`, `host.py` (Windows + WSL2 or Linux), `hardware.py`, `setup.py`, `bench.py`, `ui/` |
| `sizing.py` | the VRAM and RAM model, shared by the manager and `run.sh` |
| `run.sh` | starts the engine, sized from the hardware |
| `bin/qwenflash-backend` | runs the engine detached (a systemd unit where available) |
| `setup/` | the install steps, shared by both platforms; `CUDA_VER` and `DRIVER_MIN` in `lib.sh` |
| `tools/` | the checks (`scan.sh`), the package locks (`lock.sh`) and the pre-push hook |
| settings and logs | Windows `%LOCALAPPDATA%\QwenFlash`, Linux `~/.local/state/qwenflash` |

## Security

- **Dashboard:** every port needs the dashboard token, from the private link's cookie or a Bearer header. It's
  `dashboard_token` in `config.json`.
- **Other websites:** browser requests from other sites, and DNS rebinding, are refused
  ([details](api.md#control-api-the-dashboards-not-openai)).
- **Settings:** paths, programs, ports and the token change only in `config.json`. Other devices can't run setup,
  quit, change autostart, make a new link or change the PC's name in it.
- **Model server:** it listens on `127.0.0.1` and answers only requests that carry the manager's key (`backend.key`
  in the settings folder).
- **Files:** the settings folder is private to your user. On Windows, setup limits the WSL folder to you, SYSTEM and
  Administrators.
- **Packages and downloads:**
  - every Python package is pinned by hash;
  - NVIDIA's repository key is checked before it's trusted;
  - model files are checked against Hugging Face's hashes.
- **The API (`/v1`) takes no key,** like a local OpenAI server: anything that reaches the port can use the model.
  Only expose the network port, on a private network such as a tailnet.

## Other machines

Every port uses the same layout: the API is under `/v1`, and the web page is at `/`. Port 30000 listens on
`127.0.0.1` only. To reach the PC from your phone and other devices, use the network port:

1. **Turn it on:** set `"network_port"` in `config.json` to any free port, such as `8080`, then restart the service.
2. **Expose it,** for example on a tailnet:

   ```
   tailscale serve --bg --tcp=8080 tcp://127.0.0.1:8080
   ```

3. **API:** clients use `http://<tailnet-host>:8080/v1`.
4. **Dashboard:**
   1. On the PC, copy the private link from **Settings > Other devices**.
   2. Open it once on each device; the device then remembers it.
   3. **New link** makes the old one stop working.

   Without the link, the port's root shows a page where you can paste the link, and the dashboard's controls
   refuse requests.

Don't expose port 30000 itself; it only answers requests addressed to `127.0.0.1` or `localhost`. On the PC, the
dashboard needs the same link: `QwenFlash.cmd` and `qwenflash open` sign the browser in.

Over SSH, `ssh -L 30000:127.0.0.1:30000 host` reaches everything instead.

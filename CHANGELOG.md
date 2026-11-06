# Changelog

What changed in this fork, newest first. [FORK.md](FORK.md) describes how it differs from the original.

## 2026-11-06

### Tuning

From the manager's log and the new benchmark ([report](research/vram-scaling-torch-2.15/REPORT.md#follow-up-2026-11-06-short-prompts-skip-staging)):

- **Short prompts skip staging.** Prompts and conversation turns of up to 1,638 tokens no longer go through prefill
  staging, which read every expert not in VRAM from the NVMe each time and didn't keep it. They fill the RAM cache
  instead, so longer prompts after them read from RAM. At 4.05 bpw on the whole card:
  - a 1-2K-token prompt gets its first token in 1.4-2.6 s instead of 6-7;
  - documents read 1.2-5 times faster (8K: 1,129 to 2,546 tok/s; 100K: 4,364 to 5,312);
  - answers are as fast as before.
- **A steadier RAM cache.** The log showed full -> lean -> full every few minutes, and restarts that started small and
  restarted again to grow:
  - the RAM cache holds 90% of the experts together with the VRAM cache: 33-35 GiB instead of 40 at 4.05 on a 32 GB
    card, at the same speed. Free RAM with the model at full went from 3-5 GB to about 12;
  - a restart reuses the RAM the stopped model held, which WSL hands back to Windows only gradually. It used to start
    with 8-19 GiB and restart again to grow;
  - after full runs short of RAM, full starts get a smaller RAM cache for an hour;
  - the estimate of the model's RAM use beyond its cache rises fast and falls slowly.
- **Less idle VRAM:** full mode leaves at most 2.4 GB free, down from 3.2 GB on 32 GB cards. It ran stably for days with
  about 2.1 GB free, before the engine counted the VRAM CUDA can't see under WSL2.
- **Benchmarks** size the RAM cache the same way, and a model restarted after one gets its RAM cache back.

## 2026-11-05

### Security

From two security reviews of the fork's code.

- **Dashboard:** it needs its token on every port, not just the network port. `QwenFlash.cmd` and `qwenflash open`
  sign the browser in, and the CLI sends the token.
- **Other websites:**
  - a browser request from another site gets 403 on every port;
  - port 30000 only answers requests addressed to `127.0.0.1`, `localhost` or `::1`;
  - pages on the network port must come from one of the PC's names (DNS rebinding);
  - pages can't be framed.
- **Requests:**
  - strict parsing: no bare CR, LF or NUL in headers, and no conflicting lengths;
  - body limits (1 MB for the dashboard, 64 MB for `/v1`) and timeouts;
  - the completion routes take POST only, so a link or an image can't wake the model.
- **Settings:**
  - paths, programs, ports and the token change only in `config.json`;
  - numbers are range-checked, and a change is checked in full before any of it applies;
  - other devices can't run setup, quit, change autostart, make a new link or rename the PC.
- **Model server:**
  - it listens on `127.0.0.1` and answers only the manager, which passes it a key;
  - the token and the key never reach its logs;
  - it never runs code from a model folder.
- **Windows:** programs run by full path, never from the current folder. New WSL distros get a folder that only you,
  SYSTEM and Administrators can open.
- **Packages and downloads:**
  - every Python package (engine, manager and scanners) is pinned by hash;
  - NVIDIA's repository key is checked before it's trusted;
  - model files are checked against Hugging Face's hashes.
- **B70 tier:** its shared ring file is owner-only; it was world-writable. This fork doesn't use the tier.

### Checks

- **`bash tools/scan.sh`:** runs ruff, shellcheck, a secret scan, pip-audit on the pinned packages, and the unit tests.
- **GitHub:** runs the same scan on every push, plus CodeQL; Dependabot proposes updates.
- **`tools/pre-push`:** runs the scan before each push.

### Benchmarks

- **One tool,** from the dashboard's new Benchmark tab or `qwenflash bench`. It replaces `bench/vram_matrix.py` and
  `bench/vram_table.py`:
  - it runs any configuration, or a preset: this machine's settings, the lean profile, both quants, or 8-32 GB cards
    emulated on a 32 GB one;
  - the manager frees the GPU for it and puts the model back afterwards;
  - each result records the machine and its stack, and the change since the last run of the same configuration;
  - it measures the way the original's `benchmark.py` does, so results compare with MaxDevv's: thinking off, greedy
    decoding, his prompts and filler documents (1K-64K tokens), and his retrieval test. Answers end on their own after
    about 250 tokens, so a run takes 3-6 minutes and the matrix about an hour. A first try with thinking on and
    open-ended prompts wrote 12,000-55,000 tokens an answer and would have taken over a day;
  - every matrix run gets the same 38 GiB RAM cache, as the earlier matrix. Sized from the RAM free when each run
    started, it varied from 8 to 25 GiB with how much memory WSL had handed back from the run before.
- **The PC comes first:** when a game or another app needs the GPU, or RAM nearly runs out, the run steps aside and
  starts again once the PC is free. A benchmark cut short by a cancel, a restart or a reboot resumes where it stopped
  (`qwenflash bench --resume`, or the tab's Resume button).
- **The dashboard says Benchmarking** while one runs, not Paused, and the start and pause buttons are off until it ends.
- **Output is never capped:** the old runner cut answers at 400-1,500 tokens.
- **Run records:** [research/README.md](research/README.md) lists every benchmark with its date, machine and stack.
- **The README's tables, measured again** on PyTorch 2.15 with the new method
  ([report](research/vram-scaling-torch-2.15/REPORT.md)):
  - 3.05 bpw decodes 74 tok/s on 30 GB of VRAM, down to 20 on 7 GB; 4.05 decodes 63 down to 30 tok/s;
  - an 8 GB card that drives a display now runs (20 tok/s here), where it ran out of VRAM on PyTorch 2.11;
  - a staged prompt cost about 5.5 s whatever its size (the NVMe, it turned out; fixed on 2026-11-06).

### Engine

- **PyTorch 2.15.0 on CUDA 13.2,** up from 2.11 on CUDA 12.8 ([report](research/torch-2.15/REPORT.md)):
  - prefill is 47-72% faster at 8K-64K tokens, and decode is the same;
  - the model starts in about 34 s instead of 50;
  - it needs NVIDIA driver 595 or newer;
  - pip-audit no longer finds any known advisories in the pinned packages.
- **The upgrade itself:**
  - the offload kernels build as C++20, which PyTorch 2.15 needs;
  - a rebuild removes exllamav3 modules made for another PyTorch;
  - setup moves to the new CUDA toolkit on upgrade.

### Fixes

- **Clean installs on CUDA 13:** the engine looked for CUDA 12's runtime (`libcudart.so.12`) when it set up pinned
  memory. The fork's machine only worked because the 2.11 install had left it behind; a fresh install has only CUDA
  13's, so the engine would not start. It now uses the runtime PyTorch loaded. On upgrade, setup removes the CUDA
  packages an older PyTorch left behind, then reinstalls the current ones, because some share folders with them.
- **The VRAM limit under WSL2:** CUDA there doesn't see part of what Windows apps use (about 1 GB with a desktop and a
  browser open). The limit counts everything, so the expert cache overshot it by that much:
  - emulated cards of 7-16 GB stopped at startup;
  - a lean profile used about 1 GB more than its budget alongside a game.

  The engine now measures the difference at startup and counts it.
- **Stopping the model:** under very low RAM, a stop could fail silently, leaving the model running while the
  dashboard showed it stopped. The manager now checks, retries, and keeps tracking a server that won't stop.
- **Driver check:** setup now checks for driver 595 or newer, and suggests one that is (it used to suggest 570).
- **Old Linux releases:** Ubuntu 20.04 and Debian 11 have no CUDA 13.2. Setup now says so, instead of failing at a
  package error.
- **Serving the network:** `qwen_server.py --host 0.0.0.0` (or `HOST=0.0.0.0 ./run.sh server`) works again. The
  host-name check now applies only when the server listens on localhost.

## 2026-11-04

The first published version of the fork. [FORK.md](FORK.md) has the details.

- **Manager:** it runs the model as a service. The model loads on the first request, the manager picks a full or lean
  profile, scales back or pauses for games, and restarts it after a crash. A dashboard shows it all.
- **Setup:** for Windows (WSL2) and native Linux, from the dashboard or a terminal.
- **Presets:** for 8-48 GB cards, with the 4.05 bpw quant and VRAM scaling benchmarks.
- **API:** the OpenAI Responses API, and OpenAI's `reasoning_effort` values.
- **Other devices:** a network port, with a private link for the dashboard.
- **Engine fixes that upstream can cherry-pick:**
  - pinned memory under WSL2;
  - 4.05 bpw prefill;
  - configurable VRAM limits;
  - kernels built for the local GPU.

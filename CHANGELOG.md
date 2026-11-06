# Changelog

What changed in this fork, newest first. [FORK.md](FORK.md) describes how it differs from the original.

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
  - prompt reading follows the engine's `sweep.py` protocol (fresh random tokens at 8K-64K).
- **Output is never capped:** the old runner cut answers at 400-1,500 tokens.
- **Run records:** [research/README.md](research/README.md) lists every benchmark with its date, machine and stack.

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

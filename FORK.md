# About this fork

[eltorr](https://github.com/eltorr)'s fork of **[MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash)**,
forked at [`fa9a772`](https://github.com/MaxDevv/lean-qwenflash/commit/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d)
(26 Oct 2026).

It started because my AI machine is also my gaming PC: an RTX 5090, a Ryzen 9 9950X, 64 GB of RAM, and Windows 11
with the engine in WSL2. I wanted the model's resources to adapt. When a game needs the GPU or RAM, the model scales
itself back or pauses. When the game closes, the model takes the resources back. The rest of the fork grew around
that. All the measurements come from that PC.

Clone it and do what you like with it; just keep the credit to MaxDevv for the engine. The engine fixes are
standalone commits that apply to the original too; see [Taking changes upstream](#taking-changes-upstream).

## What it adds

- **Manager** ([`qwenflash/`](qwenflash), [guide](docs/manager.md)). It runs the model as a service:
  - loads it on the first request;
  - picks a full or lean profile from the free VRAM and RAM;
  - scales back or pauses for games, and comes back when they're done, without flapping between profiles;
  - keeps the RAM cache to what pays (90% of the experts with VRAM) and keeps it across restarts;
  - restarts it after a crash, using smaller limits after running out of memory;
  - shows it all on a local dashboard.
- **Setup** ([`setup/`](setup), [`qwenflash/setup.py`](qwenflash/setup.py)). It detects Windows or Linux and runs
  each install step, from the dashboard or a terminal.
- **Sizing and presets** ([`sizing.py`](sizing.py), [`run.sh`](run.sh)). They pick the quant, context and RAM cache
  from the VRAM and RAM, and let prompts of up to 1,638 tokens skip prefill staging (`QWEN38_EC_STAGE_MIN`). The
  README's tables are generated from it.
- **Benchmarks** ([guide](docs/manager.md#benchmarks), [run records](research/README.md)). The dashboard's
  Benchmark tab and `qwenflash bench` measure any configuration on this machine, including smaller cards emulated on a
  bigger one, and record the stack. A run steps aside while a game needs the GPU, and a benchmark cut short resumes.
  It measures the way the original's `benchmark.py` does (thinking off, greedy, his prompts and documents), so
  results compare with MaxDevv's. `benchmark.py` and the engine's `bench/sweep.py` are kept as they were.
- **Newer stack.** PyTorch 2.15 on CUDA 13.2, where the original installs PyTorch 2.11 on CUDA 12.8. Prefill is
  47-72% faster ([report](research/torch-2.15/REPORT.md)); it needs NVIDIA driver 595 or newer.
- **Security** ([manager guide](docs/manager.md#security)):
  - the dashboard needs a sign-in token on every port;
  - requests from other websites and DNS rebinding are refused;
  - the model server only answers the manager;
  - every package is pinned by hash, and model downloads are checked against Hugging Face's hashes.
- **Checks** ([below](#checks)). Lint, shellcheck, secret and CVE scans and the unit tests, locally and on GitHub.
- **Tests** ([`tests/`](tests)).

[CHANGELOG.md](CHANGELOG.md) lists the changes by date.

## API: what's original and what's new

- **Original:**
  - `/v1/chat/completions`, with streaming, function calling and reasoning levels;
  - `/v1/completions`;
  - `/v1/models`;
  - the tool endpoints `/generate`, `/tokenize`, `/server_info` and `/stats`.
- **Fixed in the engine:** OpenAI's `reasoning_effort` values map onto the model's levels. Before, `high` returned
  HTTP 500.
- **New in the engine:** the Responses API, `/v1/responses`, with streaming, function tools, reasoning and
  `previous_response_id`.
- **New in the manager:**
  - loads the model on demand;
  - queues requests: one runs at a time, and up to 8 wait;
  - OpenAI-style errors;
  - `/v1/models` without loading the model;
  - speed stats.
- **Not supported:** embeddings, images, audio and Anthropic's `/v1/messages`.

Details are in [docs/api.md](docs/api.md).

## Changes to upstream files

Everything not listed here is new. Defaults are unchanged unless the new environment variables are set, except for
two safety changes: the model server refuses web pages, and it never runs code from a model folder.

| File (`qwen38/` is `qwen38-flash-offload/qwen38/`) | Change |
|---|---|
| `qwen38/nv2.py`, `nv_tier.py`, `k_lookahead.py`, `exl3_tiers.py` (+ `wsl_pinned.py`) | Pinned host memory works under WSL2 |
| `qwen38/nv2.py`, `cpu_tier.py` | Built as C++20, which PyTorch 2.15's headers need |
| `qwen38/expert_cache.py` | Prefill staging holds a whole layer, so 4.05 bpw fits; the VRAM limit counts what CUDA can't see under WSL2 |
| `qwen38/serve.py` (+ `openai_compat.py`, `openai_responses.py`) | `reasoning_effort` mapping; the Responses API; `trust_remote_code` off |
| `qwen_offload.py` | Configurable VRAM limits; kernels built for the local GPU; measures the VRAM CUDA can't see (WSL2) |
| `memory_guard.py` | Configurable limit; no crash without a memory cgroup; `device_used()` for the VRAM limit |
| `qwen_server.py` | Prefill chunk from `QWEN_CHUNK`; refuses web pages, and on localhost other host names (DNS rebinding); takes the manager's key; no access log |
| `download_http.py` | `--branch` and `--sha`, to fetch 4.05 bpw; both quants are pinned to one upload; every file is checked against its hash |
| `qwen38-flash-offload/bench/sweep.py` | Sends the manager's key when it's set |
| `qwen38-flash-offload/b70tier/ring.py` | The shared ring file is owner-only; it was world-writable |
| `.gitignore` (+ `.gitattributes`) | The fork's files; LF endings for scripts that run in Linux |
| `README.md` | Rewritten for the fork; the original's intro, benchmarks and manual install are kept |

## Taking changes upstream

These seven commits each stand alone and cherry-pick onto upstream `master`, in this order:

| Commit | What |
|---|---|
| `59ca8ce` | LF endings for files that run inside Linux, so Windows checkouts used from WSL work |
| `e3ae7c2` | `QWEN_VRAM_LIMIT_BYTES`, `QWEN_TORCH_LIMIT_BYTES` and `QWEN_VRAM_GUARD_BYTES`, so cards over 16 GB get more cache |
| `f05f734` | `cudaHostAlloc` when registered host memory isn't identity-mapped, which WSL2 needs |
| `2f2401c` | `download_http.py --branch/--sha` |
| `0f7740f` | OpenAI `reasoning_effort` values; `high` returned HTTP 500 |
| `5e0167e` | Prefill staging sized per layer; kernels for the local GPU; the RAM guard without a cgroup |
| `ec034b9` | Both quants pinned to one upload and fetched by revision (needs `2f2401c`) |

```bash
git remote add eltorr https://github.com/eltorr/lean-qwenflash.git
git fetch eltorr
git cherry-pick 59ca8ce e3ae7c2 f05f734 2f2401c 0f7740f 5e0167e ec034b9
```

The security changes to upstream files, the C++20 build, and the WSL2 VRAM accounting fix (which also loads the
CUDA runtime PyTorch uses) are part of larger commits, so take those by hand. The
original's quickstart stays on PyTorch 2.11, which doesn't need C++20.

The rest builds in layers: sizing and `run.sh`, then setup, the manager, benchmarks, and presets.
`./run.sh server` is useful on its own: it starts a server sized for the card.

## Branches

- **`qwenflash`** (default): this fork.
- **`master`:** upstream as forked. To pull in new upstream work:

  ```bash
  git remote add upstream https://github.com/MaxDevv/lean-qwenflash.git
  git fetch upstream
  git merge upstream/master
  ```

## Checks

`bash tools/scan.sh` runs locally (Linux, or Git Bash on Windows) what GitHub runs on every push:

- **Lint:** ruff on this fork's Python ([`ruff.toml`](ruff.toml) lists the files).
- **Shell scripts:** shellcheck.
- **Secrets:** tokens and keys in tracked files.
- **Known CVEs:** pip-audit on the pinned packages.
- **Unit tests:** the ones that need no GPU.

`bash tools/lock.sh` regenerates the hash-pinned package lists (`setup/requirements.txt`,
`qwenflash/requirements.txt` and `tools/scan-requirements.txt`). GitHub also runs CodeQL, and Dependabot proposes
updates and security fixes. To run the scan before every push, run `cp tools/pre-push .git/hooks/pre-push`. The hook also refuses `backup/*` branches, and commits matching a pattern in
`.git/info/push-blocklist`, a local file for private names.

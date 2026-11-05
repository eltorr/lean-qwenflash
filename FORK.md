# About this fork

[eltorr](https://github.com/eltorr)'s fork of **[MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash)**,
forked at [`fa9a772`](https://github.com/MaxDevv/lean-qwenflash/commit/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d)
(26 Oct 2026).

It started because my AI machine is also my gaming PC: an RTX 5090, a Ryzen 9 9950X, 64 GB of RAM, and Windows 11
with the engine in WSL2. I wanted the model's resources to adapt. When a game needs the GPU or RAM, the model scales
itself back or pauses. When the game closes, the model takes the resources back. The rest of the fork grew around
that. All the measurements come from that PC.

**For MaxDevv:** take anything you like. The engine fixes are standalone commits; see
[Taking changes upstream](#taking-changes-upstream).

## What it adds

- **Manager** ([`qwenflash/`](qwenflash), [guide](docs/manager.md)). It runs the model as a service:
  - loads it on the first request;
  - picks a full or lean profile from the free VRAM and RAM;
  - scales back or pauses for games, and comes back when they're done;
  - restarts it after a crash, using smaller limits after running out of memory;
  - shows it all on a local dashboard.
- **Setup** ([`setup/`](setup), [`qwenflash/setup.py`](qwenflash/setup.py)). It detects Windows or Linux and runs
  each install step, from the dashboard or a terminal.
- **Sizing and presets** ([`sizing.py`](sizing.py), [`run.sh`](run.sh)). They pick the quant, context and RAM cache
  from the VRAM and RAM. The README's tables are generated from it.
- **Benchmarks** ([`bench/`](bench), [`research/`](research)). VRAM scaling for 8-32 GB cards, on a 5090 capped to
  each size.
- **Tests** ([`tests/`](tests)).

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

Everything not listed here is new. Defaults are unchanged unless the new environment variables are set.

| File (`qwen38/` is `qwen38-flash-offload/qwen38/`) | Change |
|---|---|
| `qwen38/nv2.py`, `nv_tier.py`, `k_lookahead.py`, `exl3_tiers.py` (+ `wsl_pinned.py`) | Pinned host memory works under WSL2 |
| `qwen38/expert_cache.py` | Prefill staging holds a whole layer, so 4.05 bpw fits |
| `qwen38/serve.py` (+ `openai_compat.py`, `openai_responses.py`) | `reasoning_effort` mapping; the Responses API |
| `qwen_offload.py` | Configurable VRAM limits; kernels built for the local GPU |
| `memory_guard.py` | Configurable limit; no crash without a memory cgroup |
| `qwen_server.py` | Prefill chunk from `QWEN_CHUNK` |
| `download_http.py` | `--branch` and `--sha`, to fetch 4.05 bpw |
| `.gitignore` (+ `.gitattributes`) | The fork's files; LF endings for scripts that run in Linux |
| `README.md` | Rewritten for the fork; the original's intro, benchmarks and manual install are kept |

## Taking changes upstream

These six commits each stand alone and cherry-pick onto upstream `master`:

| Commit | What |
|---|---|
| `59ca8ce` | LF endings for files that run inside Linux, so Windows checkouts used from WSL work |
| `e3ae7c2` | `QWEN_VRAM_LIMIT_BYTES`, `QWEN_TORCH_LIMIT_BYTES` and `QWEN_VRAM_GUARD_BYTES`, so cards over 16 GB get more cache |
| `f05f734` | `cudaHostAlloc` when registered host memory isn't identity-mapped, which WSL2 needs |
| `2f2401c` | `download_http.py --branch/--sha` |
| `0f7740f` | OpenAI `reasoning_effort` values; `high` returned HTTP 500 |
| `5e0167e` | Prefill staging sized per layer; kernels for the local GPU; the RAM guard without a cgroup |

```bash
git remote add eltorr https://github.com/eltorr/lean-qwenflash.git
git fetch eltorr
git cherry-pick 59ca8ce e3ae7c2 f05f734 2f2401c 0f7740f 5e0167e
```

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

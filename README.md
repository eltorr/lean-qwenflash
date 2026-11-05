# Lean-QwenFlash (eltorr's fork)

> **This is a personal fork of [MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash), with
> additions.** The engine is MaxDevv's work. It runs Qwen3.8-Flash-Next, a 125B mixture-of-experts model, at full
> 262K context on a 16 GB card by streaming experts from NVMe. This fork keeps that engine and adds:
>
> - a manager with a web dashboard;
> - setup for Windows (WSL2) and Linux;
> - presets that size the model to the machine;
> - a few engine fixes.
>
> It is tuned for my own setup: an RTX 5090 on Windows 11, serving my home network in place of Ollama.
>
> MaxDevv, anything here is yours to take into upstream or to build your own improvements on.

![QwenFlash dashboard: model state, requests, GPU and RAM use, and activity](docs/images/dashboard.png)

## Differences from the original

| | [Original](https://github.com/MaxDevv/lean-qwenflash) | This fork |
|---|---|---|
| **Engine** | Direct-from-SSD expert offloading, packed expert store, OpenAI server | The same engine, plus fixes:<ul><li>pinned memory under WSL2</li><li>prefill staging sized per layer, so 4.05 bpw works</li><li>configurable VRAM limits</li><li>kernels built for the local GPU</li><li>a RAM guard that works without a memory cgroup</li><li>OpenAI `reasoning_effort` values mapped (`high` gave HTTP 500)</li></ul> |
| **Platforms** | Linux | Linux, and Windows 10/11 (engine in WSL2, manager on Windows) |
| **Cards** | Tuned and measured for 16 GB (RTX 4090 capped, RTX 3080 Laptop) | Sized for 8-48 GB; measured at 8-32 GB (RTX 5090 capped to each size) |
| **Quants** | 3.05 bpw | 3.05 and 4.05 bpw |
| **Settings** | Server flags chosen by hand | Picked by `sizing.py`. Presets set quant, context and RAM cache from VRAM + RAM, and `./run.sh server` starts a server sized to the card |
| **Install** | Manual steps | Setup tab or `qwenflash setup`:<ul><li>Windows: WSL and Ubuntu</li><li>CUDA 12.8 and the Python environment</li><li>kernel build</li><li>model download and packing</li></ul> |
| **Running** | Start `qwen_server.py`; it runs until stopped | A manager service:<ul><li>starts the model on the first request</li><li>scales back or pauses for games and other GPU apps</li><li>restarts after a crash</li><li>keeps the model up across its own restarts</li></ul> |
| **API** | OpenAI-compatible on port 30000 | The same, plus an Ollama-style front door on port 11434 that passes other models through to Ollama |
| **Monitoring** | Terminal output | Local web dashboard (loopback only): state, speed, GPU/RAM over time, settings, setup, logs |
| **Benchmarks** | 4090 / 3080 table | Adds a VRAM-scaling benchmark, a preset matrix by VRAM and RAM, and reports |

[FORK.md](FORK.md) has the full list: every upstream file changed, every commit, and the standalone engine-fix
commits that upstream can cherry-pick. It also covers what is personal to my setup and what is verified.

## Credit

- **[lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash) by MaxDevv** is the engine this fork is built on:
  direct-from-SSD expert offloading, the packed expert store, the OpenAI server, and the original benchmarks and
  documentation. Upstream's README text is kept below, from *About the engine* on.
- Through upstream:
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp;
  - [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) by sybil-solutions;
  - [Qwen3.8-Flash-Next](https://github.com/QwenLM) by the Qwen team.

  See [Lineage](#lineage).
- **Licenses:** upstream has no top-level license file yet, so this fork doesn't add one. The bundled ExLlamaV3 and
  offload code keep their MIT licenses.

---

## About the engine

*From the original README.*

**Qwen3.8-Flash-Next at full 262k context on 8 GB RAM and 16 GB VRAM—20 tok/s generation and 1,600 tok/s prefill.**

![Qwen3.8-Flash-Next inference showcase](media/showcase.gif)

*Inference run on an RTX 4090, capped at 16 GB VRAM and 8 GB system RAM.*

Direct-from-SSD expert offloading for **Qwen3.8-Flash-Next** (125B MoE, 512 experts).

Most setups for this model require 48–64 GB of system RAM just to hold the expert weights in memory. This runtime streams uncached experts directly from NVMe storage over PCIe into a bounded GPU cache via `O_DIRECT`, while streaming the 51B N-gram embedding table from disk. All expert computation runs on the GPU.

Built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and adapted from the [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) engine.

---

## Using this fork

| | Windows 10/11 | Linux |
|---|---|---|
| Get it | `git clone https://github.com/eltorr/lean-qwenflash.git` (needs Git and Python 3) | same (needs Python 3) |
| Start | double-click `QwenFlash.cmd` | `./qwenflash.sh open` |
| Then | Setup tab: run the steps top to bottom | Setup tab, or `./qwenflash.sh setup STEP` in a terminal |
| Clients | `http://127.0.0.1:30000/v1`, model `qwen3.8-flash-next` | same |

The Setup tab shows what it detected, this machine's preset, and each setup step:

![Setup tab: detected hardware, the preset for this machine, and the setup steps](docs/images/dashboard-setup.png)

Guides: [manager](docs/manager.md) · [API](docs/api.md) · [VRAM scaling results](research/vram-scaling/REPORT.md) ·
[multi-GPU notes](docs/research/multi-gpu.md).

### Presets: a baseline from VRAM and RAM

The manager starts from a preset for the card, the RAM and the platform: quant, context window, RAM cache, and its
own RAM thresholds. A new install applies it. Later, the dashboard offers it as a recommendation when your settings
differ. `./qwenflash.sh plan` shows your machine's preset; `python3 sizing.py baseline --vram-gb 24 --ram-gb 128`
shows any machine's.

Linux, with the card also driving the desktop (except where the 8 GB rows say otherwise). Each cell: quant ·
context, then the RAM cache and how much of the model it keeps in memory (VRAM + RAM). *Italics:* measured
generation speed of that configuration (chart below).

<!-- presets:begin -->
| Card (VRAM) | 16 GB RAM | 32 GB RAM | 64 GB RAM | 128 GB RAM |
|---|---|---|---|---|
| **8 GB, no display** | **3.05** · 16K, no staging<br>4 GiB RAM cache, 10% in memory | **3.05** · 16K, no staging<br>15 GiB RAM cache, 36% in memory | **3.05** · 16K, no staging<br>42 GiB RAM cache, all in memory<br>*20.2 tok/s* | **3.05** · 16K, no staging<br>42 GiB RAM cache, all in memory<br>*20.2 tok/s* |
| **8 GB, with display** | not enough VRAM | not enough VRAM | not enough VRAM | not enough VRAM |
| **12 GB** | **3.05** · 64K, no staging<br>4 GiB RAM cache, 15% in memory | **3.05** · 64K, no staging<br>15 GiB RAM cache, 41% in memory | **3.05** · 64K, no staging<br>40 GiB RAM cache, all in memory | **3.05** · 64K, no staging<br>40 GiB RAM cache, all in memory |
| **16 GB** | **3.05** · 256K<br>4 GiB RAM cache, 18% in memory | **3.05** · 256K<br>15 GiB RAM cache, 44% in memory | **3.05** · 256K<br>39 GiB RAM cache, all in memory<br>*37.8 tok/s* | **3.05** · 256K<br>39 GiB RAM cache, all in memory<br>*37.8 tok/s* |
| **24 GB** | **3.05** · 256K<br>4 GiB RAM cache, 34% in memory | **3.05** · 256K<br>15 GiB RAM cache, 60% in memory | **4.05** · 256K<br>45 GiB RAM cache, all in memory<br>*42.6 tok/s* | **4.05** · 256K<br>47 GiB RAM cache, all in memory<br>*42.6 tok/s* |
| **32 GB** | **3.05** · 256K<br>4 GiB RAM cache, 51% in memory | **3.05** · 256K<br>15 GiB RAM cache, 77% in memory | **4.05** · 256K<br>40 GiB RAM cache, all in memory<br>*56.1 tok/s* | **4.05** · 256K<br>40 GiB RAM cache, all in memory<br>*56.1 tok/s* |
| **48 GB** | **3.05** · 256K<br>4 GiB RAM cache, 89% in memory | **3.05** · 256K<br>9 GiB RAM cache, all in memory | **4.05** · 256K<br>24 GiB RAM cache, all in memory | **4.05** · 256K<br>24 GiB RAM cache, all in memory |
<!-- presets:end -->

- **Quant:** 4.05 bpw (higher fidelity) where it keeps 256K context, full-speed prompt processing and nearly every
  expert in VRAM or RAM. Otherwise 3.05 bpw: its experts are 25% smaller, so it is faster at every card size.
- **Context:** the largest that keeps staged prefill. 12 GB cards get 64K without staging (agents need the window
  more than the faster prefill); 8 GB cards get 16K without staging, and only when the card drives no display.
- **RAM:** the RAM cache holds every expert that isn't in VRAM, as far as RAM allows; the rest is read from the NVMe
  drive when picked. Experts are picked very unevenly, so the share of *picks* served from memory is far higher
  than the share of experts held (a 16 GB card's 2,400 cached experts served 66% of picks). On 16-32 GB machines the
  manager's RAM thresholds scale down with the RAM.
- **Windows** keeps about 8 GB more for itself and WSL: 16 GB of RAM is too little there, and 32-64 GB machines get
  smaller RAM caches (`python3 sizing.py presets --platform windows-wsl`).

**Measured.** An RTX 5090 (WSL2, 64 GB RAM) capped to each card's VRAM
([method and full results](research/vram-scaling/REPORT.md)). The caps emulate VRAM capacity only: real smaller
cards also have less bandwidth and slower PCIe, so expect less. Not measured yet: more RAM with less VRAM (128 GB
with a 24 GB card or smaller), less RAM (16-32 GB), and real 8-24 GB cards. New results go into `sizing.MEASURED`;
`python3 sizing.py presets --write README.md` regenerates both tables.

<!-- measured:begin -->
| Card (VRAM) | Quant | Context | RAM cache | Prefill | Decode | Measured on |
|---|---|---|---|---|---|---|
| **32 GB** | 3.05 | 256K | 25 GiB | 5,754 tok/s | 64.0 tok/s | RTX 5090, capped (64 GB RAM) |
| **24 GB** | 3.05 | 256K | 32 GiB | 5,896 tok/s | 51.5 tok/s | RTX 5090, capped (64 GB RAM) |
| **16 GB** | 3.05 | 256K | 38 GiB | 4,338 tok/s | 37.8 tok/s | RTX 5090, capped (64 GB RAM) |
| **12 GB** | 3.05 | 64K | 38 GiB | 1,816 tok/s | 33.2 tok/s | RTX 5090, capped (64 GB RAM) |
| **12 GB** | 3.05 | 128K, no staging | 38 GiB | 1,564 tok/s | 29.9 tok/s | RTX 5090, capped (64 GB RAM) |
| **8 GB, no display** | 3.05 | 16K, no staging | 38 GiB | 1,166 tok/s | 20.2 tok/s | RTX 5090, capped (64 GB RAM) |
| **32 GB** | 4.05 | 256K | 38 GiB | 5,293 tok/s | 56.1 tok/s | RTX 5090, capped (64 GB RAM) |
| **24 GB** | 4.05 | 256K | 38 GiB | 5,006 tok/s | 42.6 tok/s | RTX 5090, capped (64 GB RAM) |
| **16 GB** | 4.05 | 256K | 38 GiB | 1,384 tok/s | 30.3 tok/s | RTX 5090, capped (64 GB RAM) |
| **12 GB** | 4.05 | 32K, no staging | 38 GiB | 431 tok/s | 26.6 tok/s | RTX 5090, capped (64 GB RAM) |
<!-- measured:end -->

---

## Measured Performance

Benchmarked using `turboderp/Qwen3.8-Flash-Next-exl3` (`3.05bpw_h5_ng5`) with 4-bit KV cache unless noted:

| Hardware | Context | System RAM Used | VRAM Used | Prefill | Decode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090** | 8k | ~8 GB | <16 GB | 1,080 tok/s | 28–33 tok/s (prose)<br>19–22 tok/s (code) |
| **RTX 4090** | 100k | ~8 GB | 15.6 GB | 1,536 tok/s | 20.8 tok/s |
| **RTX 4090** | 262k | ~8 GB | 15.6 GB | 1,627 tok/s | 20.0 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (8 GB cache) | 19.0 GB | 15.5 GB | 647 tok/s | 17.6 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (16 GB cache) | 23.9 GB | 15.5 GB | 693 tok/s | 20.9 tok/s |
| **RTX 4090** | 17 tokens (4 GB cache, unquantized KV) | 8.55 GB | 14.90 GB | — | 16.0 tok/s (code) |

This fork's measurements on 8-32 GB cards, and the preset for each card and amount of RAM, are under
[Presets](#presets-a-baseline-from-vram-and-ram).

---

## How It Works

Qwen3.8-Flash-Next activates only 10 of its 512 experts per layer per token (~2% of weights).

* **GPU (16 GB VRAM):** Dense attention mixers, DeltaNet state, routers, shared experts, output head, 4-bit KV cache, and an adaptive GPU expert cache.
* **Host RAM:** Configurable cache (4–16 GB) holding the most recently activated experts.
* **NVMe Storage:** The full ~47 GB packed expert store and 32 GB N-gram table sit on SSD. Cache misses are read directly via asynchronous ring buffers and staged ahead of the compute stream.
* **GPU-Only Math:** All expert computation stays on the GPU. Misses stream over PCIe rather than falling back to CPU execution, avoiding host DRAM bandwidth limits.
* **Prefix Caching:** Recurrent state checkpoints are saved alongside the KV cache so multi-turn agent requests start decoding immediately (~0.8s–1.4s TTFT on cached follow-ups).

---

## Quickstart

### 1. Requirements
* Linux (x86_64), or Windows 10/11 with the engine in WSL2 (the [manager](docs/manager.md) sets that up)
* NVIDIA GPU, Turing or newer, with at least 12 GB VRAM (16 GB+ recommended). An 8 GB card works at 16k context if
  it doesn't also drive a display ([measurements](research/vram-scaling/REPORT.md))
* RAM: 16 GB on Linux and 32 GB on Windows at minimum; 64 GB keeps the whole model in VRAM + RAM
  ([presets](#presets-a-baseline-from-vram-and-ram))
* Fast NVMe SSD (PCIe 3.0 or 4.0, ~3+ GB/s read throughput)
* CUDA 12.4+ (12.8 for RTX 50 cards) and Python 3.10–3.12

With the manager, `qwenflash setup` (or the dashboard's Setup tab) runs the steps below for you and sizes everything
for your card. `./run.sh server` starts the server sized from the hardware, without the manager.

### 2. Install

```bash
git clone https://github.com/eltorr/lean-qwenflash.git   # this fork; upstream: MaxDevv/lean-qwenflash
cd lean-qwenflash

python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install fastapi uvicorn safetensors jinja2 huggingface_hub
pip install -e exllamav3
```

Compile the native offload extensions:
```bash
python compile_local.py
```

### 3. Download & Build Store

Download the 3.05bpw checkpoint and N-gram table (~85 GB total):
```bash
python download_http.py --model-dir ./model
```

Pack the routed experts into a direct-I/O binary store (~47 GB):
```bash
mkdir -p ./store
python qwen38-flash-offload/scripts/pack_qwen38_store.py pack \
  --model ./model \
  --out-dir ./store \
  --name qwen
```

---

## Serving

Start the OpenAI-compatible server on port `30000`:

```bash
python qwen_server.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 8 \
  --cache 262144 \
  --chunk 24576 \
  --port 30000
```

### Agent Configuration
Point tools like Pi, OpenCode, or Claude Code to:
* **Base URL:** `http://127.0.0.1:30000/v1`
* **Model:** `qwen3.8-flash-next`
* **Context Window:** `262144`

Supports streaming, reasoning effort control (`off`, `low`, `medium`, `xhigh`), and OpenAI function calling.

### CLI Test
To run a one-shot generation from the terminal:

```bash
python infer.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 4 \
  --prompt "Write an implementation of merge sort in C."
```

---

## To Do

If I get some free time—or someone wants to contribute or put an agent on it:

- ~~**Lower VRAM:** Adapt and benchmark on 12 GB, and potentially 8 GB GPUs.~~
  - **Done in this fork:** `sizing.py` sizes the cache, prefill chunk, staging and reserve for the card. 12 GB runs
    3.05 bpw up to 128k context and 4.05 at 32k. 8 GB runs 3.05 at 16k when the card has no display attached.
    Measured with the engine capped on an RTX 5090 ([report](research/vram-scaling/REPORT.md)).
  - **Still open:** real 8-12 GB cards; an 8 GB card that also drives a desktop. For the latter, shrink the ~6.2 GB
    fixed footprint first (≈2.5 GB of it is not yet attributed; per-expert scales could move into the expert
    records).
- ~~**Higher VRAM:** Benchmark 24 GB and 32 GB GPUs.~~ **Done:** decode 51.5 / 64 tok/s (3.05) at 24 / 32 GB; prefill
  stays ~5,800 tok/s ([report](research/vram-scaling/REPORT.md)).
- **Presets, next measurements:** more RAM with less VRAM (128 GB with a 24 GB card and smaller), 16-32 GB of RAM,
  and real 8-24 GB cards. `bench/vram_matrix.py` runs them (`:ram=N` sets the RAM cache) and
  `bench/vram_table.py --sizing` prints the rows for `sizing.MEASURED`.
- **Prefill on 12-16 GB cards:** a single staging buffer, to keep big chunks where two don't fit.
- **Two GPUs:** not supported by the offload layer yet. A layer split looks like the best fit
  ([research notes](docs/research/multi-gpu.md)).

---

## Lineage

* [lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash) by MaxDevv: the upstream of this fork (the engine,
  expert store and server). This fork's additions are listed in [FORK.md](FORK.md).
* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization format & inference runtime).
* [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) by sybil-solutions (NVMe staging & memory tiering architecture).
* [Qwen3.8-Flash-Next](https://github.com/QwenLM) by Alibaba Cloud Qwen team.

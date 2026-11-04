"""VRAM and RAM sizing for Qwen3.8-Flash-Next on this engine, as plain arithmetic (no torch, no GPU).

One source of truth for run.sh (`python3 sizing.py env ...`), the server (`--chunk auto`) and the QwenFlash manager.

Where VRAM goes (code refs: qwen38-flash-offload/qwen38/expert_cache.py, nv2.py; exllamav3 cache/quant.py):
  fixed     weights that stay on the GPU (attention, shared experts, routers, lm_head, per-expert scales), KV cache,
            DeltaNet state, CUDA context/kernels/graphs. Grows with context only through the KV cache.
  cache     expert cache: S slots of one expert record each (rec bytes), in 32-slot chunks.
  reserve   VRAM left free after the cache is allocated (QWEN38_EC_RESERVE_GB).
  prefill   two staging buffers of a whole layer (512 records each) plus activations for one chunk. They come out of
            the reserve plus cache chunks the elastic cache lends back during prefill, so they are not extra.
A plan fits when fixed + cache + reserve <= budget and cache + reserve - 0.35 GB >= prefill(chunk).

Calibration (research/vram-scaling): VRAM used outside the expert cache measured 9.0 GB (3.05 bpw) and 9.44 GB
(4.05 bpw) at 262,144 context and 6.85 GB at 64K (3.05), i.e. 6.15 / 6.62 GB plus the KV cache's 10,752 bytes per
token; an 8 GB-class run used 6.39 GB at 16K. Activations per chunk token come from the RTX 4090 research runs (a
24,576-token chunk fit with a 16 GB cap; research/4090-long). The reserves include ~0.3 GB of margin found by
running long prompts on emulated 8-16 GB cards.
"""
import argparse
import json
import sys

GB = 1e9
GIB = 1024 ** 3
EXPERTS_PER_LAYER = 512
MOE_LAYERS = 48
ACT_BYTES_PER_TOKEN = 0.21e6          # research/4090-long: a 24,576-token chunk fit, 32,768 did not
PREFILL_EXTRA = 0.3 * GB
FRAGMENTATION = 0.5 * GB   # torch held 0.8-1 GB more than it used when staging was allocated on small cards
ELASTIC_SLACK = 0.35 * GB
CHUNKS = (24576, 16384, 8192, 4096, 2048, 1024)
NO_STAGE_CHUNK = 1536           # without staging, nv2's pick buffer limits a chunk to 1,638 tokens
KV_BYTES_PER_TOKEN = {2: 7_680, 4: 10_752, 8: 16_896, 16: 28_416}

QUANTS = {
    # rec: one expert record (gate + up + down trellis, 256-aligned); fixed0: non-expert VRAM at zero context
    "3.05": {"rec": 1_843_200, "fixed0": 6.15 * GB, "model_dir": "model", "store": "store/qwen.bin",
             "branch": "3.05bpw_h5_ng5", "download_gb": 85, "store_gb": 47, "label": "3.05 bpw (fastest)"},
    "4.05": {"rec": 2_457_600, "fixed0": 6.62 * GB, "model_dir": "model-4.05", "store": "store/qwen405.bin",
             "branch": "4.05bpw_h6_ng6", "download_gb": 107, "store_gb": 62, "label": "4.05 bpw (higher fidelity)"},
}


def expert_bytes(quant):
    """Every routed expert of the model (what VRAM cache + RAM tier + NVMe hold between them)."""
    return QUANTS[quant]["rec"] * EXPERTS_PER_LAYER * MOE_LAYERS


def fixed_bytes(quant, context, kv_bits=4):
    return QUANTS[quant]["fixed0"] + KV_BYTES_PER_TOKEN[kv_bits] * context


def stage_bytes(quant):
    return 2 * EXPERTS_PER_LAYER * QUANTS[quant]["rec"]


def prefill_bytes(quant, chunk, staging=True):
    return (stage_bytes(quant) if staging else 0) + ACT_BYTES_PER_TOKEN * chunk + PREFILL_EXTRA


def reserve_bytes(budget):
    """Free VRAM kept after the cache: 1.7 GB (upstream default) on big cards, down to 0.9 GB on 12 GB ones.
    (0.4-0.87 GB let long prompts overrun emulated 12-16 GB cards by 20-80 MB.)"""
    return min(1.7 * GB, max(0.9 * GB, 0.06 * budget))


def nostage_reserve(quant, chunk):
    return max(0.7 * GB, prefill_bytes(quant, chunk, staging=False) + 0.2 * GB)


def plan(quant, budget, context=262144, kv_bits=4, chunk=None, max_slots=None):
    """Best configuration for `budget` bytes of VRAM for this process (excluding other apps).

    Returns a dict with ok, slots, chunk, staging, reserve and the byte breakdown; ok=False carries a reason and
    the smallest budget that would work at this context."""
    q = QUANTS[quant]
    rec = q["rec"]
    fixed = fixed_bytes(quant, context, kv_bits)
    reserve = reserve_bytes(budget)
    cache = max(0.0, budget - fixed - reserve)
    slots = int(cache // rec) // 32 * 32
    if max_slots is not None:
        slots = min(slots, int(max_slots))
    cache = slots * rec
    out = {"quant": quant, "context": context, "kv_bits": kv_bits, "budget": budget, "fixed": fixed,
           "reserve": reserve, "cache": cache, "slots": slots, "staging": True, "chunk": None, "ok": False,
           "experimental": budget < 11 * GB, "ram_for_all_experts": max(0.0, expert_bytes(quant) - cache),
           "min_budget": min_budget(quant, context, kv_bits)}
    room = cache + reserve - ELASTIC_SLACK
    candidates = [chunk] if chunk else CHUNKS
    for c in candidates:
        if room >= prefill_bytes(quant, c) + FRAGMENTATION:
            out.update(chunk=c, ok=True)
            return out
    # staging-free fallback: every miss is read zero-copy from pinned RAM (supported by nv2, not yet measured)
    c = min(chunk or NO_STAGE_CHUNK, NO_STAGE_CHUNK)
    r = nostage_reserve(quant, c)
    if budget >= fixed + r:
        cache = max(0.0, budget - fixed - r)
        slots = int(cache // rec) // 32 * 32
        out.update(chunk=c, staging=False, ok=True, experimental=True, reserve=r, slots=slots, cache=slots * rec,
                   ram_for_all_experts=max(0.0, expert_bytes(quant) - slots * rec))
        return out
    out["reason"] = (f"{budget / GB:.1f} GB is not enough at {context // 1024}K context: the model needs about "
                     f"{fixed / GB:.1f} GB before any expert cache, and {out['min_budget'] / GB:.1f} GB in total")
    return out


def min_budget(quant, context, kv_bits=4):
    """Smallest budget with staging and the smallest chunk."""
    fixed = fixed_bytes(quant, context, kv_bits)
    need_cache_plus_reserve = prefill_bytes(quant, CHUNKS[-1]) + FRAGMENTATION + ELASTIC_SLACK
    return fixed + max(need_cache_plus_reserve, 0.7 * GB + 32 * QUANTS[quant]["rec"])


def max_context(quant, budget, kv_bits=4, step=16384):
    """Largest context (multiple of `step`, at most 262,144) that still leaves a workable plan."""
    best = 0
    for ctx in range(step, 262144 + 1, step):
        if plan(quant, budget, ctx, kv_bits)["ok"] and plan(quant, budget, ctx, kv_bits)["staging"]:
            best = ctx
    return best


SMALL_CARD = 16 * GB


def engine_env(p, cap_slots=False):
    """Environment for the engine that realizes plan p (run.sh / the manager add the device-wide VRAM limit).

    cap_slots pins the cache size (a budgeted run must not grow if other apps free VRAM later); a full run leaves
    it to the engine, which measures free VRAM itself and gets a slightly bigger cache than this model predicts.
    Budgets of 16 GB and more keep the tuned upstream reserve/elastic settings."""
    env = {"QWEN_CHUNK": str(p["chunk"])}
    if cap_slots:
        env["QWEN38_EC_MAX_SLOTS"] = str(p["slots"])
    if not p["staging"]:
        env.update({"QWEN38_EC_STAGE_GB": "0", "QWEN38_EC_ELASTIC_GB": "0",
                    "QWEN38_EC_RESERVE_GB": f"{p['reserve'] / GIB:.2f}"})
    elif p["budget"] < SMALL_CARD:
        # small cards: a smaller reserve, and the whole cache may be lent back for prefill if needed (the engine's
        # own activation estimate stays: 0.25 MB/token let a 4.05 prefill overrun an emulated 16 GB card)
        env.update({"QWEN38_EC_RESERVE_GB": f"{p['reserve'] / GIB:.2f}",
                    "QWEN38_EC_ELASTIC_GB": f"{max(6.0, p['cache'] / GIB + 1):.1f}"})
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "env"):
        s = sub.add_parser(name)
        s.add_argument("--quant", default="3.05", choices=sorted(QUANTS))
        s.add_argument("--budget-gb", type=float, required=True, help="VRAM for this process, GB (1e9)")
        s.add_argument("--context", type=int, default=262144)
        s.add_argument("--kv-bits", type=int, default=4, choices=sorted(KV_BYTES_PER_TOKEN))
        s.add_argument("--chunk", type=int)
    a = ap.parse_args(argv)
    p = plan(a.quant, a.budget_gb * GB, a.context, a.kv_bits, a.chunk)
    if a.cmd == "plan":
        big = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1e6  # noqa: E731
        print(json.dumps({k + ("_gb" if big(v) else ""): (round(v / GB, 2) if big(v) else v) for k, v in p.items()},
                         indent=2))
    else:
        if not p["ok"]:
            print(f"echo 'sizing: {p['reason']}' >&2; exit 3")
            return 3
        for k, v in engine_env(p).items():
            print(f"export {k}={v}")
        print(f"QWEN_RAM_ALL_GIB={int(p['ram_for_all_experts'] // GIB) + 1}")
    return 0 if p["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())

"""VRAM scaling benchmark (README To-Do): emulate smaller cards on a big one and measure prefill/decode.

Runs on the engine side (native Linux or inside WSL) with the engine's venv python, from the repo root:

  .venv/bin/python bench/vram_matrix.py --runs 3.05:30:262144,3.05:14.5:262144,3.05:10.5:65536 \
      --out research/vram-scaling/results

Each run is QUANT:USABLE_GB:CONTEXT[:nostage]. USABLE_GB is the VRAM the model may use, as on a smaller card after
the desktop's share. The engine is capped the way the upstream 16 GB numbers were measured: its VRAM limit
(QWEN_VRAM_LIMIT_BYTES) sizes the expert cache to what is in use now + USABLE_GB, and its guard
(QWEN_VRAM_GUARD_BYTES) stops the server if device usage ever goes past that, which is what running out of memory
would do on the real card. run.sh asks sizing.py for the chunk/staging plan of that budget.

":nostage" forces the staging-free path (every prefill miss read zero-copy from pinned RAM, 1,536-token chunks,
small reserve) even where sizing.py says the card is too small: that is how the 8 GB experiments run.

What this cannot emulate: a smaller card's memory bandwidth, compute and PCIe link. Absolute speeds are those of
this GPU with less VRAM; the scaling between rows is the result.

Why not a ballast process holding the rest of the card: under Windows/WSL2 the driver (WDDM) over-commits VRAM and
pages idle allocations out to system RAM, so the engine got the ballast's memory back (research/vram-scaling).

The QwenFlash manager must not be managing the model meanwhile (pause it and turn auto-switch off first).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import sizing  # noqa: E402

GB, GIB = 1e9, 1024 ** 3
PROMPTS = [
    "Write a Python asyncio TCP port scanner with a concurrency limit, timeouts, and a CLI. Include docstrings.",
    "Explain how a B+ tree handles insertions that split nodes, with a worked example.",
    "Write a Rust function that parses an RFC 3339 timestamp without external crates, with tests.",
    "Compare Raft and Paxos for a team choosing a consensus algorithm. Be concrete.",
    "Write a bash script that rotates logs older than 7 days, compresses them, and keeps 30 archives.",
    "Write a TypeScript React hook for debounced search with cancellation via AbortController.",
]
TOKENS_PER_HELPER = 22.5            # measured: 4,200 filler functions = 94,466 prompt tokens


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO, **kw)


def gpu_used_total():
    out = sh("nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits").stdout
    used, total = (int(x) * 1048576 for x in out.split(","))
    return used, total


def chat(port, content, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next", "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = usage = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if piece:
                now = time.time()
                t_first = t_first or now
                t_last = now
                text.append(d.get("content") or "")
    n, p = (usage or {}).get("completion_tokens", 0), (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    dec = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and n > 1 else None
    return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": ttft, "decode_tps": dec, "wall_s": time.time() - t0,
            "answer": "".join(text)}


def workload(port, context):
    res = {"cold_first": chat(port, PROMPTS[0], 600)}
    chat(port, PROMPTS[1], 400)                                       # warm-up, not scored
    runs = [chat(port, p, 700) for p in PROMPTS[1:]]
    rates = [r["decode_tps"] for r in runs if r["decode_tps"]]
    res["decode_tps_median"] = statistics.median(rates) if rates else None
    res["decode_tps_min"] = min(rates) if rates else None
    res["short_ttft_median_s"] = statistics.median(r["ttft_s"] for r in runs if r["ttft_s"]) if runs else None
    # long prompt sized to the context window: prefill throughput + needle check
    n = int(min(94000, context - 6000) / TOKENS_PER_HELPER)
    filler = "".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(n))
    doc = filler[: len(filler) // 2] + "\n# NOTE: the deployment password is 'violet-otter-42'.\n" + filler[len(filler) // 2:]
    q = doc + "\n\nWhat is the deployment password mentioned in the code? Answer in one line."
    long = chat(port, q, 1500)
    long["prefill_tps"] = long["prompt_tokens"] / long["ttft_s"] if long["ttft_s"] else None
    long["needle_found"] = "violet-otter-42" in long["answer"]
    res["long"] = long
    for r in [res["cold_first"], long]:
        r.pop("answer", None)
    res["decode_runs"] = [{k: v for k, v in r.items() if k != "answer"} for r in runs]
    return res


def backend(*args):
    return sh("bash", "bin/qwenflash-backend", *args)


def engine_stats(port):
    st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10))
    ec = (st.get("expert_cache") or [{}])[0]
    h = (st.get("nv2") or {}).get("host", {})
    picks = max(1, h.get("vram", 0) + h.get("ram", 0) + h.get("nvme", 0))
    return {"slots": ec.get("slots"), "arena_gb": ec.get("arena_gb"), "mem_peak_gb": ec.get("mem_peak_gb"),
            "hit_rate": ec.get("hit_rate"), "stage_overflow": (ec.get("stage") or {}).get("overflow"),
            "ram_slots": (st.get("nv2") or {}).get("ram_slots"),
            "picks_vram_pct": round(100 * h.get("vram", 0) / picks, 1),
            "picks_ram_pct": round(100 * h.get("ram", 0) / picks, 1),
            "picks_nvme_pct": round(100 * h.get("nvme", 0) / picks, 1)}


def one(quant, usable_gb, context, port, out_dir, nostage=False):
    label = f"{quant}-{usable_gb:g}gb-{context // 1024}k" + ("-nostage" if nostage else "")
    print(f"=== {label}", flush=True)
    backend("stop", "20")
    time.sleep(3)
    used0, total = gpu_used_total()
    usable = usable_gb * GB
    plan = sizing.plan(quant, usable, context)
    rec = {"label": label, "quant": quant, "usable_gb": usable_gb, "context": context, "nostage": nostage,
           "device_used_before_gb": round(used0 / GB, 2), "method": "engine cap (limit + guard)",
           "plan": {k: v for k, v in plan.items() if isinstance(v, (int, float, str, bool)) or v is None}}
    if not plan["ok"] and not nostage:
        rec["error"] = plan.get("reason")
        print("  does not fit:", rec["error"], flush=True)
    else:
        need_all = plan["ram_for_all_experts"] if plan["ok"] else sizing.expert_bytes(quant)
        ram_gib = int(min(need_all / GIB + 1, 38))
        env = [f"QUANT={quant}", f"CONTEXT={context}", f"PORT={port}", "HOST=127.0.0.1", f"RAM_GB={ram_gib}",
               f"QWEN_VRAM_LIMIT_BYTES={int(used0 + usable)}",
               # a little slack for the desktop changing its own use during the run
               f"QWEN_VRAM_GUARD_BYTES={int(used0 + usable + 0.25 * GB)}"]
        if nostage:
            r_gb = sizing.nostage_reserve(quant, 1536) / GIB
            env += ["QWEN_CHUNK=1536", "QWEN38_EC_STAGE_GB=0", "QWEN38_EC_ELASTIC_GB=0", f"QWEN38_EC_RESERVE_GB={r_gb:.2f}"]
        rec["ram_gib"] = ram_gib
        t0 = time.time()
        r = backend("start", *env, "--", "server")
        ok = False
        while time.time() - t0 < 900:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                ok = True
                break
            except Exception:  # noqa: BLE001
                if not backend("status").stdout.startswith("running"):
                    break
                time.sleep(2)
        rec["startup_s"] = round(time.time() - t0, 1)
        log = REPO / ".qwenflash" / "backend.log"
        if ok:
            used_ready, _ = gpu_used_total()
            rec["model_vram_gb"] = round((used_ready - used0) / GB, 2)
            try:
                rec.update(workload(port, context))
                rec["engine"] = engine_stats(port)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"workload failed: {e}"
            if not backend("status").stdout.startswith("running"):
                rec["error"] = rec.get("error") or "server exited during the workload"
        else:
            rec["error"] = f"server did not come up ({r.stdout.strip()})"
        if rec.get("error"):
            rec["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
        backend("stop", "20")
    out = Path(out_dir) / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    summary = {k: rec.get(k) for k in ("label", "startup_s", "model_vram_gb", "decode_tps_median", "error")}
    summary["slots"] = (rec.get("engine") or {}).get("slots")
    if rec.get("long"):
        summary["prefill_tps"] = rec["long"].get("prefill_tps")
        summary["needle"] = rec["long"].get("needle_found")
    print("  ", json.dumps(summary)[:600], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="comma-separated QUANT:USABLE_GB:CONTEXT[:nostage]")
    ap.add_argument("--out", default=str(REPO / "research" / "vram-scaling" / "results"))
    ap.add_argument("--port", type=int, default=30001)
    a = ap.parse_args()
    for spec in a.runs.split(","):
        q, gb, ctx, *flags = spec.split(":")
        one(q, float(gb), int(ctx), a.port, a.out, nostage="nostage" in flags)
    print("MATRIX_DONE", flush=True)


if __name__ == "__main__":
    os.chdir(REPO)
    main()

"""Benchmarks: start the engine with a given configuration, run a fixed workload against it, and record the results.

One tool for the dashboard's Benchmark tab, `qwenflash bench`, and the runs behind the README's tables (research/).
It takes the GPU: the manager stops the model first, refuses requests meanwhile, and puts the model back afterwards.

Runs are QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB], or a preset:
  current  the current settings on the whole card, as the full profile runs them
  lean     the current settings in the lean profile's VRAM and RAM
  quants   3.05 and 4.05 bpw at the current context, on the whole card
  matrix   8-32 GB cards emulated on this one: the README's measured table (needs a 32 GB card)
VRAM_GB holds the engine to that much VRAM, as on a smaller card after the desktop's share: QWEN_VRAM_LIMIT_BYTES sizes
the expert cache, and QWEN_VRAM_GUARD_BYTES stops the server past it, as running out of memory would on that card. A
smaller card's bandwidth can't be emulated, so real smaller cards are slower. ":nostage" forces staging-free prefill
(1,536-token chunks); ":ram=GIB" sets the RAM cache (default: what the full profile would get now).

Workloads. Every answer runs to its natural end; output is never capped:
  quick     two answers, then prompt reading at 8K and 32K tokens                                  ~5 min a run
  standard  a warm-up and five answers, a long document with one fact to find, then prompt
            reading at 8K, 16K, 32K and 64K tokens (3 each)                                      ~15 min a run
Prompt reading uses fresh random tokens, so nothing comes from the prefix cache, timed to the first generated token
(the protocol of qwen38-flash-offload/bench/sweep.py). The long document is natural text, as clients send it.

Results: one JSON per run in the settings folder (bench/<date-time>/), saved after every step, with the machine and
its stack (GPU, driver, CUDA, PyTorch, commit) and the change since the last run of the same configuration.
"""
import asyncio
import collections
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from . import config
from .config import GB, GIB, REPO, log
from .httpio import BACKEND_KEY_HEADER, http_ok

sys.path.insert(0, str(REPO))
import sizing  # noqa: E402

BENCH_DIR = config.STATE / "bench"
RUN_DIR = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
MAX_RUNS = 32
PRESETS = {"current": "This machine's settings, whole card", "lean": "The lean profile",
           "quants": "3.05 and 4.05 bpw, whole card", "matrix": "8-32 GB cards emulated on this one"}
# the README's measured table: card sizes after the desktop's share, each at the largest context that keeps staging
MATRIX = ("3.05:30:262144", "3.05:22:262144", "3.05:14.5:262144", "3.05:10.5:65536", "3.05:10.5:131072:nostage",
          "3.05:8:16384:nostage", "3.05:7:16384:nostage", "4.05:30:262144", "4.05:22:262144", "4.05:14.5:262144",
          "4.05:10.5:32768:nostage")
WORKLOADS = {
    "quick": {"answers": 2, "warmup": False, "document": False, "prefill": (8192, 32768), "reps": 1},
    "standard": {"answers": 5, "warmup": True, "document": True, "prefill": (8192, 16384, 32768, 65536), "reps": 3},
}
# the same prompts as the runs before this tool (research/vram-scaling), so answer speeds stay comparable
PROMPTS = (
    "Explain how a B+ tree handles insertions that split nodes, with a worked example.",
    "Write a Rust function that parses an RFC 3339 timestamp without external crates, with tests.",
    "Compare Raft and Paxos for a team choosing a consensus algorithm. Be concrete.",
    "Write a bash script that rotates logs older than 7 days, compresses them, and keeps 30 archives.",
    "Write a TypeScript React hook for debounced search with cancellation via AbortController.",
)
WARMUP = "Write a Python asyncio TCP port scanner with a concurrency limit, timeouts, and a CLI. Include docstrings."
NEEDLE = "violet-otter-42"
TOKENS_PER_HELPER = 22.5            # measured: 4,200 filler functions = 94,466 prompt tokens
SETTLE_S = 3                        # after stopping a server, before reading free VRAM


# ---------------------------------------------------------------------------------------------------------------- runs
def parse_run(spec):
    """'4.05:22:262144:nostage:ram=16' -> a run; ValueError when malformed."""
    parts = str(spec).strip().split(":")
    if len(parts) < 3:
        raise ValueError(f"bad run {spec!r}: use QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB] or a preset "
                         f"({', '.join(PRESETS)})")
    quant, vram, ctx, *flags = parts
    if quant not in sizing.QUANTS:
        raise ValueError(f"unknown quant {quant!r} (3.05 or 4.05)")
    vram_gb = None if vram == "card" else float(vram)
    if vram_gb is not None and not 4 <= vram_gb <= 512:
        raise ValueError(f"VRAM {vram!r} out of range (4-512 GB, or card)")
    if not ctx.isdigit() or not config.MINIMUM["context"] <= int(ctx) <= config.MAXIMUM["context"]:
        raise ValueError(f"context {ctx!r} out of range ({config.MINIMUM['context']}-{config.MAXIMUM['context']})")
    context, nostage, ram = int(ctx), False, None
    for f in flags:
        if f == "nostage":
            nostage = True
        elif f.startswith("ram=") and f[4:].isdigit() and 1 <= int(f[4:]) <= 1024:
            ram = int(f[4:])
        else:
            raise ValueError(f"unknown option {f!r} in {spec!r} (nostage, ram=GIB)")
    label = (f"{quant}-{'card' if vram_gb is None else f'{vram_gb:g}gb'}-{context // 1024}k"
             + ("-nostage" if nostage else "") + (f"-ram{ram}" if ram else ""))
    return {"quant": quant, "vram_gb": vram_gb, "context": context, "nostage": nostage, "ram": ram, "label": label}


def expand(specs, cfg, lean_gb=None):
    """Presets and run specs -> runs."""
    if isinstance(specs, str):
        specs = [s for s in re.split(r"[,\s]+", specs) if s]
    runs = []
    for s in specs:
        s = str(s).strip()
        if s == "current":
            runs.append(parse_run(f"{cfg['quant']}:card:{cfg['context']}"))
        elif s == "lean":
            if not lean_gb:
                raise ValueError("the lean profile's VRAM isn't known yet (no GPU sample)")
            runs.append(parse_run(f"{cfg['quant']}:{lean_gb:.1f}:{cfg['context']}:ram={int(cfg['lean_ram_gb'])}"))
        elif s == "quants":
            runs += [parse_run(f"{q}:card:{cfg['context']}") for q in ("3.05", "4.05")]
        elif s == "matrix":
            runs += [parse_run(x) for x in MATRIX]
        else:
            runs.append(parse_run(s))
    if not runs:
        raise ValueError("no runs given")
    if len(runs) > MAX_RUNS:
        raise ValueError(f"{len(runs)} runs; at most {MAX_RUNS} at a time")
    return runs


# -------------------------------------------------------------------------------------------------------------- engine
def random_ids(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def document(context):
    """A long code file with one fact in the middle, sized to the context (at most ~94K tokens)."""
    n = int(min(94000, context - 6000) / TOKENS_PER_HELPER)
    filler = "".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(n))
    half = len(filler) // 2
    doc = filler[:half] + f"\n# NOTE: the deployment password is '{NEEDLE}'.\n" + filler[half:]
    return doc + "\n\nWhat is the deployment password mentioned in the code? Answer in one line."


class Client:
    """HTTP to the engine under test, with the manager's key on every request."""

    def __init__(self, port, key):
        self.base = f"http://127.0.0.1:{port}"
        self.headers = {"Content-Type": "application/json", **({BACKEND_KEY_HEADER: key} if key else {})}

    def get(self, path, timeout=10):
        with urllib.request.urlopen(urllib.request.Request(self.base + path, headers=self.headers),
                                    timeout=timeout) as r:
            return json.loads(r.read())

    def events(self, path, payload, timeout=7200):
        """A streamed request's server-sent events, as (arrival time, event)."""
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                yield time.perf_counter(), json.loads(body)

    def read_prompt(self, n, seed):
        """Prompt reading in tokens/s: n fresh random tokens, timed to the first generated token. Leaving then ends
        the request (the server stops a job whose client is gone)."""
        t0 = time.perf_counter()
        ev = self.events("/generate", {"input_ids": random_ids(n, seed), "stream": True,
                                       "sampling_params": {"temperature": 0}})
        try:
            for t, _ in ev:
                return n / (t - t0)
        finally:
            ev.close()
        raise RuntimeError("the server streamed nothing")

    def answer(self, content):
        """One chat answer, streamed to its natural end (no max_tokens)."""
        body = {"model": "qwen3.8-flash-next", "stream": True, "messages": [{"role": "user", "content": content}]}
        t0 = time.perf_counter()
        first = last = usage = None
        text = []
        for t, ev in self.events("/v1/chat/completions", body):
            usage = ev.get("usage") or usage
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if d.get("reasoning_content") or d.get("content"):
                first = first or t
                last = t
                text.append(d.get("content") or "")
        n, p = int((usage or {}).get("completion_tokens") or 0), int((usage or {}).get("prompt_tokens") or 0)
        return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": round(first - t0, 3) if first else None,
                "decode_tps": round((n - 1) / (last - first), 2) if first and last and last > first and n > 1 else None,
                "wall_s": round(time.perf_counter() - t0, 1), "answer": "".join(text)}

    def engine_stats(self):
        """Where the expert picks came from, and the cache's size."""
        st = self.get("/stats")
        ec = (st.get("expert_cache") or [{}])[0]
        h = (st.get("nv2") or {}).get("host", {})
        picks = max(1, h.get("vram", 0) + h.get("ram", 0) + h.get("nvme", 0))
        return {"slots": ec.get("slots"), "arena_gb": ec.get("arena_gb"), "mem_peak_gb": ec.get("mem_peak_gb"),
                "hit_rate": ec.get("hit_rate"), "stage_overflow": (ec.get("stage") or {}).get("overflow"),
                "ram_slots": (st.get("nv2") or {}).get("ram_slots"),
                "picks_vram_pct": round(100 * h.get("vram", 0) / picks, 1),
                "picks_ram_pct": round(100 * h.get("ram", 0) / picks, 1),
                "picks_nvme_pct": round(100 * h.get("nvme", 0) / picks, 1)}


def machine_facts(hw, kind, display):
    """The machine and stack a run was measured on, from the manager's hardware check."""
    g = (hw.get("gpus") or [{}])[0]
    e, repo = hw.get("engine") or {}, hw.get("repo") or {}
    pcie = f"{g['pcie_gen_max']}.0 x{g['pcie_width_max']}" if g.get("pcie_gen_max") else None
    return {"host": {"gpu": g.get("name"), "pcie": pcie, "display": display, "platform": kind,
                     "ram_total_gb": (hw.get("ram") or {}).get("total_gb")},
            "stack": {"torch": e.get("torch"), "cuda": e.get("cuda_version"),
                      "driver": (hw.get("driver") or {}).get("version"),
                      "commit": (repo.get("head") or "")[:7] or None}}


# ------------------------------------------------------------------------------------------------------------- results
def load_records(folder):
    """Every run's JSON in a results folder."""
    out = []
    for p in sorted(Path(folder).glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def new_run_dir():
    """A results folder named after the date and time (with a suffix if one started in the same second)."""
    base, n = time.strftime("%Y%m%d-%H%M%S"), 1
    d = BENCH_DIR / base
    while d.exists():
        n += 1
        d = BENCH_DIR / f"{base}-{n}"
    d.mkdir(parents=True)
    return d


def history(limit=12):
    """Earlier result folders, newest first."""
    if not BENCH_DIR.is_dir():
        return []
    dirs = sorted((d for d in BENCH_DIR.iterdir() if d.is_dir() and RUN_DIR.match(d.name)), reverse=True)
    return [{"name": d.name, "runs": len(list(d.glob("*.json")))} for d in dirs[:limit]]


def previous_record(rec, skip=None):
    """The latest earlier run of the same configuration and workload on the same GPU."""
    if not BENCH_DIR.is_dir():
        return None
    for d in sorted((d for d in BENCH_DIR.iterdir() if d.is_dir() and RUN_DIR.match(d.name)), reverse=True):
        p = d / f"{rec['label']}.json"
        if d == skip or not p.is_file():
            continue
        try:
            old = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (not old.get("error") and old.get("workload") == rec.get("workload")
                and (old.get("host") or {}).get("gpu") == (rec.get("host") or {}).get("gpu")):
            return dict(old, _dir=d.name)
    return None


def compare(rec, old):
    """Percent change against an earlier run."""
    if not old:
        return None

    def pct(a, b):
        return round(100 * (a - b) / b, 1) if a and b else None
    prefill = {n: pct((v or {}).get("median"), ((old.get("prefill") or {}).get(n) or {}).get("median"))
               for n, v in (rec.get("prefill") or {}).items()}
    return {"run": old.get("_dir"), "date": old.get("date"), "stack": old.get("stack"),
            "decode_pct": pct(rec.get("decode_tps_median"), old.get("decode_tps_median")),
            "document_pct": pct((rec.get("long") or {}).get("prefill_tps"), (old.get("long") or {}).get("prefill_tps")),
            "prefill_pct": {n: v for n, v in prefill.items() if v is not None}}


def summary(r):
    """A run in a few figures, for the dashboard and the CLI."""
    e, long = r.get("engine") or {}, r.get("long") or {}
    return {"label": r.get("label"), "quant": r.get("quant"), "context": r.get("context"),
            "vram": "whole card" if uncapped(r) else f"{r.get('usable_gb'):g} GB", "ram_gib": r.get("ram_gib"),
            "decode": r.get("decode_tps_median"), "decode_min": r.get("decode_tps_min"),
            "ttft": r.get("short_ttft_median_s"), "startup_s": r.get("startup_s"),
            "model_vram_gb": r.get("model_vram_gb"),
            "prefill": {n: v.get("median") for n, v in (r.get("prefill") or {}).items() if v.get("median")},
            "document": ({"tps": round(long["prefill_tps"]), "tokens": long.get("prompt_tokens"),
                          "found": long.get("needle_found")} if long.get("prefill_tps") else None),
            "picks": (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f}"
                      if e.get("picks_vram_pct") is not None else None),
            "error": r.get("error"), "previous": r.get("previous"), "stack": r.get("stack"), "date": r.get("date")}


def uncapped(r):
    return r.get("method") == "uncapped card"


def card_class(usable):
    return next(name for floor, name in ((26, "32 GB"), (19, "24 GB"), (12.5, "16 GB"), (8.5, "12 GB"), (0, "8 GB"))
                if usable >= floor)


def fmt(v, spec="{:.1f}", none="-"):
    return none if v is None else spec.format(v)


def table(records):
    """The report table (markdown) of a results folder; also reads results from before this tool."""
    def order(r):
        vram = r["card_total_gb"] if uncapped(r) else r["usable_gb"]
        return r["quant"], uncapped(r), -vram, r["context"], r.get("ram_gib") or 0
    rows = sorted(records, key=order)
    out = ["| Card | Quant | VRAM for the model | Context | RAM cache | Prefill chunk | Cached experts "
           "| Expert picks VRAM / RAM / NVMe | Decode, median (min) | Prefill | First token | Startup |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e, plan, long = r.get("engine") or {}, r.get("plan") or {}, r.get("long") or {}
        nostage = r.get("nostage") or plan.get("staging") is False
        chunk = "1,536, no staging" if nostage else f"{plan.get('chunk') or 0:,}"
        card = f"{round(r['card_total_gb'] * 1e9 / 2**30)} GB, uncapped" if uncapped(r) else card_class(r["usable_gb"])
        vram = "whole card" if uncapped(r) else f"{r['usable_gb']:g} GB"
        head = (f"| {card} | {r['quant']} | {vram} | {r['context'] // 1024}K | {fmt(r.get('ram_gib'), '{} GiB')} | "
                f"{chunk if plan.get('ok') or r.get('nostage') else '-'} |")
        if not r.get("decode_tps_median") or r.get("error"):
            tail = r.get("log_tail") or ""
            when = " on the long prompt" if r.get("decode_tps_median") else " on the first request"
            why = ("ran out of VRAM" + when
                   if "VRAM LIMIT EXCEEDED" in tail else str(r.get("error") or "no result").splitlines()[0][:100])
            dec = f" (decode {fmt(r['decode_tps_median'])} tok/s before that)" if r.get("decode_tps_median") else ""
            out.append(f"{head} - | - | **fails**: {why}{dec} | - | - | {fmt(r.get('startup_s'), '{:.0f} s')} |")
            continue
        picks = (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f} %"
                 if e.get("picks_vram_pct") is not None else "-")
        prefill = (f"{long['prefill_tps']:,.0f} tok/s ({long['prompt_tokens'] // 1000}K prompt"
                   f"{'' if long.get('needle_found') else ', needle missed'})" if long.get("prefill_tps") else "-")
        out.append(f"{head} {e.get('slots') or 0:,} | {picks} | **{fmt(r['decode_tps_median'])}** "
                   f"({fmt(r.get('decode_tps_min'))}) tok/s | {prefill} | "
                   f"{fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | {fmt(r.get('startup_s'), '{:.0f} s')} |")
    return "\n".join(out)


def prefill_table(records):
    """Prompt reading by size (fresh random tokens), when the runs measured it."""
    sizes = sorted({int(n) for r in records for n, v in (r.get("prefill") or {}).items() if v.get("median")})
    if not sizes:
        return ""
    out = ["| Run | " + " | ".join(f"{n // 1024}K" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for r in records:
        p = r.get("prefill") or {}
        if any((p.get(str(n)) or {}).get("median") for n in sizes):
            out.append(f"| {r['label']} | " + " | ".join(fmt((p.get(str(n)) or {}).get("median"), "{:,.0f} tok/s")
                                                         for n in sizes) + " |")
    return "\n".join(out)


def measured_rows(records, source):
    """sizing.MEASURED entries for the completed runs (check gpu and ram describe the machine)."""
    out = []
    for r in records:
        long = r.get("long") or {}
        if r.get("error") or not r.get("decode_tps_median") or not long.get("prefill_tps"):
            continue
        h, plan = r.get("host") or {}, r.get("plan") or {}
        card = round(r["card_total_gb"] * 1e9 / 2**30) if uncapped(r) else int(card_class(r["usable_gb"]).split()[0])
        display = h.get("display") if uncapped(r) else card > 8 or r["usable_gb"] < 7.5
        ram = round(h["ram_total_gb"] * 1e9 / 2**30) if h.get("ram_total_gb") else None
        gpu = (h.get("gpu") or "?").replace("NVIDIA GeForce ", "")
        extra = ("" if uncapped(r) else "capped=True, ") + ("" if display is not False else "display=False, ") + \
                ("" if plan.get("staging", True) and not r.get("nostage") else "staging=False, ")
        out.append(f'    dict(card={card}, {extra}quant="{r["quant"]}", context={r["context"]}, '
                   f'ram_cache_gib={r.get("ram_gib")}, decode={r["decode_tps_median"]:.1f}, '
                   f'prefill={long["prefill_tps"]:.0f}, ram={ram}, '
                   f'platform="{h.get("platform", "linux")}", gpu="{gpu}", source="{source}"),')
    return "\n".join(out)


# -------------------------------------------------------------------------------------------------------------- runner
class Bench:
    """A benchmark inside the manager: its state, output lines and results."""

    def __init__(self, svc):
        self.svc = svc
        self.running = self.cancelled = False
        self.out = collections.deque(maxlen=3000)
        self.seq = 0
        self.runs, self.records = [], []
        self.workload = self.current = self.dir = self.task = None
        self.started = self.finished = None
        self.echo = None                       # also print each line (`qwenflash bench` without the service)
        last = history(1)
        if last:                               # show the latest results after a restart
            self.dir = BENCH_DIR / last[0]["name"]
            self.records = load_records(self.dir)

    def emit(self, line):
        self.seq += 1
        self.out.append((self.seq, line))
        log.info("bench: %s", line)
        if self.echo:
            self.echo(line)

    def report(self, since=0, name=None):
        records = self.records
        if name and name != (self.dir.name if self.dir else None):    # an earlier run, by its folder name
            records = load_records(BENCH_DIR / name) if RUN_DIR.match(name) and (BENCH_DIR / name).is_dir() else []
        return {"running": self.running, "workload": self.workload, "current": self.current,
                "runs": [r["label"] for r in self.runs], "dir": self.dir.name if self.dir else None,
                "results": [summary(r) for r in records], "records": records,
                "started": self.started, "finished": self.finished, "seq": self.seq,
                "lines": [[n, t] for n, t in self.out if n > since][-800:],
                "presets": PRESETS, "workloads": list(WORKLOADS), "history": history()}

    def start(self, specs, workload="standard"):
        """Check the request and start the benchmark in the background: (ok, message)."""
        svc = self.svc
        if self.running:
            return False, "a benchmark is already running"
        if svc.observe:
            return False, "observe mode: this manager does not control the model"
        if svc.setup.running:
            return False, f"setup step '{svc.setup.running}' is running; try again when it finishes"
        if workload not in WORKLOADS:
            return False, f"unknown workload {workload!r} ({', '.join(WORKLOADS)})"
        m = svc.mon.get()
        if not m:
            return False, "no GPU reading yet; try again in a moment"
        try:
            runs = expand(specs, svc.cfg, lean_gb=min(svc.lean_budget(m), svc.full_budget(m)) / GB)
        except ValueError as e:
            return False, str(e)
        self.running, self.cancelled = True, False
        self.runs, self.records, self.workload = runs, [], workload
        self.started, self.finished, self.current = time.time(), None, None
        self.out.clear()
        self.task = asyncio.create_task(self._run())
        return True, f"benchmark started: {len(runs)} run{'s' if len(runs) > 1 else ''}, {workload} workload"

    async def cancel(self):
        if not self.running:
            return False, "no benchmark is running"
        self.cancelled = True
        self.emit("cancelling: stopping the benchmark's model server")
        await asyncio.to_thread(self.svc.host.backend_stop)
        return True, "cancelling"

    async def _run(self):
        svc = self.svc
        before = svc.be.tier if svc.be.state in ("running", "starting") else None
        try:
            self.dir = new_run_dir()
            await svc.setup.check()                       # GPU, driver, CUDA, PyTorch and commit for the records
            facts = machine_facts(svc.setup.hw or {}, svc.host.kind, svc.gpu_display)
            st = facts["stack"]
            self.emit(f"{len(self.runs)} run(s), {self.workload} workload, on {facts['host']['gpu']} "
                      f"(driver {st['driver']}, CUDA {st['cuda']}, PyTorch {st['torch']}, commit {st['commit']})")
            if svc.be.state != "stopped":
                self.emit("stopping the model for the benchmark; requests get an error until it's done")
                await svc.stop("benchmark", "benchmark")
            for r in self.runs:
                if self.cancelled:
                    break
                self.current = r["label"]
                self.records.append(await self._one(r, facts))
        except Exception as e:  # noqa: BLE001 - report it, then restore the service
            log.exception("benchmark failed")
            self.emit(f"benchmark failed: {e}")
        finally:
            self.current = None
            await asyncio.to_thread(svc.host.backend_stop)
            self.running, self.finished = False, time.time()
            done = sum(1 for r in self.records if not r.get("error"))
            self.emit(f"{'cancelled' if self.cancelled else 'finished'}: {done} of {len(self.runs)} run(s) completed; "
                      f"results in {self.dir}")
            if before:
                self.emit(f"starting the model again ({before})")
                await svc.start(before, "benchmark finished")

    def engine_env(self, r, plan, m, budget):
        """What the manager would pass the engine for this run (a capped run also gets the emulated card's guard)."""
        svc, cfg = self.svc, self.svc.cfg
        need_all = plan["ram_for_all_experts"] if plan["ok"] else sizing.expert_bytes(r["quant"])
        if r["ram"]:
            ram_gib = r["ram"]
        else:
            caps = [c for c in (cfg["max_ram_cache_gib"],
                                svc.learned.get(svc.learned_key(r["quant"], r["context"]), {}).get("ram_cap")) if c]
            ram_gib = sizing.ram_cache_gib(need_all, m["ram_avail"], cfg, svc.ram_overhead(r["quant"]),
                                           min(caps) if caps else None)
        env = {"QUANT": r["quant"], "CONTEXT": str(r["context"]), "PORT": str(cfg["backend_port"]),
               "HOST": "127.0.0.1", "RAM_GB": str(ram_gib), "QWENFLASH_BACKEND_KEY": svc.backend_key}
        if r["vram_gb"] is None:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_total"] - svc.headroom(m)))
            env["QWEN_VRAM_GUARD_BYTES"] = str(int(m["vram_total"] + 2 * GB))
        else:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_used"] + budget))
            # past this the server exits, as running out of memory would on the emulated card (slack for the desktop)
            env["QWEN_VRAM_GUARD_BYTES"] = str(int(m["vram_used"] + budget + 0.25 * GB))
        if r["nostage"]:
            env.update({"QWEN_CHUNK": "1536", "QWEN38_EC_STAGE_GB": "0", "QWEN38_EC_ELASTIC_GB": "0",
                        "QWEN38_EC_RESERVE_GB": f"{sizing.nostage_reserve(r['quant'], 1536) / GIB:.2f}"})
        else:
            env.update(sizing.engine_env(plan))
        return env, ram_gib

    async def _one(self, r, facts):
        svc, cfg = self.svc, self.svc.cfg
        self.emit(f"=== {r['label']}")
        await asyncio.to_thread(svc.host.backend_stop)
        await asyncio.sleep(SETTLE_S)
        m = await svc.fresh_sample(time.time())
        card = r["vram_gb"] is None
        budget = m["vram_total"] - m["vram_used"] - svc.headroom(m) if card else r["vram_gb"] * GB
        plan = sizing.plan(r["quant"], budget, r["context"])
        rec = {"label": r["label"], "quant": r["quant"], "usable_gb": round(budget / GB, 2), "context": r["context"],
               "nostage": r["nostage"], "workload": self.workload, "date": time.strftime("%Y-%m-%d %H:%M"),
               "device_used_before_gb": round(m["vram_used"] / GB, 2), "card_total_gb": round(m["vram_total"] / GB, 2),
               "method": "uncapped card" if card else "engine cap (limit + guard)", **facts,
               "plan": {k: v for k, v in plan.items() if isinstance(v, (int, float, str, bool)) or v is None}}
        path = self.dir / f"{r['label']}.json"

        def save():
            path.write_text(json.dumps(rec, indent=1))
        have = (svc.setup.hw or {}).get("engine") or {}
        if have.get("driver") and have.get("m305" if r["quant"] == "3.05" else "m405") != "1":
            rec["error"] = f"the {r['quant']} bpw model isn't downloaded (Setup tab)"
        elif not plan["ok"] and not r["nostage"]:
            rec["error"] = f"does not fit: {plan.get('reason')}"
        if rec.get("error"):
            self.emit(f"  skipped: {rec['error']}")
            save()
            return rec
        env, rec["ram_gib"] = self.engine_env(r, plan, m, budget)
        t0 = time.time()
        ok, pid = await asyncio.to_thread(svc.host.backend_start, env, ["server"])
        up = False
        if ok:
            url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
            while time.time() - t0 < cfg["start_timeout_s"] and not self.cancelled:
                if await asyncio.to_thread(http_ok, url, 2.0, svc.backend_auth):
                    up = True
                    break
                st, _ = await asyncio.to_thread(svc.host.backend_status)
                if st != "running":
                    break
                await asyncio.sleep(2)
        rec["startup_s"] = round(time.time() - t0, 1)
        if up:
            m2 = await svc.fresh_sample(time.time())
            rec["model_vram_gb"] = round((m2["vram_used"] - m["vram_used"]) / GB, 2)
            self.emit(f"  ready in {rec['startup_s']:.0f} s: {rec['model_vram_gb']:.1f} GB VRAM, "
                      f"{rec['ram_gib']} GiB RAM cache")
            save()
            client = Client(cfg["backend_port"], svc.backend_key)
            try:
                await self.workload_steps(client, r, rec, save)
                rec["engine"] = await asyncio.to_thread(client.engine_stats)
            except Exception as e:  # noqa: BLE001 - one failed run must not end the others
                rec["error"] = "cancelled" if self.cancelled else f"workload failed: {e}"
            st, _ = await asyncio.to_thread(svc.host.backend_status)
            if st != "running" and not rec.get("error"):
                rec["error"] = "the server exited during the workload"
        else:
            rec["error"] = ("cancelled" if self.cancelled else
                            f"the server did not come up{'' if ok else f' (launcher: {pid})'}")
        if rec.get("error") and svc.host.follower:
            rec["log_tail"] = svc.host.follower.tail(8)
        await asyncio.to_thread(svc.host.backend_stop)
        if not rec.get("error"):
            rec["previous"] = compare(rec, previous_record(rec, skip=self.dir))
        save()
        s = summary(rec)
        if rec.get("error"):
            self.emit(f"  failed: {rec['error']}")
        else:
            doc = s["document"]
            self.emit(f"  answers {fmt(s['decode'])} tok/s (min {fmt(s['decode_min'])})"
                      + (f", document {doc['tps']:,} tok/s ({doc['tokens'] // 1000}K, fact "
                         f"{'found' if doc['found'] else 'MISSED'})" if doc else "")
                      + "".join(f", {int(n) // 1024}K {v:,.0f}" for n, v in s["prefill"].items())
                      + (f"; vs {rec['previous']['date']}: answers {rec['previous']['decode_pct']:+.1f}%"
                         if rec.get("previous") and rec["previous"].get("decode_pct") is not None else ""))
        return rec

    async def workload_steps(self, client, r, rec, save):
        wl = WORKLOADS[self.workload]

        def check():
            if self.cancelled:
                raise RuntimeError("cancelled")
        if wl["warmup"]:
            await asyncio.to_thread(client.answer, WARMUP)            # not scored
        runs = []
        for p in PROMPTS[:wl["answers"]]:
            check()
            a = await asyncio.to_thread(client.answer, p)
            a.pop("answer")
            runs.append(a)
            self.emit(f"  answer {len(runs)}: {a['completion_tokens']:,} tokens at {fmt(a['decode_tps'])} tok/s")
            rec["decode_runs"] = runs
            save()
        rates = [a["decode_tps"] for a in runs if a["decode_tps"]]
        rec["decode_tps_median"] = round(statistics.median(rates), 2) if rates else None
        rec["decode_tps_min"] = min(rates) if rates else None
        ttfts = [a["ttft_s"] for a in runs if a["ttft_s"]]
        rec["short_ttft_median_s"] = round(statistics.median(ttfts), 3) if ttfts else None
        if wl["document"]:
            check()
            a = await asyncio.to_thread(client.answer, document(r["context"]))
            a["needle_found"] = NEEDLE in a.pop("answer")
            a["prefill_tps"] = round(a["prompt_tokens"] / a["ttft_s"], 1) if a["ttft_s"] else None
            rec["long"] = a
            self.emit(f"  document: {a['prompt_tokens']:,} tokens read at {fmt(a['prefill_tps'], '{:,.0f}')} tok/s, "
                      f"fact {'found' if a['needle_found'] else 'MISSED'}")
            save()
        seed = int(time.time())
        rec["prefill"] = {}
        for n in wl["prefill"]:
            if n + 1024 > r["context"]:
                continue
            rates = []
            for i in range(wl["reps"]):
                check()
                rates.append(await asyncio.to_thread(client.read_prompt, n, seed + 1000 * n + i))
            rec["prefill"][str(n)] = {"median": round(statistics.median(rates), 1),
                                      "runs": [round(x, 1) for x in rates]}
            self.emit(f"  prompt reading {n // 1024}K: {statistics.median(rates):,.0f} tok/s")
            save()

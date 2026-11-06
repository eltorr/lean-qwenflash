"""Benchmarks: start the engine with a given configuration, run a fixed workload against it, and record the results.

One tool for the dashboard's Benchmark tab, `qwenflash bench`, and the runs behind the README's tables (research/).
It takes the GPU: the manager stops the model first, refuses requests meanwhile, and puts the model back afterwards.

Runs are QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB], or a preset:
  current  the current settings on the whole card, as the full profile runs them
  lean     the current settings in the lean profile's VRAM and RAM
  quants   3.05 and 4.05 bpw at the current context, on the whole card
  matrix   8-32 GB cards emulated on this one: the README's measured table (needs a 32 GB card and 64 GB of RAM;
           every run gets a 38 GiB RAM cache, so runs differ only in VRAM)
VRAM_GB holds the engine to that much VRAM, as on a smaller card after the desktop's share: QWEN_VRAM_LIMIT_BYTES sizes
the expert cache, and QWEN_VRAM_GUARD_BYTES stops the server past it, as running out of memory would on that card. A
smaller card's bandwidth can't be emulated, so real smaller cards are slower. ":nostage" forces staging-free prefill
(1,536-token chunks); ":ram=GIB" sets the RAM cache (default: what the full profile would get now).

Workloads follow the original's benchmark.py (MaxDevv), so results compare with his: thinking off, greedy decoding,
his prompts, and every answer to its natural end (output is never capped; his answers end after ~250 tokens). A warm-up
first (his "17 times 23", then one unscored trial, so cold caches don't count), then trials of one answer ("explain how
a jet engine works in three concise paragraphs") and documents of his filler text, each read and answered in one
sentence. Each trial's text differs from the start, so nothing comes from the prefix cache.
  quick     1 trial: the answer, documents of 1K and 8K tokens                                   ~2 min a run
  standard  3 trials: the answer, documents of 1K, 8K, 32K and 64K tokens; then his retrieval
            test: three access codes at 10%, 50% and 90% of a document of up to 100K tokens       ~3-6 min a run
Documents that don't fit the run's context are skipped; the retrieval document is sized to it.

The PC comes first. When a game or another app needs the GPU (a watched app, another app's GPU load, or its VRAM
crowding the card) or RAM nearly runs out, the run steps aside: its server stops, the benchmark waits until the PC has
been free for `recover_after_s`, then does that run again from the start. Finished runs are kept. A benchmark cut short
(cancelled, the manager restarted, the PC rebooted) continues with `qwenflash bench --resume` or the tab's Resume
button: the runs left go into the same folder.

Results: one JSON per run in the settings folder (bench/<date-time>/), saved after every step, with the machine and
its stack (GPU, driver, CUDA, PyTorch, commit) and the change since the last run of the same configuration. The
folder's _plan.json lists the benchmark's runs, for resuming.
"""
import asyncio
import collections
import json
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
# the README's measured table: card sizes after the desktop's share, each at the largest context that keeps staging.
# Every run gets the same RAM cache, as the earlier matrix (research/vram-scaling): sized from the RAM free when a run
# starts, it varied with how much of the previous run's memory WSL had handed back yet (8-25 GiB a run).
MATRIX_RAM = 38
MATRIX = tuple(f"{s}:ram={MATRIX_RAM}" for s in (
    "3.05:30:262144", "3.05:22:262144", "3.05:14.5:262144", "3.05:10.5:65536", "3.05:10.5:131072:nostage",
    "3.05:8:16384:nostage", "3.05:7:16384:nostage", "4.05:30:262144", "4.05:22:262144", "4.05:14.5:262144",
    "4.05:10.5:32768:nostage"))
WORKLOADS = {
    "quick": {"trials": 1, "documents": (1024, 8192), "retrieval": False},
    "standard": {"trials": 3, "documents": (1024, 8192, 32768, 65536), "retrieval": True},
}
# The original's benchmark.py (MaxDevv): its prompts, filler text and retrieval test, word for word
PROTOCOL = "benchmark.py"           # recorded with each run; runs are compared only with runs of the same protocol
WARMUP = ("Compute 17 times 23. Reply with only the number.",
          "Trial warm-up: Explain how a jet engine works in three concise paragraphs.")
ANSWER = "Trial {t}: Explain how a jet engine works in three concise paragraphs."
DOC_HEAD = "Trial {t}, document {size}: Read the following document."
DOC_TAIL = "Now write exactly one sentence explaining why the sky is blue."
FILLER = "The fox observes the river, while the engineer records temperature and pressure. "
CODES = ((0.1, "ARCTIC", "739164"), (0.5, "DESERT", "285903"), (0.9, "FOREST", "614827"))
RETRIEVAL_HEAD = "Read the following document carefully and retain its access codes."
RETRIEVAL_TAIL = ("What are the secret access codes for ARCTIC, DESERT, and FOREST? Reply only with the three codes in "
                  "that order.")
RETRIEVAL_MAX = 100000              # the longest context in the original's README table, short of 262K
SETTLE_S = 3                        # after stopping a server, before reading free VRAM
WATCH_S = 1.0                       # how often a run checks whether the PC needs the GPU or RAM back
YIELD_AFTER_S = 10                  # how long the PC must need them before the run steps aside
MAX_YIELDS = 3                      # a run interrupted more often is recorded as interrupted (resume redoes it)
RAM_FLOOR_GB = 1.0                  # free RAM below this: the PC is about to run out
PLAN = "_plan.json"                 # the benchmark's runs and workload, for resuming (not a result)


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
    return {"quant": quant, "vram_gb": vram_gb, "context": context, "nostage": nostage, "ram": ram, "label": label,
            "spec": ":".join(parts)}


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
class Client:
    """HTTP to the engine under test, with the manager's key on every request."""

    def __init__(self, port, key):
        self.base = f"http://127.0.0.1:{port}"
        self.headers = {"Content-Type": "application/json", **({BACKEND_KEY_HEADER: key} if key else {})}

    def get(self, path, timeout=10):
        with urllib.request.urlopen(urllib.request.Request(self.base + path, headers=self.headers),
                                    timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path, payload, timeout=60):
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tokens(self, text):
        return len(self.post("/tokenize", {"prompt": text})["tokens"])

    def document(self, head, size, tail, codes=False):
        """The original's document: head, its filler text to `size` tokens less 40, tail. codes: its retrieval test,
        an access code in place of the filler at 10%, 50% and 90%."""
        if not getattr(self, "per_sentence", None):
            self.per_sentence = (self.tokens(FILLER * 200) - self.tokens(FILLER * 100)) / 100
        n = max(10, int((size - 40 - self.tokens(head + "\n")) / self.per_sentence))
        body = [FILLER] * n
        for frac, name, value in codes and CODES or ():
            body[int(n * frac)] = f"\nThe secret access code for {name} is {value}.\n"
        return head + "\n" + "".join(body) + "\n" + tail

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

    def answer(self, content):
        """One chat answer as the original's benchmark.py asks for it: thinking off, greedy, streamed to its natural
        end (no max_tokens)."""
        body = {"model": "qwen3.8-flash-next", "stream": True, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
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
        ttft = first - t0 if first else None
        return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": round(ttft, 3) if ttft is not None else None,
                # from the unrounded time: a fast first token rounds to 0.000 s
                "prefill_tps": round(p / max(ttft, 1e-6), 1) if ttft is not None and p else None,
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
def record_files(folder):
    """The runs' JSON files in a results folder (not its _plan.json)."""
    return [p for p in sorted(Path(folder).glob("*.json")) if not p.name.startswith("_")]


def load_records(folder):
    """Every run's JSON in a results folder."""
    out = []
    for p in record_files(folder):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("label"):
            out.append(rec)
    return out


def counts(rec):
    """A run that needs no redoing: it finished, and wasn't cancelled or interrupted (a failure counts: it would fail
    again)."""
    return bool(rec.get("finished")) and not rec.get("interrupted") and rec.get("error") != "cancelled"


def unfinished():
    """The latest benchmark, if it has runs left: (folder, plan, runs left), else None."""
    for h in history(50):
        d = BENCH_DIR / h["name"]
        try:
            plan = json.loads((d / PLAN).read_text())
        except (OSError, ValueError):
            continue                                   # from before resuming existed
        if plan.get("protocol") != PROTOCOL:
            return None                                # measured another way: finishing it would mix the two
        done = {r["label"] for r in load_records(d) if counts(r)}
        left = [r for r in plan.get("runs") or [] if isinstance(r, dict) and r.get("label") not in done]
        return (d, plan, left) if left else None
    return None


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
    return [{"name": d.name, "runs": len(record_files(d))} for d in dirs[:limit]]


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
                and old.get("protocol") == rec.get("protocol")
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
                          "found": long.get("codes_found", long.get("needle_found"))}
                         if long.get("prefill_tps") else None),
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
                   f"{'' if long.get('codes_found', long.get('needle_found')) else ', missed what it had to find'})"
                   if long.get("prefill_tps") else "-")
        out.append(f"{head} {e.get('slots') or 0:,} | {picks} | **{fmt(r['decode_tps_median'])}** "
                   f"({fmt(r.get('decode_tps_min'))}) tok/s | {prefill} | "
                   f"{fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | {fmt(r.get('startup_s'), '{:.0f} s')} |")
    return "\n".join(out)


def prefill_table(records):
    """Document reading by size, when the runs measured it."""
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
        self.planned = 0
        self.resume_dir = None                 # the folder of the benchmark being continued
        self.base = None                       # the GPU just before this run's server started: other apps' baseline
        self.yielded = self.waiting = None     # why this run stepped aside; what the PC still needs
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
        left = None if self.running else unfinished()
        return {"running": self.running, "workload": self.workload, "current": self.current,
                "runs": [r["label"] for r in self.runs], "dir": self.dir.name if self.dir else None,
                "waiting": self.waiting,
                "resumable": {"dir": left[0].name, "left": len(left[2]), "workload": left[1].get("workload")}
                if left else None,
                "results": [summary(r) for r in records], "records": records,
                "started": self.started, "finished": self.finished, "seq": self.seq,
                "lines": [[n, t] for n, t in self.out if n > since][-800:],
                "presets": PRESETS, "workloads": list(WORKLOADS), "history": history()}

    def start(self, specs, workload="standard", resume=False):
        """Check the request and start the benchmark in the background: (ok, message). resume: continue the latest
        unfinished benchmark instead (its runs left, into its folder, with its workload)."""
        svc = self.svc
        if self.running:
            return False, "a benchmark is already running"
        if svc.observe:
            return False, "observe mode: this manager does not control the model"
        if svc.setup.running:
            return False, f"setup step '{svc.setup.running}' is running; try again when it finishes"
        m = svc.mon.get()
        if not m:
            return False, "no GPU reading yet; try again in a moment"
        folder, done = None, []
        if resume:
            left = unfinished()
            if not left:
                return False, "no unfinished benchmark to resume"
            folder, plan, todo = left
            workload = plan.get("workload")
            try:
                runs = [parse_run(r["spec"]) for r in todo]
            except (KeyError, TypeError, ValueError) as e:
                return False, f"can't resume {folder.name}: {e}"
            labels = {r["label"] for r in runs}
            done = [r for r in load_records(folder) if r["label"] not in labels]
        if workload not in WORKLOADS:
            return False, f"unknown workload {workload!r} ({', '.join(WORKLOADS)})"
        if not resume:
            try:
                runs = expand(specs, svc.cfg, lean_gb=min(svc.lean_budget(m), svc.full_budget(m)) / GB)
            except ValueError as e:
                return False, str(e)
        self.running, self.cancelled = True, False
        self.runs, self.records, self.workload, self.resume_dir = runs, done, workload, folder
        self.planned = len(done) + len(runs)
        self.started, self.finished, self.current = time.time(), None, None
        self.out.clear()
        self.task = asyncio.create_task(self._run())
        n = f"{len(runs)} run{'s' if len(runs) > 1 else ''}"
        if folder:
            return True, f"resuming the benchmark of {folder.name}: {n} left, {workload} workload"
        return True, f"benchmark started: {n}, {workload} workload"

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
            self.dir = self.resume_dir or new_run_dir()
            if not self.resume_dir:
                (self.dir / PLAN).write_text(json.dumps({"workload": self.workload, "protocol": PROTOCOL,
                                                            "runs": self.runs}, indent=1))
            await svc.setup.check()                       # GPU, driver, CUDA, PyTorch and commit for the records
            facts = machine_facts(svc.setup.hw or {}, svc.host.kind, svc.gpu_display)
            st = facts["stack"]
            self.emit((f"resuming {self.dir.name} ({self.planned - len(self.runs)} of {self.planned} done): "
                       if self.resume_dir else "")
                      + f"{len(self.runs)} run(s), {self.workload} workload, on {facts['host']['gpu']} "
                        f"(driver {st['driver']}, CUDA {st['cuda']}, PyTorch {st['torch']}, commit {st['commit']})")
            if svc.be.state != "stopped":
                self.emit("stopping the model for the benchmark; requests get an error until it's done")
                await svc.stop("benchmark", "benchmark")
            for r in self.runs:
                if self.cancelled:
                    break
                self.current = r["label"]
                for tries in range(1, MAX_YIELDS + 2):
                    rec = await self._one(r, facts)
                    if not rec.get("interrupted") or self.cancelled:
                        break
                    if tries > MAX_YIELDS:
                        self.emit(f"  interrupted {tries} times; moving on (resuming the benchmark redoes it)")
                        break
                    await self.wait_for_pc()
                self.records.append(rec)
        except Exception as e:  # noqa: BLE001 - report it, then restore the service
            log.exception("benchmark failed")
            self.emit(f"benchmark failed: {e}")
        finally:
            self.current = self.base = self.waiting = None
            await asyncio.to_thread(svc.host.backend_stop)
            self.running, self.finished = False, time.time()
            done = sum(1 for r in self.records if not r.get("error"))
            left = unfinished()
            self.emit(f"{'cancelled' if self.cancelled else 'finished'}: {done} of {self.planned} run(s) completed; "
                      f"results in {self.dir}"
                      + (f"; {len(left[2])} left: `qwenflash bench --resume` or the Resume button runs them"
                         if left and left[0] == self.dir else ""))
            if before:
                self.emit(f"starting the model again ({before})")
                await svc.start(before, "benchmark finished")

    def why_stopped(self, default):
        if self.cancelled:
            return "cancelled"
        if self.yielded:
            return "interrupted: " + "; ".join(self.yielded)
        return default

    def disturbed(self, m):
        """Why the PC needs the GPU or RAM back now ([] when it doesn't). The benchmark's own server never counts: on
        Windows the per-app counters can't see WSL, and on Linux its processes are left out."""
        cfg = self.svc.cfg
        why = []
        hits = sorted({w.lower() for w in cfg["watchlist"]} & set(m.get("running") or ()))
        if hits:
            why.append("watched app running: " + ", ".join(hits))
        if m.get("ext_util", 0) >= cfg["ext_util_pct"]:
            top = m.get("ext_util_top") or []
            why.append(f"GPU busy: {top[0][0] if top else 'another app'} at {m['ext_util']:.0f}%")
        if self.base is not None:
            grew = self.svc.ext_vram(m) - self.base["ext"]
            free = m["vram_total"] - m["vram_used"]
            if grew > 0.75 * GB and free < cfg["vram_min_free_gb"] * GB:
                why.append(f"other apps need VRAM (+{grew / GB:.1f} GB, {free / GB:.1f} GB free)")
        if m["ram_avail"] < RAM_FLOOR_GB * GB:
            why.append(f"RAM nearly out ({m['ram_avail'] / GB:.1f} GB free)")
        return why

    async def watch(self):
        """While a run's server is up: once the PC has needed the GPU or RAM for YIELD_AFTER_S, stop the server. The
        run is done again when the PC is free."""
        since = None
        while True:
            await asyncio.sleep(WATCH_S)
            m = self.svc.mon.get()
            why = self.disturbed(m) if m else []
            if not why:
                since = None
                continue
            since = since or time.time()
            if time.time() - since >= YIELD_AFTER_S:
                self.yielded = why
                self.emit("  stepping aside: " + "; ".join(why))
                await asyncio.to_thread(self.svc.host.backend_stop)
                return

    async def wait_for_pc(self):
        """After stepping aside: wait until the PC has been free for recover_after_s, or the benchmark is cancelled."""
        need = self.svc.cfg["recover_after_s"]
        self.waiting = self.yielded
        self.emit(f"  waiting until the PC has been free for {need:g} s, then running {self.current} again")
        clear = None
        while not self.cancelled:
            await asyncio.sleep(WATCH_S)
            m = self.svc.mon.get()
            why = self.disturbed(m) if m else ["no GPU reading"]
            if why:
                clear, self.waiting = None, why
                continue
            clear = clear or time.time()
            if time.time() - clear >= need:
                self.emit("  the PC is free again")
                break
        self.waiting = None

    def engine_env(self, r, plan, m, budget):
        """What the manager would pass the engine for this run (a capped run also gets the emulated card's guard)."""
        svc, cfg = self.svc, self.svc.cfg
        target = plan["ram_target"] if plan["ok"] else sizing.RAM_COVERAGE * sizing.expert_bytes(r["quant"])
        if r["ram"]:
            ram_gib = r["ram"]
        else:
            caps = [c for c in (cfg["max_ram_cache_gib"],
                                svc.learned.get(svc.learned_key(r["quant"], r["context"]), {}).get("ram_cap")) if c]
            ram_gib = sizing.ram_cache_gib(target, svc.ram_for_start(m), cfg, svc.ram_overhead(r["quant"]),
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
        self.yielded = None
        await asyncio.to_thread(svc.host.backend_stop)
        await asyncio.sleep(SETTLE_S)
        m = await svc.fresh_sample(time.time())
        self.base = {"vram_used": m["vram_used"], "perproc_ded": m.get("perproc_ded", 0.0)}
        self.base["ext"] = svc.ext_vram(m)
        card = r["vram_gb"] is None
        budget = m["vram_total"] - m["vram_used"] - svc.headroom(m) if card else r["vram_gb"] * GB
        plan = sizing.plan(r["quant"], budget, r["context"])
        rec = {"label": r["label"], "quant": r["quant"], "usable_gb": round(budget / GB, 2), "context": r["context"],
               "nostage": r["nostage"], "workload": self.workload, "protocol": PROTOCOL,
               "date": time.strftime("%Y-%m-%d %H:%M"),
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
            rec["finished"] = True
            save()
            return rec
        env, rec["ram_gib"] = self.engine_env(r, plan, m, budget)
        t0 = time.time()
        ok, pid = await asyncio.to_thread(svc.host.backend_start, env, ["server"])
        watcher = asyncio.create_task(self.watch()) if ok else None
        try:
            up = False
            if ok:
                url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
                while time.time() - t0 < cfg["start_timeout_s"] and not self.cancelled and not self.yielded:
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
                    rec["error"] = self.why_stopped(f"workload failed: {e}")
                st, _ = await asyncio.to_thread(svc.host.backend_status)
                if st != "running" and not rec.get("error"):
                    rec["error"] = "the server exited during the workload"
            else:
                rec["error"] = self.why_stopped(f"the server did not come up{'' if ok else f' (launcher: {pid})'}")
        finally:
            if watcher:
                watcher.cancel()
        if self.cancelled or self.yielded:                # whatever it managed, a run cut short isn't a result
            rec["error"] = self.why_stopped("")
            rec["interrupted"] = not self.cancelled
        elif rec.get("error") and svc.host.follower:
            rec["log_tail"] = svc.host.follower.tail(8)
        await asyncio.to_thread(svc.host.backend_stop)
        # the next run, or the model afterwards, can reuse this server's RAM before WSL has handed it back
        svc.last_stop_t = time.time()
        svc.freed = (svc.last_stop_t, svc.ram_for_start(m), 0.0)
        if not rec.get("error"):
            rec["previous"] = compare(rec, previous_record(rec, skip=self.dir))
        rec["finished"] = True
        save()
        s = summary(rec)
        if rec.get("interrupted"):
            pass                                          # stepping aside was reported; _run decides what's next
        elif rec.get("error"):
            self.emit(f"  failed: {rec['error']}")
        else:
            doc = s["document"]
            self.emit(f"  answers {fmt(s['decode'])} tok/s (min {fmt(s['decode_min'])})"
                      + (f", long document {doc['tps']:,} tok/s ({doc['tokens'] // 1000}K, codes "
                         f"{'found' if doc['found'] else 'MISSED'})" if doc else "")
                      + "".join(f", {int(n) // 1024}K {v:,.0f}" for n, v in s["prefill"].items())
                      + (f"; vs {rec['previous']['date']}: answers {rec['previous']['decode_pct']:+.1f}%"
                         if rec.get("previous") and rec["previous"].get("decode_pct") is not None else ""))
        return rec

    async def workload_steps(self, client, r, rec, save):
        """The original's benchmark.py over HTTP: the warm-up, then each trial's answer and documents, then the
        retrieval test."""
        wl = WORKLOADS[self.workload]

        def check():
            if self.cancelled or self.yielded:
                raise RuntimeError(self.why_stopped("stopped"))

        async def ask(text):
            check()
            return await asyncio.to_thread(client.answer, text)
        for text in WARMUP:                                           # not scored
            await ask(text)
        answers, docs = [], {}
        for trial in range(wl["trials"]):
            a = await ask(ANSWER.format(t=trial))
            a.pop("answer")
            answers.append(a)
            rec["decode_runs"] = answers
            save()
            read = []
            for n in wl["documents"]:
                if n + 1024 > r["context"]:
                    continue
                text = await asyncio.to_thread(client.document, DOC_HEAD.format(t=trial, size=n), n, DOC_TAIL)
                d = await ask(text)
                docs.setdefault(str(n), []).append(d["prefill_tps"])
                rec["prefill"] = {k: {"median": statistics.median(v), "runs": v} for k, v in docs.items() if all(v)}
                save()
                read.append(f"{n // 1024}K {fmt(d['prefill_tps'], '{:,.0f}')}")
            self.emit(f"  trial {trial + 1}: answer of {a['completion_tokens']:,} tokens at "
                      f"{fmt(a['decode_tps'])} tok/s"
                      + (f"; documents read at {', '.join(read)} tok/s" if read else ""))
        rates = [a["decode_tps"] for a in answers if a["decode_tps"]]
        rec["decode_tps_median"] = round(statistics.median(rates), 2) if rates else None
        rec["decode_tps_min"] = min(rates) if rates else None
        ttfts = [a["ttft_s"] for a in answers if a["ttft_s"] is not None]
        rec["short_ttft_median_s"] = round(statistics.median(ttfts), 3) if ttfts else None
        save()
        if wl["retrieval"]:
            n = min(RETRIEVAL_MAX, r["context"] - 2048)
            text = await asyncio.to_thread(client.document, RETRIEVAL_HEAD, n, RETRIEVAL_TAIL, True)
            a = await ask(text)
            said = a.pop("answer")
            a["codes_found"] = all(code in said for _, _, code in CODES)
            rec["long"] = a
            self.emit(f"  long document: {a['prompt_tokens']:,} tokens read at {fmt(a['prefill_tps'], '{:,.0f}')} "
                      f"tok/s, {'all three codes found' if a['codes_found'] else 'codes MISSED'}")
            save()

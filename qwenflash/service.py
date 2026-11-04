"""QwenFlash service: OpenAI-compatible front door, resource-aware model lifecycle, dashboard and control API.

Clients use http://127.0.0.1:30000/v1 (this process). The engine's server runs on the backend port behind it and
is started, resized or stopped depending on what other apps are doing:

  full    the biggest VRAM budget that is free (minus headroom): fastest
  lean    a smaller budget that coexists with games and other GPU apps (same context window)
  paused  stopped: no VRAM/RAM; a request arriving while paused wakes it

Budgets come from sizing.py (shared with run.sh). The engine runs detached (bin/qwenflash-backend), so restarting
the manager adopts the running model instead of restarting it. Crashes are recovered automatically; a crash from
running out of VRAM or RAM lowers that budget for next time ("learned limits").
"""
import asyncio
import collections
import json
import math
import re
import secrets
import statistics
import sys
import threading
import time
import urllib.request

from . import config
from .config import GB, GIB, LOG_DIR, REPO, SERVED_NAME, SERVICE_KEYS, log
from .host import make_host
from .httpio import (StreamStats, backend_request, close, http_ok, oai_error, pipe, read_request, respond)
from .setup import Setup

sys.path.insert(0, str(REPO))
import sizing  # noqa: E402

RAM_OVERHEAD = 6.5 * GB          # measured: server process + torch + page cache on top of the expert RAM tier
UI_FILE = REPO / "qwenflash" / "ui" / "index.html"
OOM_VRAM = re.compile(r"out of memory|OutOfMemoryError|VRAM LIMIT EXCEEDED|Insufficient VRAM|cudaErrorMemoryAllocation",
                      re.I)
OOM_RAM = re.compile(r"RAM LIMIT EXCEEDED|MemoryError|Killed\b|oom-kill", re.I)


class Backend:
    def __init__(self):
        self.state = "stopped"      # stopped | starting | running | stopping | error
        self.tier = self.quant = self.context = self.pid = None
        self.since = time.time()
        self.ready_at = self.startup_s = self.error = None
        self.own_vram = self.ram_gib = self.budget = None
        self.plan = None
        self.adopted = False

    def set(self, state):
        self.state = state
        self.since = time.time()


class Monitor(threading.Thread):
    def __init__(self, svc):
        super().__init__(daemon=True)
        self.svc = svc
        self.lock = threading.Lock()
        self.latest = None

    def run(self):
        while True:
            try:
                s = self.svc.host.sample(own_busy=self.svc.in_flight > 0)
                with self.lock:
                    self.latest = s
            except Exception as e:  # noqa: BLE001 - keep monitoring through transient counter errors
                log.warning("monitor sample failed: %s", e)
            time.sleep(self.svc.cfg["poll_s"])

    def get(self):
        with self.lock:
            return self.latest


class Service:
    def __init__(self, cfg, observe=False):
        self.cfg = cfg
        self.observe = observe
        self.host = make_host(cfg)
        self.mon = Monitor(self)
        self.be = Backend()
        self.setup = Setup(self)
        self.tlock = None
        self.task = None
        self.in_flight = self.held = self.served = self.failed = 0
        self.last_request = self.last_done = None
        self.override_until = 0.0
        self.paused_reason = None
        self.ext_at_start = self.perproc_at_start = None
        self.pressure = {"level": "none", "reasons": []}
        self.sev_since = self.mod_since = self.clear_since = None
        self.last_transition = 0.0
        self.last_stop_t = time.time()
        self.events = collections.deque(maxlen=300)
        self.hist = collections.deque(maxlen=400)
        self.metrics = collections.deque(maxlen=200)
        self._last_hist_t = 0
        self._last_alive_check = 0.0
        self.recover = {"attempts": 0, "window_start": 0.0, "next_at": None, "tier": None}
        self.learned = config.load_json("learned.json", {})
        self.quit = None
        self.ollama_list, self.ollama_models, self.ollama_up, self.ollama_requests = [], set(), False, 0
        self.front_server, self._front_warned = None, False

    # ================================================================ bookkeeping
    def event(self, kind, msg):
        self.events.append([time.time(), kind, msg])
        log.info("[%s] %s", kind, msg)

    @property
    def gpu_name(self):
        return self.host.gpu.name if self.host.gpu else "no GPU"

    def learned_key(self, quant=None, context=None):
        return f"{self.gpu_name}|{quant or self.cfg['quant']}|{context or self.cfg['context']}"

    def learn(self, **kw):
        entry = self.learned.setdefault(self.learned_key(), {})
        entry.update(kw, t=time.time())
        config.save_json("learned.json", self.learned)

    def ext_vram(self, m):
        if m.get("ext_vram_abs") is not None:
            return m["ext_vram_abs"]
        if self.perproc_at_start is None or self.be.state in ("stopped", "error"):
            return m["vram_used"]
        # Windows: per-process counters don't see WSL; other apps' change since our start is exact
        return max(0.0, min(m["vram_total"], self.ext_at_start + m["perproc_ded"] - self.perproc_at_start))

    # ================================================================ budgets (sizing.py)
    def lean_budget(self, m):
        if self.cfg["lean_vram_gb"] > 0:
            return self.cfg["lean_vram_gb"] * GB
        floor = sizing.min_budget(self.cfg["quant"], self.cfg["context"]) + 1 * GB
        return max(floor, min(16 * GB, 0.55 * m["vram_total"]))

    def headroom(self, m):
        """VRAM left free in full mode: the setting, or ~10% of the card (1.2-3.2 GB) when 0 (automatic)."""
        if self.cfg["full_vram_headroom_gb"] > 0:
            return self.cfg["full_vram_headroom_gb"] * GB
        return min(3.2 * GB, max(1.2 * GB, 0.1 * m["vram_total"]))

    def full_budget(self, m):
        b = m["vram_total"] - self.ext_vram(m) - self.headroom(m)
        cap = self.learned.get(self.learned_key(), {}).get("vram_cap")
        return min(b, cap) if cap else b

    def ext_lean_threshold(self, m):
        return self.cfg["ext_vram_lean_gb"] * GB if self.cfg["ext_vram_lean_gb"] > 0 else 0.25 * m["vram_total"]

    def plan_for(self, tier, m):
        cfg = self.cfg
        budget = self.full_budget(m) if tier == "full" else min(self.lean_budget(m), self.full_budget(m))
        return sizing.plan(cfg["quant"], budget, cfg["context"])

    def own_ram_estimate(self):
        if self.be.state != "running" or not self.be.tier:
            return 0.0
        return (self.be.ram_gib or self.cfg["lean_ram_gb"]) * GIB + RAM_OVERHEAD

    def ram_spare(self, m, reserve_gb=None):
        reserve = (reserve_gb if reserve_gb is not None else self.cfg["ram_reserve_gb"]) * GB
        return m["ram_avail"] + self.own_ram_estimate() - reserve

    def full_ram_gib(self, m, plan):
        """RAM tier for a full start: enough for every expert outside VRAM, shrunk to what the host can spare
        right now (never below lean's). VRAM is what makes full fast; a smaller RAM tier only sends rarer experts
        to NVMe."""
        cfg = self.cfg
        need = math.ceil(plan["ram_for_all_experts"] / GIB)
        if cfg["max_ram_cache_gib"] > 0:
            need = min(need, cfg["max_ram_cache_gib"])
        cap = self.learned.get(self.learned_key(), {}).get("ram_cap")
        if cap:
            need = min(need, cap)
        # keep free RAM after the start clearly above the scale-back threshold, or full -> lean -> full would loop
        spare = self.ram_spare(m, max(cfg["ram_reserve_gb"], cfg["ram_lean_gb"] + 3)) - RAM_OVERHEAD
        return int(max(cfg["lean_ram_gb"], min(need, spare // GIB)))

    def ram_need_lean(self):
        return self.cfg["lean_ram_gb"] * GIB + RAM_OVERHEAD

    def best_tier(self, m, level=None):
        """Biggest profile that fits right now (respecting current pressure)."""
        level = level or self.pressure["level"]
        full_p, lean_p = self.plan_for("full", m), self.plan_for("lean", m)
        ram_ok = self.ram_spare(m) >= self.ram_need_lean()
        vram_for_us = m["vram_total"] - self.ext_vram(m) - 0.8 * GB
        full_ok = full_p["ok"] and ram_ok
        lean_ok = lean_p["ok"] and ram_ok and vram_for_us >= lean_p["budget"] + 0.5 * GB
        distinct = lean_p["budget"] < full_p["budget"] - 1.5 * GB
        if full_ok and level == "none":
            return "full"
        if lean_ok and distinct:
            return "lean"
        return "full" if full_ok else ("lean" if lean_ok else None)

    def evaluate(self, m):
        cfg = self.cfg
        sev, mod = [], []
        ext = self.ext_vram(m)
        free = m["vram_total"] - m["vram_used"]
        hits = sorted({w.lower() for w in cfg["watchlist"]} & m["running"])
        if hits:
            sev.append("watched app running: " + ", ".join(hits))
        if m["ram_avail"] < cfg["ram_pause_gb"] * GB:
            sev.append(f"RAM very low ({m['ram_avail'] / GB:.1f} GB free)")
        elif m["ram_avail"] < cfg["ram_lean_gb"] * GB:
            mod.append(f"RAM low ({m['ram_avail'] / GB:.1f} GB free)")
        if self.be.state in ("running", "starting") and self.ext_at_start is not None:
            grew = ext - self.ext_at_start
            if free < cfg["vram_min_free_gb"] * GB and grew > 0.75 * GB:
                (sev if self.be.tier == "lean" else mod).append(
                    f"other apps need VRAM (+{grew / GB:.1f} GB since start, {free / GB:.1f} GB free)")
        lean_need = min(self.lean_budget(m), self.full_budget(m) + ext)
        if ext > m["vram_total"] - (lean_need + 1.0 * GB):
            sev.append(f"other apps using {ext / GB:.1f} GB VRAM")
        elif ext > self.ext_lean_threshold(m):
            mod.append(f"other apps using {ext / GB:.1f} GB VRAM")
        if m["ext_util"] >= cfg["ext_util_pct"]:
            who = m["ext_util_top"][0][0] if m["ext_util_top"] else "another app"
            (sev if cfg["gpu_busy_action"] == "pause" else mod).append(f"GPU busy: {who} at {m['ext_util']:.0f}%")
        level = "severe" if sev else "moderate" if mod else "none"
        return level, sev + mod, ext

    # ================================================================ lifecycle
    async def fresh_sample(self, after):
        for _ in range(30):
            m = self.mon.get()
            if m and m["t"] > after:
                return m
            await asyncio.sleep(0.25)
        return self.mon.get()

    def profile_env(self, tier, m, plan):
        cfg = self.cfg
        env = {"QUANT": cfg["quant"], "CONTEXT": str(cfg["context"]), "PORT": str(cfg["backend_port"]),
               # WSL forwards Windows' localhost to sockets bound on all interfaces inside the VM
               "HOST": "0.0.0.0" if self.host.kind == "windows-wsl" else "127.0.0.1",
               # the manager handles contention; never let the in-process guard hard-kill a request
               "QWEN_VRAM_GUARD_BYTES": str(int(m["vram_total"] + 2 * GB))}
        capped = tier == "lean" or bool(self.learned.get(self.learned_key(), {}).get("vram_cap"))
        if tier == "full" and not capped:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_total"] - self.headroom(m)))
            env["RAM_GB"] = str(self.full_ram_gib(m, plan))
        else:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_used"] + plan["budget"]))
            env["RAM_GB"] = str(self.full_ram_gib(m, plan) if tier == "full" else cfg["lean_ram_gb"])
        env.update(sizing.engine_env(plan, cap_slots=capped))
        return env

    async def start(self, tier, reason):
        async with self.tlock:
            return await self._start_locked(tier, reason)

    async def _start_locked(self, tier, reason):
        cfg, host = self.cfg, self.host
        if self.observe:
            return False
        if (self.be.state in ("running", "starting") and self.be.tier == tier and self.be.quant == cfg["quant"]
                and self.be.context == cfg["context"]):
            return True
        if self.be.state in ("running", "starting", "error"):
            await self._stop_locked(f"switching to {tier}")
        m = await self.fresh_sample(time.time())
        plan = self.plan_for(tier, m)
        if tier == "full" and host.ram_settle_s:
            # just after a stop, WSL is still handing RAM back (~15-30 s): wait for it before sizing the RAM tier
            need = math.ceil(plan["ram_for_all_experts"] / GIB)
            while self.full_ram_gib(m, plan) < need and time.time() - self.last_stop_t < host.ram_settle_s:
                await asyncio.sleep(2)
                m = await self.fresh_sample(time.time())
        if not plan["ok"]:
            self.be.error = f"cannot start {tier}: {plan.get('reason')}"
            self.be.set("error")
            self.event("error", self.be.error)
            return False
        self.ext_at_start, self.perproc_at_start = self.ext_vram(m), m["perproc_ded"]
        if m.get("ext_vram_abs") is None:
            self.ext_at_start = m["vram_used"]
        env = self.profile_env(tier, m, plan)
        self.be.tier, self.be.quant, self.be.context, self.be.error = tier, cfg["quant"], cfg["context"], None
        self.be.ram_gib, self.be.budget, self.be.plan, self.be.adopted = int(env["RAM_GB"]), plan["budget"], plan, False
        self.be.set("starting")
        self.paused_reason = None
        mode = "" if plan["staging"] else ", staging-free (experimental)"
        self.event("start", f"starting {tier} ({cfg['quant']} bpw, {cfg['context'] // 1024}K ctx, "
                            f"{plan['budget'] / GB:.1f} GB VRAM budget, {self.be.ram_gib} GiB RAM cache, "
                            f"chunk {plan['chunk']}{mode}): {reason}")
        url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
        st, _ = await asyncio.to_thread(host.backend_status)
        if st in ("running", "stray"):
            await asyncio.to_thread(host.backend_stop)
            self.event("cleanup", "stopped a model server that was not started by this service")
        for _ in range(30):
            if not await asyncio.to_thread(http_ok, url, 1.0):
                break
            await asyncio.sleep(1)
        if host.follower:
            host.follower.clear()
        t0 = time.time()
        ok, pid = await asyncio.to_thread(host.backend_start, env, ["server"])
        if not ok:
            return await self._start_failed(tier, f"launcher refused: {pid}")
        self.be.pid = pid
        self.save_backend_meta(env)
        alive_t = 0.0
        while time.time() - t0 < cfg["start_timeout_s"]:
            if await asyncio.to_thread(http_ok, url):
                self.be.startup_s = time.time() - t0
                self.be.ready_at = time.time()
                self.be.set("running")
                m2 = await self.fresh_sample(time.time())
                self.be.own_vram = max(0.0, m2["vram_used"] - (self.ext_at_start or 0.0))
                self.last_transition = time.time()
                self.recover["attempts"] = 0 if time.time() - self.recover["window_start"] > 600 else \
                    self.recover["attempts"]
                self.save_backend_meta(env)
                self.event("ready", f"{tier} ready in {self.be.startup_s:.0f}s, using "
                                    f"{self.be.own_vram / GB:.1f} GB VRAM")
                await asyncio.to_thread(host.after_ready)
                return True
            if time.time() - alive_t > 5:
                alive_t = time.time()
                st, cur = await asyncio.to_thread(host.backend_status)
                if st != "running":
                    break
            await asyncio.sleep(1)
        timeout = time.time() - t0 >= cfg["start_timeout_s"]
        return await self._start_failed(tier, "timeout" if timeout else "the server exited")

    async def _start_failed(self, tier, why):
        tail = self.host.follower.tail(12) if self.host.follower else ""
        kind = self.classify(tail)
        self.be.error = f"start failed ({why}{', ' + kind if kind != 'crash' else ''}): {tail[-500:]}"
        self.event("error", self.be.error.replace("\n", " | ")[:600])
        budget = self.be.budget
        await self._stop_locked("start failed", final_state="error")
        if kind != "crash":
            self.learn_from(kind, budget)
        return False

    def classify(self, tail):
        if OOM_VRAM.search(tail or ""):
            return "out of VRAM"
        if OOM_RAM.search(tail or ""):
            return "out of RAM"
        return "crash"

    def learn_from(self, kind, budget=None, ram_gib=None):
        if kind == "out of VRAM" and budget:
            cap = budget - 2 * GB
            self.learn(vram_cap=cap)
            self.event("learned", f"VRAM budget for {self.cfg['quant']} bpw at {self.cfg['context'] // 1024}K on "
                                  f"this GPU capped at {cap / GB:.1f} GB after running out of VRAM")
        elif kind == "out of RAM":
            cap = max(self.cfg["lean_ram_gb"], (ram_gib or self.be.ram_gib or 16) - 4)
            self.learn(ram_cap=cap)
            self.event("learned", f"RAM cache capped at {cap} GiB after running out of RAM")

    def save_backend_meta(self, env):
        config.save_json("backend.json", {
            "pid": self.be.pid, "tier": self.be.tier, "quant": self.be.quant, "context": self.be.context,
            "ram_gib": self.be.ram_gib, "budget": self.be.budget, "own_vram": self.be.own_vram,
            "started": self.be.since, "ready_at": self.be.ready_at, "startup_s": self.be.startup_s,
            "engine_dir": self.host.engine_dir, "port": self.cfg["backend_port"], "env": env,
            "plan": {k: v for k, v in (self.be.plan or {}).items() if isinstance(v, (int, float, str, bool))}})

    async def stop(self, reason, paused_reason="manual"):
        async with self.tlock:
            await self._stop_locked(reason)
            self.paused_reason = paused_reason

    async def _stop_locked(self, reason, final_state="stopped"):
        if self.observe:
            return
        if self.be.state == "stopped" and not self.be.pid:
            return
        was = self.be.tier
        self.be.set("stopping")
        await asyncio.to_thread(self.host.backend_stop)
        await asyncio.to_thread(self.host.after_stop)
        try:
            (config.STATE / "backend.json").unlink()
        except OSError:
            pass
        self.be.pid = self.be.tier = self.be.ready_at = self.be.own_vram = self.be.ram_gib = None
        self.be.budget = self.be.plan = None
        self.ext_at_start = self.perproc_at_start = None
        self.be.set(final_state)
        self.last_transition = self.last_stop_t = time.time()
        if final_state == "stopped":
            self.event("stop", f"stopped{' ' + was if was else ''}: {reason}")

    async def restart(self, tier, reason):
        async with self.tlock:
            await self._stop_locked(reason)
            return await self._start_locked(tier, reason)

    def spawn(self, coro):
        if self.task and not self.task.done():
            coro.close()
            return False
        self.task = asyncio.create_task(coro)
        return True

    async def adopt_or_clean(self):
        """At service start: keep a model server we started earlier (manager restart/upgrade), stop any other."""
        host, cfg = self.host, self.cfg
        try:
            st, pid = await asyncio.to_thread(host.backend_status)
        except Exception as e:  # noqa: BLE001 - WSL may not be installed yet (setup handles that)
            log.warning("backend status failed: %s", e)
            return
        meta = config.load_json("backend.json", None)
        url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
        if self.observe:
            if await asyncio.to_thread(http_ok, url):
                self.be.set("running")
                self.be.tier = (meta or {}).get("tier") or "unknown"
                m = await self.fresh_sample(0)
                if m and m.get("ext_vram_abs") is None:
                    # not our start: take other apps' current per-process usage as the baseline
                    self.ext_at_start, self.perproc_at_start = min(m["vram_used"], m["perproc_ded"]), m["perproc_ded"]
                self.event("observe", "observe mode: a model server is answering on the backend port")
            return
        if st == "running" and meta and meta.get("pid") == pid and await asyncio.to_thread(http_ok, url, 3.0):
            m = await self.fresh_sample(0)
            b = self.be
            b.pid, b.tier, b.quant, b.context = pid, meta["tier"], meta["quant"], meta["context"]
            b.ram_gib, b.budget, b.own_vram = meta.get("ram_gib"), meta.get("budget"), meta.get("own_vram")
            b.plan, b.startup_s, b.adopted = meta.get("plan"), meta.get("startup_s"), True
            b.set("running")
            b.ready_at = meta.get("ready_at") or time.time()
            self.ext_at_start = m["vram_used"] - (b.own_vram or 0.0) if m else None
            self.perproc_at_start = m["perproc_ded"] if m else None
            self.last_transition = time.time()
            self.event("adopt", f"kept the running model ({b.tier}, pid {pid}) across the manager restart")
        elif st in ("running", "stray"):
            await asyncio.to_thread(host.backend_stop)
            self.event("cleanup", "stopped a model server left over from a previous session")
            self.last_stop_t = time.time()

    async def check_alive(self):
        st, pid = await asyncio.to_thread(self.host.backend_status)
        if self.be.state != "running" or (self.task and not self.task.done()):
            return
        if st == "running" and pid == self.be.pid:
            return
        if st == "unknown":
            return
        tail = self.host.follower.tail(30) if self.host.follower else ""
        kind = self.classify(tail)
        tier, budget, ram = self.be.tier, self.be.budget, self.be.ram_gib
        self.be.error = f"model server exited unexpectedly ({kind}): {tail[-400:]}"
        self.event("error", self.be.error.replace("\n", " | ")[:600])
        self.spawn(self._recover(tier, kind, budget, ram))

    async def _recover(self, tier, kind, budget, ram):
        async with self.tlock:
            await self._stop_locked("model server exited", final_state="error")
        if kind != "crash":
            self.learn_from(kind, budget, ram)
        r = self.recover
        if time.time() - r["window_start"] > 600:
            r["window_start"], r["attempts"] = time.time(), 0
        if not self.cfg["auto_recover"] or r["attempts"] >= 3:
            self.event("error", "not restarting automatically (3 failures in 10 minutes); use Start in the dashboard")
            return
        delay = (10, 30, 90)[r["attempts"]]
        r["attempts"] += 1
        self.event("recover", f"restarting in {delay}s (attempt {r['attempts']} of 3)")
        await asyncio.sleep(delay)
        if self.be.state == "error":
            m = self.mon.get()
            await self.start(self.best_tier(m) or tier, "automatic recovery")

    # ================================================================ request path
    async def ensure_ready(self):
        cfg = self.cfg
        busy_deadline = None
        while True:
            st = self.be.state
            if st == "running":
                return True, ""
            if self.observe:
                return False, "this manager is in observe mode and does not start the model"
            if self.setup.blocking:
                return False, f"setup step '{self.setup.running}' is running; try again when it finishes"
            if st in ("starting", "stopping"):
                await asyncio.sleep(0.5)
                continue
            if not cfg["wake_on_request"]:
                return False, "model is paused and wake-on-request is off (QwenFlash dashboard)"
            m = self.mon.get()
            tier = self.best_tier(m) if m else None
            # Right after a stop, WSL is still handing RAM back to Windows, so free RAM reads low and full would be
            # rejected for lean only to be re-saturated a minute later. Let it settle first.
            settling = time.time() - self.last_stop_t < self.host.ram_settle_s
            if tier == "lean" and settling and self.pressure["level"] == "none" and \
                    self.plan_for("full", m)["ok"]:
                await asyncio.sleep(2)
                continue
            if tier:
                if self.pressure["level"] != "none":
                    self.override_until = time.time() + cfg["wake_override_min"] * 60
                if not self.spawn(self.start(tier, "request arrived")):
                    await asyncio.sleep(0.5)
                    continue
                await asyncio.sleep(0.5)
                if self.be.state == "error":
                    return False, self.be.error or "model failed to start"
                continue
            busy_deadline = busy_deadline or time.time() + cfg["busy_wait_s"]
            if time.time() > busy_deadline:
                ext = self.ext_vram(m) / GB if m else 0
                return False, (f"GPU/RAM busy (other apps using {ext:.1f} GB VRAM, "
                               f"{(m['ram_avail'] / GB) if m else 0:.1f} GB RAM free); model not started")
            await asyncio.sleep(2)

    async def forward(self, method, target, headers, body, writer):
        cfg = self.cfg
        self.in_flight += 1
        self.last_request = t0 = time.time()
        if self.pressure["level"] != "none" and self.be.state == "running":
            self.override_until = time.time() + cfg["wake_override_min"] * 60
        held = was_held = self.be.state != "running"
        if held:
            self.held += 1
        bw = None
        stats = StreamStats()
        try:
            ok, msg = await self.ensure_ready()
            if held:
                self.held -= 1
                held = False
            if not ok:
                self.failed += 1
                await respond(writer, 503, oai_error(msg))
                return
            try:
                br, bw = await asyncio.open_connection("127.0.0.1", cfg["backend_port"])
            except OSError as e:
                self.failed += 1
                await respond(writer, 502, oai_error(f"model server unreachable: {e}", "bad_gateway", 502))
                return
            t_send = time.time()
            bw.write(backend_request(method, target, headers, body, cfg["backend_port"]))
            await bw.drain()
            await pipe(br, writer, lambda c: stats.feed(c, time.time()))
            self.served += 1
            self.record_metrics(stats, t0, t_send, time.time(), was_held)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if held:
                self.held -= 1
            self.in_flight -= 1
            self.last_done = time.time()
            close(bw, writer)

    def record_metrics(self, stats, t0, t_send, t_end, was_held):
        u = stats.usage() or {}
        if stats.status != 200 or not u:
            return
        pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        ttft = (stats.t_first - t_send) if stats.t_first else None
        decode_s = (t_end - stats.t_first) if stats.t_first else None
        self.metrics.append({
            "t": t_end, "prompt": pt, "completion": ct, "wait_s": round(t_send - t0, 2),
            "ttft_s": round(ttft, 3) if ttft is not None else None, "total_s": round(t_end - t_send, 2),
            "decode_tps": round(ct / decode_s, 1) if decode_s and ct > 8 else None,
            "prefill_tps": round(pt / ttft) if ttft and pt > 256 else None,
            "tier": self.be.tier, "budget_gb": round((self.be.budget or 0) / GB, 1), "cold": was_held})

    def metrics_summary(self):
        recent = list(self.metrics)[-30:]

        def med(key):
            vals = [r[key] for r in recent if r.get(key) is not None]
            return round(statistics.median(vals), 1) if vals else None
        return {"count": len(self.metrics), "last": recent[-1] if recent else None,
                "decode_tps": med("decode_tps"), "prefill_tps": med("prefill_tps"), "ttft_s": med("ttft_s"),
                "tokens_out": sum(r["completion"] for r in self.metrics),
                "tokens_in": sum(r["prompt"] for r in self.metrics)}

    # ================================================================ Ollama side-by-side
    def ollama_has(self, model):
        return model in self.ollama_models or (":" not in model and model + ":latest" in self.ollama_models)

    def route_model(self, body):
        """'qwen' or 'ollama' for an OpenAI chat/completions request, by its model name. Anything that isn't an
        installed Ollama model goes to QwenFlash (it serves one model and ignores the name)."""
        try:
            model = str(json.loads(body or b"{}").get("model") or "")
        except (ValueError, AttributeError):
            return "qwen"
        if model == SERVED_NAME or model in self.cfg["qwenflash_aliases"]:
            return "qwen"
        return "ollama" if model and self.ollama_up and self.ollama_has(model) else "qwen"

    def models_list(self):
        own = {"id": SERVED_NAME, "object": "model", "created": 0, "owned_by": "qwenflash",
               "max_model_len": self.cfg["context"]}
        others = [m for m in self.ollama_list if m.get("id") != SERVED_NAME] if self.ollama_up else []
        return {"object": "list", "data": [own] + others}

    async def refresh_ollama(self):
        url = f"http://127.0.0.1:{self.cfg['ollama_port']}/v1/models"

        def fetch():
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.load(r)
        try:
            data = await asyncio.to_thread(fetch)
            self.ollama_list = list(data.get("data") or [])
            self.ollama_models = {m.get("id") for m in self.ollama_list if m.get("id")}
            if not self.ollama_up:
                self.event("ollama", f"Ollama reachable on :{self.cfg['ollama_port']} "
                                     f"({len(self.ollama_models)} models)")
            self.ollama_up = True
        except Exception:  # noqa: BLE001 - Ollama not running is a normal state
            if self.ollama_up:
                self.event("ollama", f"Ollama not reachable on :{self.cfg['ollama_port']}")
            self.ollama_up = False

    async def forward_ollama(self, method, target, headers, body, writer):
        port = self.cfg["ollama_port"]
        self.ollama_requests += 1
        bw = None
        try:
            try:
                br, bw = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                await respond(writer, 502, oai_error(f"Ollama is not running on 127.0.0.1:{port}", "bad_gateway", 502))
                return
            bw.write(backend_request(method, target, headers, body, port))
            await bw.drain()
            await pipe(br, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            close(bw, writer)

    async def bind_front(self):
        """Listen on the Ollama-compatible front door (default 127.0.0.1:11434) once Ollama has moved off it."""
        if self.front_server or not self.cfg["ollama_alias_port"] or self.observe:
            return
        try:
            self.front_server = await asyncio.start_server(self.handle, ["127.0.0.1", "::1"],
                                                           self.cfg["ollama_alias_port"], limit=8 * 1024 * 1024)
            self.event("service", f"front door listening on 127.0.0.1:{self.cfg['ollama_alias_port']} "
                                  f"(QwenFlash chat + Ollama pass-through)")
        except OSError as e:
            if not self._front_warned:
                self.event("warning", f"port {self.cfg['ollama_alias_port']} is busy ({e.strerror}); is Ollama still "
                                      f"listening there? Retrying every 30 s.")
                self._front_warned = True

    # ================================================================ HTTP entry
    def authorized(self, headers, target):
        cfg = self.cfg
        if cfg["listen_host"] in ("127.0.0.1", "localhost", "::1") or not cfg["dashboard_token"]:
            return True
        h = {k.lower(): v for k, v in headers}
        tok = cfg["dashboard_token"]
        if h.get("authorization", "") == f"Bearer {tok}":
            return True
        if f"qf_token={tok}" in h.get("cookie", "").replace(" ", ""):
            return True
        return f"token={tok}" in target

    async def handle(self, reader, writer):
        try:
            method, target, headers, body = await read_request(reader, writer)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
            close(writer)
            return
        path = target.split("?", 1)[0]
        p = path.rstrip("/") or "/"
        # The Ollama-compatible front door (also what a tailnet reaches via `tailscale serve`): chat is answered
        # here, everything else passes through to Ollama, and the dashboard/control API is never served.
        front = writer.get_extra_info("sockname")[1] == self.cfg["ollama_alias_port"]
        try:
            if path.startswith("/_switcher") or (path == "/" and not front):
                if front:
                    await respond(writer, 404, {"error": "not found"})
                elif not self.authorized(headers, target):
                    await respond(writer, 401, b"QwenFlash dashboard: open it with ?token=<dashboard_token>",
                                  "text/plain")
                else:
                    await self.api(method, path, target, body, writer)
                close(writer)
            elif method == "GET" and p == "/v1/models":
                await respond(writer, 200, self.models_list())
                close(writer)
            elif p in ("/v1/chat/completions", "/v1/completions") and self.route_model(body) == "qwen":
                await self.forward(method, target, headers, body, writer)
            elif (front or path.startswith("/v1/")) and self.ollama_up:
                await self.forward_ollama(method, target, headers, body, writer)
            elif front and method in ("GET", "HEAD") and path == "/":
                await respond(writer, 200, b"QwenFlash is running", "text/plain")
                close(writer)
            elif front or path.startswith("/v1/"):
                await respond(writer, 404, {"error": {
                    "message": f"{path} is not available: this server is QwenFlash (OpenAI-compatible: "
                               f"/v1/chat/completions, /v1/completions, /v1/models). Embeddings, images, the Anthropic "
                               f"/v1/messages API and Ollama's /api/* are not supported.",
                    "type": "not_found", "code": 404}})
                close(writer)
            else:
                # favicon.ico, scanners...: never wake the model for these
                await respond(writer, 404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})
                close(writer)
        except Exception as e:  # noqa: BLE001
            log.exception("request failed: %s", e)
            close(writer)

    # ================================================================ control API
    def engine_sync(self):
        hw = self.setup.hw or {}
        e, repo = hw.get("engine") or {}, hw.get("repo") or {}
        if not e.get("engine_head") or not repo.get("head"):
            return None
        same = e["engine_head"] == repo["head"]
        return {"repo": repo["head"][:7], "engine": e["engine_head"][:7], "in_sync": same,
                "repo_dirty": repo.get("dirty"), "separate": self.host.kind == "windows-wsl"}

    def status(self):
        m = self.mon.get() or {}
        now = time.time()
        cfg = self.cfg
        ext = self.ext_vram(m) if m else 0.0
        host = "127.0.0.1" if cfg["listen_host"] in ("127.0.0.1", "0.0.0.0") else cfg["listen_host"]
        plans = {}
        if m:
            for t in ("full", "lean"):
                p = self.plan_for(t, m)
                plans[t] = {"ok": p["ok"], "budget_gb": round(p["budget"] / GB, 1), "chunk": p["chunk"],
                            "slots": p["slots"], "staging": p["staging"], "experimental": p["experimental"],
                            "reason": p.get("reason")}
            plans["best"] = self.best_tier(m)
        hw = self.setup.hw or {}
        return {
            "now": now, "state": self.be.state, "tier": self.be.tier, "quant": self.be.quant or cfg["quant"],
            "context": cfg["context"], "endpoint": f"http://{host}:{cfg['listen_port']}/v1",
            "served_name": SERVED_NAME, "since": self.be.since, "startup_s": self.be.startup_s,
            "error": self.be.error, "paused_reason": self.paused_reason, "ram_cache_gib": self.be.ram_gib,
            "budget_gb": round(self.be.budget / GB, 1) if self.be.budget else None, "adopted": self.be.adopted,
            "in_flight": self.in_flight, "held": self.held, "served": self.served, "failed": self.failed,
            "last_request": self.last_request,
            "idle_s": (now - self.last_done) if self.last_done and not self.in_flight else None,
            "override_until": self.override_until if self.override_until > now else None,
            "transition": bool(self.task and not self.task.done()),
            "pressure": self.pressure, "plans": plans,
            "gpu": {"name": self.gpu_name, "total": m.get("vram_total"), "used": m.get("vram_used"),
                    "ext": ext, "own": max(0.0, m.get("vram_used", 0) - ext) if m else 0.0,
                    "util": m.get("gpu_util"), "ext_util": m.get("ext_util"), "ext_util_top": m.get("ext_util_top")},
            "ram": {"total": m.get("ram_total"), "avail": m.get("ram_avail"), "model_est": self.own_ram_estimate()},
            "history": list(self.hist)[-240:], "events": list(self.events)[-80:],
            "metrics": self.metrics_summary(), "learned": self.learned.get(self.learned_key()),
            "recover": {"attempts": self.recover["attempts"]},
            "autostart": self.host.autostart_enabled(), "platform": self.host.kind, "platform_label": self.host.label,
            "observe": self.observe, "engine_sync": self.engine_sync(),
            "recommendations": hw.get("recommendations"),
            "quants": {q: {"label": v["label"]} for q, v in sizing.QUANTS.items()},
            "ollama": {"up": self.ollama_up, "port": cfg["ollama_port"], "models": len(self.ollama_models),
                       "requests": self.ollama_requests, "front_port": cfg["ollama_alias_port"],
                       "front_bound": bool(self.front_server)},
            "setup_running": self.setup.running,
        }

    async def api(self, method, path, target, body, writer):
        if path == "/":
            await respond(writer, 302, b"", "text/plain", ("Location: /_switcher/",))
            return
        if path in ("/_switcher", "/_switcher/", "/_switcher/index.html"):
            extra = ()
            m = re.search(r"token=([\w-]+)", target)
            if m:
                extra = (f"Set-Cookie: qf_token={m.group(1)}; Path=/; HttpOnly; SameSite=Strict",)
            await respond(writer, 200, UI_FILE.read_bytes(), "text/html; charset=utf-8", extra)
            return
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            await respond(writer, 400, {"error": "bad json"})
            return
        route = path[len("/_switcher/api/"):] if path.startswith("/_switcher/api/") else None
        if route == "status":
            await respond(writer, 200, self.status())
        elif route == "log":
            await respond(writer, 200, {"backend": self.host.follower.tail(300) if self.host.follower else "",
                                        "switcher": tail_file(LOG_DIR / "manager.log", 150)})
        elif route == "config" and method == "GET":
            await respond(writer, 200, {"config": self.cfg, "defaults": config.DEFAULTS,
                                        "service_keys": sorted(SERVICE_KEYS)})
        elif route == "config" and method == "POST":
            await respond(writer, *self.update_config(data.get("config", {})))
        elif route == "action" and method == "POST":
            await respond(writer, *self.action(data.get("action", "")))
        elif route == "setup" and method == "GET":
            since = int((re.search(r"since=(\d+)", target) or [0, 0])[1])
            await respond(writer, 200, self.setup.report(since))
        elif route == "hardware" and method == "GET":
            if not self.setup.hw or "refresh=1" in target:
                await self.setup.check()
            await respond(writer, 200, self.setup.hw or {})
        elif route == "metrics" and method == "GET":
            await respond(writer, 200, {"recent": list(self.metrics)[-100:], "summary": self.metrics_summary()})
        elif route == "setup/check" and method == "POST":
            asyncio.create_task(self.setup.check())
            await respond(writer, 200, {"ok": True})
        elif route == "setup/run" and method == "POST":
            if self.observe:
                await respond(writer, 409, {"ok": False, "message": "observe mode"})
                return
            ok, msg = await self.setup.run(data.get("step", ""))
            await respond(writer, 200 if ok else 409, {"ok": ok, "message": msg})
        else:
            await respond(writer, 404, {"error": "not found"})

    def update_config(self, changes):
        restart_keys = {"quant", "context", "lean_vram_gb", "lean_ram_gb", "full_vram_headroom_gb", "max_ram_cache_gib"}
        changed, service = [], []
        for k, v in changes.items():
            if k not in config.DEFAULTS:
                continue
            try:
                v = config.coerce(k, v)
            except (TypeError, ValueError):
                return 400, {"ok": False, "message": f"bad value for {k}"}
            if k == "quant" and v not in sizing.QUANTS:
                return 400, {"ok": False, "message": "unknown quant"}
            if k == "context" and not 4096 <= v <= 262144:
                return 400, {"ok": False, "message": "context must be between 4096 and 262144"}
            if self.cfg[k] != v:
                self.cfg[k] = v
                (service if k in SERVICE_KEYS else changed).append(k)
        config.save_config(self.cfg)
        msg = "saved"
        if changed or service:
            self.event("config", "changed: " + ", ".join(changed + service))
        if restart_keys & set(changed) and self.be.state == "running" and not self.observe:
            self.spawn(self.restart(self.be.tier, "settings changed"))
            msg = "saved; restarting the model to apply"
        if service:
            msg += "; " + ", ".join(service) + " apply when the service restarts"
        return 200, {"ok": True, "message": msg, "changed": changed + service}

    def action(self, a):
        cfg = self.cfg
        lifecycle = {"start_full", "start_lean", "pause", "restart"}
        if self.observe and a in lifecycle:
            return 409, {"ok": False, "message": "observe mode: this manager does not control the model"}
        msg = None
        if a in ("start_full", "start_lean"):
            ok = self.spawn(self.start(a.split("_")[1], "started from dashboard"))
        elif a == "pause":
            ok = self.spawn(self.stop("paused from dashboard", "manual"))
        elif a == "restart":
            ok = self.spawn(self.restart(self.be.tier or "full", "restarted from dashboard"))
        elif a in ("auto_on", "auto_off"):
            cfg["auto_switch"] = a == "auto_on"
            config.save_config(cfg)
            self.event("config", f"auto-switch {'on' if cfg['auto_switch'] else 'off'}")
            ok = True
        elif a in ("wake_on", "wake_off"):
            cfg["wake_on_request"] = a == "wake_on"
            config.save_config(cfg)
            ok = True
        elif a in ("autostart_on", "autostart_off"):
            ok, msg = self.host.set_autostart(a == "autostart_on")
        elif a == "forget_learned":
            self.learned.pop(self.learned_key(), None)
            config.save_json("learned.json", self.learned)
            self.event("config", "forgot the learned limits for this GPU/quant/context")
            ok = True
        elif a == "quit":
            self.spawn(self.shutdown(stop_model=True))
            ok = True
        elif a == "quit_keep_model":
            self.spawn(self.shutdown(stop_model=False))
            ok = True
        else:
            return 400, {"ok": False, "message": "unknown action"}
        if not ok:
            return 409, {"ok": False, "message": msg or "another change is in progress"}
        return 200, {"ok": True, "message": msg}

    async def shutdown(self, stop_model=True):
        self.event("quit", "service stopping" + ("" if stop_model else " (model left running for the next start)"))
        if stop_model:
            await self.stop("service quit")
        self.quit.set()

    # ================================================================ control loop
    async def control_loop(self):
        n = 0
        while True:
            await asyncio.sleep(1.0)
            n += 1
            try:
                await self.tick()
                if n % 10 == 0 and self.be.state == "running" and not self.observe:
                    await self.check_alive()
                if n % 30 == 0:
                    await self.refresh_ollama()
                    await self.bind_front()
                if n % 600 == 0:
                    asyncio.create_task(self.setup.check())
            except Exception as e:  # noqa: BLE001
                log.exception("tick failed: %s", e)

    async def tick(self):
        cfg = self.cfg
        m = self.mon.get()
        if not m:
            return
        now = time.time()
        level, reasons, ext = self.evaluate(m)
        self.pressure = {"level": level, "reasons": reasons}
        self.sev_since = (self.sev_since or now) if level == "severe" else None
        self.mod_since = (self.mod_since or now) if level != "none" else None
        self.clear_since = (self.clear_since or now) if level == "none" else None
        if m["t"] != self._last_hist_t:
            self._last_hist_t = m["t"]
            self.hist.append([round(m["t"], 1), round(ext / GB, 2), round(max(0.0, m["vram_used"] - ext) / GB, 2),
                              round((m["ram_total"] - m["ram_avail"]) / GB, 2), round(m["gpu_util"])])
        if self.observe or not cfg["auto_switch"] or self.setup.blocking or (self.task and not self.task.done()):
            return
        st, tier = self.be.state, self.be.tier
        idle = self.in_flight == 0 and (self.last_done is None or now - self.last_done >= 10)
        overridden = now < self.override_until
        dwell_ok = now - self.last_transition >= cfg["min_dwell_s"]
        if st == "running":
            if self.sev_since and now - self.sev_since >= cfg["severe_after_s"] and not overridden:
                waited = now - self.sev_since
                if self.in_flight == 0 or waited >= cfg["severe_max_wait_s"]:
                    self.spawn(self.stop("pressure: " + "; ".join(reasons), "pressure"))
                return
            # scaling DOWN is never delayed by the dwell time (backing off from a game should be quick);
            # the sustain window (moderate_after_s) is its hysteresis. Dwell only guards scaling back up.
            if (tier == "full" and self.mod_since and now - self.mod_since >= cfg["moderate_after_s"]
                    and idle and not overridden):
                why = "pressure: " + "; ".join(reasons)
                if cfg["pressure_response"] == "lean" and self.best_tier(m, level="moderate") == "lean":
                    self.spawn(self.restart("lean", why))
                else:
                    self.spawn(self.stop(why, "pressure"))
                return
            # Re-saturate lean -> full. A busy agent never leaves a 10 s idle gap, so don't wait for one: once lean
            # has run 2 min with no pressure, restart between requests (requests arriving meanwhile are held).
            if (tier == "lean" and cfg["upgrade_when_idle"] and self.clear_since
                    and now - self.clear_since >= cfg["recover_after_s"] and self.in_flight == 0 and dwell_ok
                    and (idle or now - (self.be.ready_at or now) >= 120)
                    and self.best_tier(m, level="none") == "full"):
                self.spawn(self.restart("full", "resources free again: re-saturating"))
                return
            if (cfg["idle_unload_min"] > 0 and self.in_flight == 0
                    and now - (self.last_done or self.be.ready_at or now) >= cfg["idle_unload_min"] * 60):
                self.spawn(self.stop(f"idle for {cfg['idle_unload_min']} min", "idle"))
        elif st == "stopped":
            if (cfg["preload_when_free"] and self.paused_reason == "pressure" and self.clear_since
                    and now - self.clear_since >= cfg["recover_after_s"] and dwell_ok):
                t = self.best_tier(m, level="none")
                if t:
                    self.spawn(self.start(t, "resources free again: preloading"))

    # ================================================================ main
    async def serve(self, open_window=False):
        self.tlock = asyncio.Lock()
        self.quit = asyncio.Event()
        cfg = self.cfg
        if cfg["listen_host"] not in ("127.0.0.1", "localhost", "::1") and not cfg["dashboard_token"]:
            cfg["dashboard_token"] = secrets.token_urlsafe(18)
            config.save_config(cfg)
            log.warning("listening on %s: generated a dashboard token (config.json dashboard_token)", cfg["listen_host"])
        self.mon.start()
        hosts = ["127.0.0.1", "::1"] if cfg["listen_host"] == "127.0.0.1" else [cfg["listen_host"]]
        try:
            server = await asyncio.start_server(self.handle, hosts, cfg["listen_port"], limit=8 * 1024 * 1024)
        except OSError as e:
            raise SystemExit(f"port {cfg['listen_port']} is in use: {e}")
        try:
            self.host.start_follower()
        except Exception as e:  # noqa: BLE001 - no WSL yet: setup will create it
            log.warning("log follower not started: %s", e)
        await self.adopt_or_clean()
        self.event("service", f"{'observing' if self.observe else 'listening'} on {cfg['listen_host']}:"
                              f"{cfg['listen_port']} ({self.host.label}; model on :{cfg['backend_port']})")
        await self.refresh_ollama()
        await self.bind_front()
        asyncio.create_task(self.control_loop())
        asyncio.create_task(self.setup.check())
        if cfg["preload_on_launch"] and self.be.state == "stopped" and not self.observe:
            await asyncio.sleep(3)
            m = self.mon.get()
            t = self.best_tier(m) if m else None
            if t:
                self.spawn(self.start(t, "preload on launch"))
        if open_window:
            self.host.open_ui(f"http://127.0.0.1:{cfg['listen_port']}/_switcher/")
        async with server:
            await self.quit.wait()
        if self.host.follower:
            self.host.follower.stop()


def tail_file(path, lines=200):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 96 * 1024))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""

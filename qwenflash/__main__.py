"""QwenFlash manager.

  python -m qwenflash [serve] [--open] [--observe] [--port N]   run the service (dashboard + OpenAI API)
  python -m qwenflash open                                     open the dashboard, starting the service if needed
  python -m qwenflash status                                   what the running service is doing
  python -m qwenflash detect                                   hardware, platform and recommendations
  python -m qwenflash setup [STEP]                             list setup steps, or run one in this terminal
  python -m qwenflash plan [--budget-gb N]                     VRAM/RAM plan for this GPU (sizing.py)
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .config import GB, IS_WINDOWS, REPO
from .httpio import http_get_json, http_ok


def service_url(cfg):
    """Base of the dashboard's control API (append api/...)."""
    return f"http://127.0.0.1:{cfg['listen_port']}/_switcher/"


def dashboard_url(cfg):
    """The dashboard, at the root of the port that serves the API on /v1."""
    return f"http://127.0.0.1:{cfg['listen_port']}/"


def cmd_serve(cfg, a):
    from .service import Service
    if a.port:
        cfg["listen_port"] = a.port
    if http_ok(service_url(cfg) + "api/status", 1.5):
        if a.open:
            from .host import make_host
            make_host(cfg).open_ui(dashboard_url(cfg))
        else:
            print(f"QwenFlash is already running on port {cfg['listen_port']}")
        return 0
    if a.observe:
        # never compete with the real service for its ports
        cfg["network_port"] = 0
    asyncio.run(Service(cfg, observe=a.observe).serve(a.open))
    return 0


def cmd_open(cfg, a):
    url = service_url(cfg)
    if not http_ok(url + "api/status", 1.5):
        exe = sys.executable
        if IS_WINDOWS:
            pyw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            exe = pyw if os.path.exists(pyw) else exe
            subprocess.Popen([exe, "-m", "qwenflash", "serve"], cwd=REPO,
                             creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            subprocess.Popen([exe, "-m", "qwenflash", "serve"], cwd=REPO, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            if http_ok(url + "api/status", 1.0):
                break
            time.sleep(0.5)
    from .host import make_host
    make_host(cfg).open_ui(dashboard_url(cfg))
    return 0


def cmd_status(cfg, a):
    try:
        s = http_get_json(service_url(cfg) + "api/status", 3)
    except Exception as e:  # noqa: BLE001
        print(f"QwenFlash is not running ({e})")
        return 1
    if a.json:
        print(json.dumps(s, indent=2))
        return 0
    g = s["gpu"]
    print(f"model    {s['state']}{' (' + s['tier'] + ')' if s['tier'] else ''}  {s['quant']} bpw  "
          f"{s['context'] // 1024}K context  endpoint {s['endpoint']}")
    if s.get("budget_gb"):
        print(f"budget   {s['budget_gb']} GB VRAM, {s['ram_cache_gib']} GiB RAM cache")
    print(f"gpu      {g['name']}: {(g['used'] or 0) / GB:.1f} of {(g['total'] or 0) / GB:.1f} GB used "
          f"(model {g['own'] / GB:.1f}, other apps {g['ext'] / GB:.1f})")
    print(f"pressure {s['pressure']['level']}  {'; '.join(s['pressure']['reasons'])}")
    mt = s.get("metrics") or {}
    if mt.get("count"):
        print(f"speed    decode {mt['decode_tps']} tok/s, prefill {mt['prefill_tps']} tok/s, "
              f"first token {mt['ttft_s']} s (median of recent requests)")
    print(f"requests served {s['served']}, failed {s['failed']}, in flight {s['in_flight']}")
    if s.get("error"):
        print(f"error    {s['error'][:300]}")
    return 0


def print_recommendations(recs):
    icon = {"ok": "ok  ", "info": "info", "warn": "WARN", "error": "FAIL"}
    for r in recs:
        print(f"  [{icon.get(r['level'], r['level'])}] {r['title']}")
        if r.get("detail"):
            print(f"         {r['detail']}")
        if r.get("settings"):
            print(f"         suggested: {r['settings']}")


def cmd_detect(cfg, a):
    from . import hardware
    from .host import make_host
    hw = hardware.detect(make_host(cfg), cfg)
    if a.json:
        print(json.dumps(hw, indent=2, default=str))
        return 0
    h = hw["host"]
    print(f"platform {h['label']}: {h.get('os')}{' (inside WSL)' if h.get('in_wsl') else ''}")
    print(f"cpu      {hw['cpu'].get('name')} ({hw['cpu'].get('cores')} cores), RAM {hw['ram']['total_gb']} GB")
    for g in hw["gpus"]:
        print(f"gpu {g['index']}    {g['name']}  {g['vram_total'] / 2**30:.0f} GB  sm_{(g['compute_capability'] or '?').replace('.', '')}"
              f"  PCIe {g.get('pcie_gen_max')}.0 x{g.get('pcie_width_max')}")
    if "wsl" in hw:
        w = hw["wsl"]
        print(f"wsl      {'installed ' + (w['version'] or '') if w['installed'] else 'not installed'}; "
              f"{w['distro']}: {'present (WSL ' + str(w['distro_wsl_version']) + ')' if w['distro_present'] else 'missing'}"
              f"; .wslconfig memory {hw['wslconfig'].get('memory_gb')} GB")
    e = hw.get("engine") or {}
    if e:
        print(f"engine   {'ready' if e.get('engine') == '1' else 'not set up'} at {cfg['repo_dir'] or 'this checkout'}; "
              f"CUDA {e.get('cuda_version')}, torch {e.get('torch')}{', built for ' + e['build_arch'] if e.get('build_arch') else ''}; "
              f"models: 3.05 {'yes' if e.get('m305') == '1' else 'no'}, 4.05 {'yes' if e.get('m405') == '1' else 'no'}")
    print("recommendations:")
    print_recommendations(hw["recommendations"])
    return 0


def cmd_setup(cfg, a):
    from .service import Service
    svc = Service(cfg)
    asyncio.run(svc.setup.check())
    if not a.step:
        for s in svc.setup.steps:
            print(f"  {s['id']:10s} {svc.setup.status.get(s['id'], '?'):9s} {s['title']}")
        print("run one with: python -m qwenflash setup STEP")
        return 0

    async def go():
        svc.tlock = asyncio.Lock()
        svc.setup.emit = lambda line: print(line, flush=True)
        ok, msg = await svc.setup.run(a.step)
        if not ok:
            print(msg)
            return 2
        while svc.setup.running:
            await asyncio.sleep(0.5)
        return 0 if svc.setup.status.get(a.step) in ("ok", "outdated") else 1
    return asyncio.run(go())


def first_run_baseline(cfg):
    """A new install starts from the preset for this machine (sizing.baseline: card + RAM + platform)."""
    import psutil
    from .hardware import machine_baseline
    from .host import make_host
    try:
        h = make_host(cfg)
        if not h.gpu:
            return None
        b = machine_baseline(h.gpu.info(), psutil.virtual_memory().total, h.kind)
    except Exception as e:  # noqa: BLE001 - no GPU access yet: keep the defaults
        config.log.info("first run: no baseline (%s)", e)
        return None
    if not b["ok"]:
        config.log.info("first run: %s", b["reason"])
        return b
    settings = dict(b["settings"])
    if h.kind == "linux":
        # an engine checkout that already holds the other model: keep it rather than ask for a second download
        sys.path.insert(0, str(REPO))
        import sizing

        def present(q):
            root = Path(cfg["repo_dir"] or h.engine_dir)
            return (root / sizing.QUANTS[q]["model_dir"]).is_dir() and (root / sizing.QUANTS[q]["store"]).is_file()
        other = "3.05" if settings["quant"] == "4.05" else "4.05"
        if not present(settings["quant"]) and present(other):
            settings.pop("quant")
            settings.pop("context")
    cfg.update(settings)
    config.log.info("first run: baseline for this machine: %s", settings)
    return b


def cmd_plan(cfg, a):
    sys.path.insert(0, str(REPO))
    import psutil
    import sizing
    from .host import make_host
    from .hardware import gpu_budget, has_display, machine_baseline
    h = make_host(cfg)
    if h.gpu and not a.budget_gb:
        info = h.gpu.info()
        ram = psutil.virtual_memory().total
        print(f"this machine: {info['name']} ({info['vram_total'] / 2**30:.0f} GB, "
              f"{'display' if has_display(info) else 'no display'}), "
              f"{ram / 2**30:.0f} GB RAM, {sizing.PLATFORMS.get(h.kind, h.kind)}")
        print("baseline:     " + sizing.describe(machine_baseline(info, ram, h.kind)).replace("\n  ", "\n              "))
        print(f"your config:  {cfg['quant']} bpw, {cfg['context'] // 1024}K context")
        print()
    budget = a.budget_gb * GB if a.budget_gb else (gpu_budget(h.gpu.info(), cfg) if h.gpu else 0)
    for q in sizing.QUANTS:
        p = sizing.plan(q, budget, a.context or cfg["context"])
        state = "ok" if p["ok"] else "does not fit"
        print(f"{q} bpw, {budget / GB:.1f} GB, {p['context'] // 1024}K: {state}; chunk {p['chunk']}, "
              f"{p['slots']} cached experts, {p['ram_for_all_experts'] / 2**30:.0f} GiB RAM holds the rest; "
              f"max context {sizing.max_context(q, budget, staging=False) // 1024}K"
              + ("" if p["staging"] else " (staging-free)"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qwenflash", description="QwenFlash manager")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the service")
    for p in (ap, s):
        p.add_argument("--open", action="store_true", help="open the dashboard")
        p.add_argument("--observe", action="store_true", help="watch only: never start, stop or resize the model")
        p.add_argument("--port", type=int, help="dashboard/API port (default from config, 30000)")
    sub.add_parser("open", help="open the dashboard (starts the service if needed)")
    st = sub.add_parser("status", help="show what the running service is doing")
    st.add_argument("--json", action="store_true")
    d = sub.add_parser("detect", help="hardware, platform and recommendations")
    d.add_argument("--json", action="store_true")
    su = sub.add_parser("setup", help="list or run setup steps")
    su.add_argument("step", nargs="?")
    pl = sub.add_parser("plan", help="VRAM/RAM plan for this GPU")
    pl.add_argument("--budget-gb", type=float)
    pl.add_argument("--context", type=int)
    a = ap.parse_args(argv)
    cfg = config.load_config()
    if not config.CONFIG_PATH.exists():
        first_run_baseline(cfg)
        config.save_config(cfg)
    handler = {None: cmd_serve, "serve": cmd_serve, "open": cmd_open, "status": cmd_status, "detect": cmd_detect,
               "setup": cmd_setup, "plan": cmd_plan}[a.cmd]
    return handler(cfg, a)


if __name__ == "__main__":
    sys.exit(main())

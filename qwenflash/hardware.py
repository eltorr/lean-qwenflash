"""Hardware and platform detection, and the recommendations the dashboard shows.

detect() is cheap enough to run on demand (a few hundred ms; WSL queries are the slow part) and never raises:
missing pieces are reported as such.
"""
import configparser
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

import psutil


from .config import GB, GIB, IS_WINDOWS, REPO

sys.path.insert(0, str(REPO))
import sizing  # noqa: E402  (repo root: shared with run.sh and the engine)

NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
WSL_ENV = dict(os.environ, WSL_UTF8="1")


def _run(args, timeout=20, env=None):
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, env=env, creationflags=NO_WINDOW,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "").replace("\x00", ""), (r.stderr or "").replace("\x00", "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def parse_kv(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


# ---------------------------------------------------------------------------------------------------------------
# Windows / WSL
# ---------------------------------------------------------------------------------------------------------------
def wsl_info(distro):
    info = {"installed": False, "version": None, "kernel": None, "default_version": None, "distros": [],
            "distro": distro, "distro_present": False, "distro_wsl_version": None, "distro_state": None,
            "distro_path": None}
    rc, out, _ = _run(["wsl.exe", "--version"], env=WSL_ENV)
    if rc == 0 and "WSL" in out:
        info["installed"] = True
        m = re.search(r"WSL[^:\n]*:\s*([\d.]+)", out)
        info["version"] = m.group(1) if m else None
        m = re.search(r"Kernel[^:\n]*:\s*([\w.+-]+)", out)
        info["kernel"] = m.group(1) if m else None
    rc, out, _ = _run(["wsl.exe", "--status"], env=WSL_ENV)
    m = re.search(r"Default Version:\s*(\d)", out)
    info["default_version"] = int(m.group(1)) if m else None
    rc, out, _ = _run(["wsl.exe", "-l", "-v"], env=WSL_ENV)
    for line in out.splitlines()[1:]:
        parts = line.replace("*", " ").split()
        if len(parts) >= 3 and parts[-1].isdigit():
            d = {"name": " ".join(parts[:-2]), "state": parts[-2], "version": int(parts[-1]),
                 "default": line.strip().startswith("*")}
            info["distros"].append(d)
            if d["name"] == distro:
                info.update(distro_present=True, distro_wsl_version=d["version"], distro_state=d["state"])
    if info["distros"]:
        info["installed"] = True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Lxss") as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                with winreg.OpenKey(k, winreg.EnumKey(k, i)) as d:
                    try:
                        if winreg.QueryValueEx(d, "DistributionName")[0] == distro:
                            info["distro_path"] = winreg.QueryValueEx(d, "BasePath")[0].replace("\\\\?\\", "")
                    except OSError:
                        continue
    except OSError:
        pass
    return info


def wslconfig():
    p = Path.home() / ".wslconfig"
    out = {"path": str(p), "exists": p.exists(), "memory_gb": None, "processors": None, "swap_gb": None}
    if not p.exists():
        return out
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read_string(p.read_text(errors="replace"))
    except configparser.Error as e:
        out["error"] = str(e)
        return out
    sec = cp["wsl2"] if cp.has_section("wsl2") else {}

    def size_gb(v):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(GB|MB|TB)?", str(v), re.I)
        if not m:
            return None
        n, unit = float(m.group(1)), (m.group(2) or "B").upper()
        return n * {"TB": 1024, "GB": 1, "MB": 1 / 1024}.get(unit, 1 / GIB)
    out["memory_gb"] = size_gb(sec.get("memory")) if sec.get("memory") else None
    out["swap_gb"] = size_gb(sec.get("swap")) if sec.get("swap") else None
    out["processors"] = int(sec["processors"]) if str(sec.get("processors", "")).isdigit() else None
    return out


def windows_cpu():
    name = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    k32 = ctypes.windll.kernel32
    return {"name": name, "avx2": bool(k32.IsProcessorFeaturePresent(40)),
            "virtualization_firmware": bool(k32.IsProcessorFeaturePresent(21))}


def fixed_drives():
    out = []
    for p in psutil.disk_partitions(all=False):
        if "fixed" in p.opts or not IS_WINDOWS:
            try:
                u = psutil.disk_usage(p.mountpoint)
                out.append({"mount": p.mountpoint, "fs": p.fstype, "free_gb": round(u.free / GB),
                            "total_gb": round(u.total / GB)})
            except OSError:
                continue
    return out


# ---------------------------------------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------------------------------------
def linux_cpu():
    name, flags = None, ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and not name:
                name = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line
    except OSError:
        pass
    return {"name": name, "avx2": " avx2" in flags}


def repo_state():
    rc, head, _ = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    rc2, dirty, _ = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    rc3, subject, _ = _run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"])
    return {"head": head.strip() if rc == 0 else None, "dirty": bool(dirty.strip()) if rc2 == 0 else None,
            "subject": subject.strip() if rc3 == 0 else None}


def engine_check(host):
    """setup/check.sh in the engine's environment (WSL or here)."""
    script = host.to_engine_path(REPO / "setup" / "check.sh")
    try:
        r = host.run("bash", script, host.engine_dir, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    vals = parse_kv(r.stdout or "")
    if "driver" not in vals:
        vals["error"] = ((r.stderr or "") + (r.stdout or "")).strip()[-300:] or f"exit {r.returncode}"
    return vals


def detect(host, cfg, with_engine=True):
    hw = {"host": host.describe(), "repo": repo_state()}
    vm = psutil.virtual_memory()
    hw["ram"] = {"total_gb": round(vm.total / GB, 1), "avail_gb": round(vm.available / GB, 1)}
    hw["cpu"] = windows_cpu() if IS_WINDOWS else linux_cpu()
    hw["cpu"].update(cores=psutil.cpu_count(logical=False), threads=psutil.cpu_count())
    gpus = []
    if host.nvml:
        for d in host.nvml.devices:
            try:
                gpus.append(d.info())
            except OSError:
                continue
        hw["driver"] = {"version": host.nvml.driver_version(), "cuda": host.nvml.cuda_driver_version()}
    else:
        hw["driver"] = {"error": host.nvml_error}
    hw["gpus"] = gpus
    hw["drives"] = fixed_drives()
    if IS_WINDOWS:
        hw["wsl"] = wsl_info(cfg["distro"])
        hw["wslconfig"] = wslconfig()
    engine_ready = not IS_WINDOWS or hw["wsl"]["distro_present"]
    hw["engine"] = engine_check(host) if with_engine and engine_ready else {}
    hw["recommendations"] = recommend(hw, cfg)
    return hw


# ---------------------------------------------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------------------------------------------
def wsl_memory_gb(total_gb):
    """Give WSL most of the RAM (the expert cache lives there) but keep Windows responsive."""
    return int(max(8, total_gb - max(8, total_gb * 0.125)))


def gpu_budget(gpu, cfg):
    """VRAM the model could use with nothing else running: total minus headroom and the desktop's share."""
    total = gpu["vram_total"]
    desktop = 1.5 * GB if gpu.get("display_active") else 0.4 * GB
    return max(0.0, total - desktop - min(cfg["full_vram_headroom_gb"] * GB, 0.1 * total + 0.4 * GB))


def recommend(hw, cfg):
    """[{level: ok|info|warn|error, title, detail, settings?}] plus the best settings for this machine."""
    out = []

    def add(level, title, detail="", settings=None):
        out.append({"level": level, "title": title, "detail": detail, "settings": settings or {}})
    if not hw["gpus"]:
        add("error", "No NVIDIA GPU found", hw.get("driver", {}).get("error") or "Install the NVIDIA driver.")
        return out
    g = hw["gpus"][0]
    if hw["host"].get("in_wsl"):
        add("warn", "The manager is running inside WSL", "It can't see Windows apps' GPU use from here, so it can't "
                                                         "make room for them. Run QwenFlash.cmd on Windows instead.")
    cc = float(g["compute_capability"] or 0)
    vram = g["vram_total"]
    if cc < 7.5:
        add("error", f"{g['name']} is too old", "The engine needs an NVIDIA GPU of the Turing generation (RTX 20) "
                                                "or newer.")
    elif cc < 8.0:
        add("warn", f"{g['name']}: Turing is untested", "The EXL3 kernels have Turing fallbacks, but this engine has "
                                                        "only been run on Ampere and newer.")
    budget = gpu_budget(g, cfg)
    quant = cfg["quant"]
    ctx_best = {q: sizing.max_context(q, budget) for q in sizing.QUANTS}
    if ctx_best["3.05"] == 0:
        add("error", f"{vram / GIB:.0f} GB of VRAM is not enough",
            f"With the desktop's share, about {budget / GB:.1f} GB would be left; the model needs at least "
            f"{sizing.min_budget('3.05', 16384) / GB:.1f} GB at 16K context. 8 GB cards are a research item "
            f"(README To-Do).")
    else:
        ctx = min(cfg["context"], ctx_best[quant] or ctx_best["3.05"])
        level = "ok" if vram >= 15e9 else "warn"
        tested = "" if vram >= 15e9 else " Below 16 GB is experimental: smaller prefill chunks and a smaller cache."
        add(level, f"{g['name']} ({vram / GIB:.0f} GB): up to {ctx_best[quant] // 1024}K context at {quant} bpw",
            f"About {budget / GB:.1f} GB for the model with nothing else on the GPU.{tested}",
            {"context": ctx} if ctx != cfg["context"] else None)
        if ctx_best["4.05"] == 0 and quant == "4.05":
            add("warn", "4.05 bpw does not fit this GPU", "Use 3.05 bpw.", {"quant": "3.05"})
    if len(hw["gpus"]) > 1:
        add("info", f"{len(hw['gpus'])} GPUs found: the engine uses GPU 0 ({g['name']})",
            "Multi-GPU is a research item; see docs/research/multi-gpu.md.")
    gen, width = g.get("pcie_gen_max"), g.get("pcie_width_max")
    if gen and width and (gen < 4 or width < 16):
        add("warn", f"GPU link is PCIe {gen}.0 x{width}",
            "Experts that miss the VRAM cache cross this link every token; Gen4 x16 or better is recommended. "
            "Check the slot (a second card or M.2 drive can halve the lanes).")
    elif gen and width:
        add("ok", f"GPU link PCIe {gen}.0 x{width}")
    if g.get("display_active"):
        add("info", "This GPU also drives the display", "About 1-2 GB of VRAM stays with the desktop.")

    # RAM: VRAM cache + RAM tier should hold every expert, or the rest is read from NVMe each time it's needed
    plan = sizing.plan(quant, budget, cfg["context"])
    need_gib = plan["ram_for_all_experts"] / GIB
    host_ram = hw["ram"]["total_gb"]
    usable = (hw.get("wslconfig", {}).get("memory_gb") or host_ram) if IS_WINDOWS else host_ram
    spare = usable - cfg["ram_reserve_gb"] - 6.5
    if spare >= need_gib:
        add("ok", f"RAM: every expert fits in VRAM + RAM ({need_gib:.0f} GiB RAM cache)")
    elif spare >= cfg["lean_ram_gb"]:
        add("info", f"RAM: {spare:.0f} of {need_gib:.0f} GiB for the expert cache",
            "The rest is read from the NVMe drive when needed. More RAM, or 3.05 bpw, keeps more of the model in "
            "memory.")
    else:
        add("warn", "Little RAM for the expert cache", f"About {max(0, spare):.0f} GiB can be spared.")
    if IS_WINDOWS:
        w, wc = hw["wsl"], hw["wslconfig"]
        if not w["installed"]:
            add("error", "WSL is not installed", "Run the 'WSL' setup step (needs administrator approval and "
                                                  "usually a restart).")
        elif w.get("distro_present") and w.get("distro_wsl_version") == 1:
            add("error", f"{w['distro']} runs under WSL 1", f"Convert it: wsl --set-version {w['distro']} 2")
        want = wsl_memory_gb(host_ram)
        have = wc.get("memory_gb")
        if have is None:
            add("warn", f"WSL may use only half the RAM ({host_ram / 2:.0f} GB)",
                f"Run the 'WSL memory' step to allow {want} GB.")
        elif have < want * 0.9:
            add("info", f"WSL is limited to {have:.0f} GB", f"{want} GB would leave more for the expert cache.")
        if not hw["cpu"].get("virtualization_firmware") and not w["installed"]:
            add("warn", "Virtualization looks disabled in the firmware", "WSL 2 needs it (Intel VT-x / AMD SVM).")
    e = hw.get("engine") or {}
    if e.get("fs") and e["fs"] not in ("ext2/ext3", "xfs", "btrfs"):
        add("warn", f"Engine is on a '{e['fs']}' filesystem", "The expert store needs O_DIRECT reads (ext4 or xfs).")
    if e.get("disk_free_gb"):
        q = sizing.QUANTS[quant]
        need = 0 if e.get("m305" if quant == "3.05" else "m405") == "1" else q["download_gb"] + q["store_gb"]
        if need and int(e["disk_free_gb"]) < need:
            add("error", f"Not enough disk space for {quant} bpw", f"Needs ~{need} GB, {e['disk_free_gb']} GB free.")
    return out

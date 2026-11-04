"""Deployment steps for both platforms, their status (from hardware detection) and a runner that streams output.

Windows + WSL2: WSL itself, its memory limit, the Ubuntu distro, then the shared Linux steps run inside the distro.
Native Linux:   the NVIDIA driver (checked; installing one needs a reboot), then the same shared steps.
The Linux steps are the scripts in setup/ (10_base, 20_env, 30_build, 50_model); every step can be re-run safely.
"""
import asyncio
import collections
import os
import re
from pathlib import Path

import psutil

from . import hardware
from .config import GB, IS_WINDOWS, REPO, log

W = "windows-wsl"
L = "linux"
STEPS = [
    {"id": "wsl", "title": "WSL 2", "platforms": [W], "blocks": True,
     "desc": "Installs or updates the Windows Subsystem for Linux. Windows asks for administrator approval; a first "
             "install usually needs a restart."},
    {"id": "wslconfig", "title": "WSL memory limit", "platforms": [W], "blocks": True,
     "desc": "Writes %USERPROFILE%\\.wslconfig so WSL can use most of the RAM (its default is half) for the expert "
             "cache. Restarts WSL."},
    {"id": "distro", "title": "Ubuntu 24.04", "platforms": [W], "blocks": True,
     "desc": "Installs the Ubuntu-24.04 distro on the chosen drive (use your fastest NVMe) with systemd enabled."},
    {"id": "driver", "title": "NVIDIA driver", "platforms": [L], "blocks": False,
     "desc": "Checks the driver. Installing one needs a reboot (and key enrollment with Secure Boot), so this step "
             "shows the commands instead of running them."},
    {"id": "base", "title": "Build tools + CUDA 12.8", "platforms": [W, L], "blocks": True, "script": "10_base.sh",
     "desc": "Compiler, git, Python headers and NVIDIA's CUDA 12.8 toolkit (no driver packages). Needs root/sudo."},
    {"id": "engine", "title": "Engine + Python environment", "platforms": [W, L], "blocks": True,
     "script": "20_env.sh",
     "desc": "Puts this repo's engine code where it runs (on Windows: a clone inside WSL, updated to this "
             "checkout's commit) and creates its Python environment with PyTorch (CUDA 12.8)."},
    {"id": "build", "title": "Compile GPU kernels", "platforms": [W, L], "blocks": True, "script": "30_build.sh",
     "desc": "Builds exllamav3 and the offload kernels for this GPU's architecture. Re-run after changing GPUs."},
    {"id": "m305", "title": "Model: 3.05 bpw (fastest)", "platforms": [W, L], "blocks": False,
     "script": "50_model.sh", "args": ["3.05"], "desc": "~85 GB download + ~47 GB expert store. Resumable."},
    {"id": "m405", "title": "Model: 4.05 bpw (higher fidelity)", "platforms": [W, L], "blocks": False,
     "script": "50_model.sh", "args": ["4.05"], "desc": "~107 GB download + ~62 GB expert store. Resumable."},
]


class Setup:
    def __init__(self, svc):
        self.svc = svc
        self.host = svc.host
        self.steps = [s for s in STEPS if self.host.kind in s["platforms"]]
        self.status = {s["id"]: "unknown" for s in self.steps}
        self.hw = None
        self.out = collections.deque(maxlen=5000)
        self.seq = 0
        self.running = None
        self.checking = False

    def emit(self, line):
        self.seq += 1
        self.out.append((self.seq, line.rstrip("\r\n")))

    @property
    def blocking(self):
        return self.running is not None and next(s for s in self.steps if s["id"] == self.running)["blocks"]

    def step(self, sid):
        return next((s for s in self.steps if s["id"] == sid), None)

    async def check(self):
        if self.checking:
            return self.hw
        self.checking = True
        try:
            hw = await asyncio.to_thread(hardware.detect, self.host, self.svc.cfg)
            self.hw = hw
            self.status.update(self.statuses(hw))
            if self.running:
                self.status[self.running] = "running"
            log.info("setup check: %s", {k: v for k, v in self.status.items()})
        except Exception as e:  # noqa: BLE001 - detection must never take the service down
            log.exception("setup check failed: %s", e)
        finally:
            self.checking = False
        return self.hw

    def statuses(self, hw):
        e = hw.get("engine") or {}
        st = {}
        if IS_WINDOWS:
            w, wc = hw["wsl"], hw["wslconfig"]
            st["wsl"] = "ok" if w["installed"] and w.get("default_version") in (2, None) else "missing"
            want = hardware.wsl_memory_gb(hw["ram"]["total_gb"])
            st["wslconfig"] = "ok" if (wc.get("memory_gb") or 0) >= want * 0.9 else "missing"
            st["distro"] = "ok" if w["distro_present"] and w["distro_wsl_version"] == 2 else "missing"
        else:
            st["driver"] = "ok" if hw["gpus"] else "missing"
        if not e or "error" in e:
            for sid in ("base", "engine", "build", "m305", "m405"):
                st[sid] = "missing" if (IS_WINDOWS and st.get("distro") == "missing") else "unknown"
            return st
        st["base"] = "ok" if e.get("cuda") == "1" and e.get("driver") == "1" else "missing"
        if e.get("engine") != "1" and e.get("engine_head") and IS_WINDOWS:
            st["engine"] = "outdated"           # an older (pre-integration) engine checkout: updating converts it
        elif e.get("engine") != "1" or e.get("env") != "1":
            st["engine"] = "missing"
        elif IS_WINDOWS and hw["repo"]["head"] and e.get("engine_head") != hw["repo"]["head"]:
            st["engine"] = "outdated"
        else:
            st["engine"] = "ok"
        st["build"] = "ok" if e.get("build") == "1" else "missing"
        st["m305"] = "ok" if e.get("m305") == "1" else "missing"
        st["m405"] = "ok" if e.get("m405") == "1" else "missing"
        return st

    async def run(self, sid):
        if self.running:
            return False, f"'{self.running}' is already running"
        step = self.step(sid)
        if not step:
            return False, "unknown step"
        self.running = sid
        self.status[sid] = "running"
        asyncio.create_task(self._run(step))
        return True, "started"

    def suggested_location(self):
        """Distro location: the configured one, else \\WSL\\<distro> on the fixed drive with the most free space."""
        cfg = self.svc.cfg
        if cfg["wsl_location"]:
            return cfg["wsl_location"]
        best = max((p for p in psutil.disk_partitions() if "fixed" in p.opts),
                   key=lambda p: psutil.disk_usage(p.mountpoint).free, default=None)
        drive = best.mountpoint if best else "C:\\"
        return str(Path(drive) / "WSL" / cfg["distro"])

    async def _run(self, step):
        host = self.host
        self.emit(f"===== {step['title']} =====")
        rc = 1
        try:
            if step["blocks"] and self.svc.be.state != "stopped" and not self.svc.observe:
                self.emit("stopping the model first (this step changes the engine or WSL)")
                await self.svc.stop("setup: " + step["title"])
            sid = step["id"]
            if sid == "wsl":
                rc = await self._wsl()
            elif sid == "wslconfig":
                rc = await self._wslconfig()
            elif sid == "distro":
                rc = await self._distro()
            elif sid == "driver":
                rc = self._driver()
            else:
                script = host.to_engine_path(REPO / "setup" / step["script"])
                args = [host.engine_dir, *step.get("args", [])]
                if sid == "engine" and IS_WINDOWS:
                    args.append(host.to_engine_path(REPO))        # clone/update the engine from this checkout
                rc = await host.stream(host.engine_argv("bash", script, *args), self.emit)
                if rc == 10:
                    self.emit("(needs sudo: run the command above in a terminal, then press Re-check)")
        except Exception as e:  # noqa: BLE001
            self.emit(f"ERROR: {e}")
            rc = 1
        ok = rc == 0
        self.emit(f"===== {step['title']}: {'done' if ok else f'FAILED (exit {rc})'} =====")
        self.svc.event("setup", f"{step['title']}: {'done' if ok else 'failed'}")
        self.running = None
        await self.check()
        if not ok:
            self.status[step["id"]] = "failed"

    async def _wsl(self):
        hw = self.hw or await self.check()
        if hw and hw["wsl"]["installed"]:
            self.emit("WSL is installed; updating it")
            return await self.host.stream(["wsl.exe", "--update"], self.emit)
        self.emit("installing WSL (approve the administrator prompt)...")
        ps = ("Start-Process wsl.exe -ArgumentList '--install','--no-distribution' -Verb RunAs -Wait "
              "-PassThru | ForEach-Object { exit $_.ExitCode }")
        rc = await self.host.stream(["powershell.exe", "-NoProfile", "-Command", ps], self.emit)
        if rc == 0:
            self.emit("WSL installed. If Windows asks for a restart, restart, then reopen QwenFlash and continue.")
        return rc

    async def _wslconfig(self):
        total_gb = psutil.virtual_memory().total / GB
        mem = hardware.wsl_memory_gb(total_gb)
        p = Path.home() / ".wslconfig"
        lines = p.read_text(errors="replace").splitlines() if p.exists() else []
        if p.exists():
            p.with_suffix(".bak").write_text("\n".join(lines) + "\n")
            self.emit(f"backed up the existing file to {p.with_suffix('.bak')}")
        # update [wsl2] memory/processors/swap in place; keep every other setting the user has
        want = {"memory": f"{mem}GB", "processors": str(os.cpu_count()), "swap": "8GB"}
        out, section, seen = [], None, set()
        for line in lines:
            m = re.match(r"\s*\[(.+?)\]", line)
            if m:
                if section == "wsl2":
                    out += [f"{k}={v}" for k, v in want.items() if k not in seen]
                section = m.group(1).strip().lower()
            elif section == "wsl2":
                k = line.split("=", 1)[0].strip().lower()
                if k in want:
                    out.append(f"{k}={want[k]}")
                    seen.add(k)
                    continue
            out.append(line)
        if section == "wsl2":
            out += [f"{k}={v}" for k, v in want.items() if k not in seen]
        elif "wsl2" not in {re.sub(r"[\[\]\s]", "", x).lower() for x in lines if x.strip().startswith("[")}:
            out = ["[wsl2]", "# QwenFlash: most of the RAM for the expert cache",
                   *[f"{k}={v}" for k, v in want.items()], ""] + out
        p.write_text("\n".join(out).strip() + "\n")
        self.emit(f"wrote {p}: memory={mem}GB processors={os.cpu_count()} swap=8GB")
        self.emit("restarting WSL so the limit applies (wsl --shutdown)")
        return await self.host.stream(["wsl.exe", "--shutdown"], self.emit)

    async def _distro(self):
        cfg = self.svc.cfg
        hw = self.hw or await self.check()
        if not (hw and hw["wsl"]["distro_present"]):
            loc = self.suggested_location()
            self.emit(f"installing {cfg['distro']} at {loc}")
            rc = await self.host.stream(["wsl.exe", "--install", "-d", cfg["distro"], "--location", loc,
                                         "--no-launch"], self.emit)
            if rc != 0:
                return rc
            self.emit("first start (unpacks the distro)...")
            await self.host.stream(["wsl.exe", "-d", cfg["distro"], "-u", "root", "--", "true"], self.emit)
        # the engine's launcher uses systemd scopes for its RAM guard
        rc = await self.host.stream(self.host.engine_argv(
            "bash", "-c", "grep -q '^systemd=true' /etc/wsl.conf 2>/dev/null && echo 'systemd already enabled' || "
                          "{ printf '[boot]\\nsystemd=true\\n' >> /etc/wsl.conf; echo 'enabled systemd'; }"), self.emit)
        await self.host.stream(["wsl.exe", "--terminate", cfg["distro"]], self.emit)
        return rc

    def _driver(self):
        hw = self.hw or {}
        if hw.get("gpus"):
            self.emit(f"driver {hw['driver'].get('version')} (CUDA {hw['driver'].get('cuda')}) sees "
                      f"{', '.join(g['name'] for g in hw['gpus'])}")
            return 0
        self.emit("No working NVIDIA driver. Install one, then reboot:")
        self.emit("  Ubuntu: sudo ubuntu-drivers install   (or: sudo apt install nvidia-driver-570-open)")
        self.emit("  Other distros: your distribution's NVIDIA driver package (570 or newer for RTX 50 cards)")
        return 1

    def report(self, since=0):
        return {"platform": self.host.kind, "steps": [dict(s, status=self.status.get(s["id"], "unknown"))
                                                      for s in self.steps],
                "running": self.running, "checking": self.checking, "seq": self.seq,
                "lines": [[n, t] for n, t in self.out if n > since][-800:],
                "location": self.suggested_location() if IS_WINDOWS else None}

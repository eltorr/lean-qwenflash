"""The benchmark (qwenflash/bench.py) against a fake model server: runs, workloads, records and tables, no GPU."""
import asyncio
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

os.environ["QWENFLASH_HOME"] = tempfile.mkdtemp(prefix="qf-test-")      # never a real settings dir
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qwenflash import bench, config  # noqa: E402
from qwenflash.service import Service  # noqa: E402

bench.SETTLE_S = 0
bench.WATCH_S, bench.YIELD_AFTER_S = 0.02, 0.05
KEY = "test-key"


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Answers like the engine: the manager's key on every request, streamed answers with usage at the end."""
    bodies, slow = [], 0.0

    def log_message(self, *a):
        pass

    def send_events(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01 + self.slow)
            self.wfile.write(b"data: [DONE]\n\n")
        except ConnectionError:
            pass                                        # the client left after the first token (prompt reading)

    def keyed(self):
        if self.headers.get("X-QwenFlash-Key") != KEY:
            self.send_response(403)
            self.end_headers()
            return False
        return True

    def do_GET(self):
        if not self.keyed():
            return
        body = {"/v1/models": {"data": [{"id": "qwen3.8-flash-next"}]},
                "/stats": {"expert_cache": [{"slots": 1234}], "nv2": {"host": {"vram": 90, "ram": 9, "nvme": 1}}}}
        data = json.dumps(body.get(self.path, {})).encode()
        self.send_response(200 if self.path in body else 404)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.keyed():
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.bodies.append((self.path, req))
        if self.path == "/tokenize":                        # a token for every 4 characters
            data = json.dumps({"tokens": [0] * (len(req["prompt"]) // 4)}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            prompt = req["messages"][0]["content"]
            reply = " ".join(c for _, _, c in bench.CODES) if "access codes" in prompt else "Air in, thrust out."
            self.send_events([{"choices": [{"delta": {"content": w + " "}}]} for w in reply.split()]
                             + [{"choices": [{"delta": {}}], "usage": {"prompt_tokens": len(prompt) // 4,
                                                                       "completion_tokens": 40}}])


class FakeHost:
    kind, follower = "linux", None

    def __init__(self):
        self.envs, self.state = [], "stopped"

    def backend_start(self, env, args):
        self.envs.append(env)
        self.state = "running"
        return True, 4242

    def backend_stop(self, timeout=30):
        self.state = "stopped"
        return "stopped"

    def backend_status(self):
        return self.state, (4242 if self.state == "running" else None)


class FakeService:
    """What the benchmark uses from the manager."""
    observe, gpu_display = False, True

    def __init__(self, port):
        self.cfg = dict(config.DEFAULTS, backend_port=port, quant="3.05", context=65536, start_timeout_s=20)
        self.host, self.learned = FakeHost(), {}
        self.setup = types.SimpleNamespace(running=None, check=self._check, hw={
            "gpus": [{"name": "Test GPU", "pcie_gen_max": 5, "pcie_width_max": 16}], "ram": {"total_gb": 68.7},
            "engine": {"driver": "1", "m305": "1", "m405": "0", "torch": "2.15.0+cu132", "cuda_version": "13.2"},
            "driver": {"version": "617.14"}, "repo": {"head": "abcdef0123"}})
        self.backend_key, self.backend_auth = KEY, {"X-QwenFlash-Key": KEY}
        self.be = types.SimpleNamespace(state="running", tier="full")
        self.game = False                                   # another app has the GPU
        self.mon = types.SimpleNamespace(get=self.sample)
        self.started, self.stopped = [], []

    def sample(self):
        busy = self.game() if callable(self.game) else self.game
        return {"t": time.time(), "vram_total": 32e9, "vram_used": 2e9, "ram_avail": 50e9, "perproc_ded": 1e9,
                "running": {"game.exe"} if busy else set(), "ext_util": 90.0 if busy else 0.0,
                "ext_util_top": [["game.exe", 90]] if busy else []}

    def ext_vram(self, m):
        return 2e9

    async def _check(self):
        pass

    async def fresh_sample(self, after):
        return self.mon.get()

    def headroom(self, m):
        return 3.2e9

    def lean_budget(self, m):
        return 16e9

    def full_budget(self, m):
        return 26.8e9

    def learned_key(self, quant=None, context=None):
        return f"gpu|{quant}|{context}"

    def ram_overhead(self, quant=None):
        return 9e9

    async def stop(self, reason, why):
        self.stopped.append(reason)
        self.be.state, self.be.tier = "stopped", None

    async def start(self, tier, reason):
        self.started.append(tier)
        self.be.state, self.be.tier = "running", tier
        return True


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeEngine.bodies.clear()
        FakeEngine.slow = 0.0
        self.svc = FakeService(self.server.server_address[1])

    def run_bench(self, runs, workload="quick", cancel_after=None):
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start(runs, workload)
            self.assertTrue(ok, msg)
            if cancel_after is not None:
                await asyncio.sleep(cancel_after)
                await b.cancel()
            await b.task
        asyncio.run(go())
        return b

    def test_a_run_measures_answers_and_prompt_reading_then_restores_the_model(self):
        b = self.run_bench(["current"])
        r = b.records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertEqual(r["label"], "3.05-card-64k")
        self.assertGreater(r["decode_tps_median"], 0)
        self.assertEqual(set(r["prefill"]), {"1024", "8192"})
        self.assertEqual(r["protocol"], "benchmark.py")
        self.assertEqual(r["engine"]["slots"], 1234)
        self.assertEqual(r["engine"]["picks_vram_pct"], 90.0)
        self.assertEqual(r["stack"], {"torch": "2.15.0+cu132", "cuda": "13.2", "driver": "617.14", "commit": "abcdef0"})
        self.assertTrue((b.dir / "3.05-card-64k.json").is_file())
        self.assertEqual((self.svc.stopped, self.svc.started), (["benchmark"], ["full"]))
        env = self.svc.host.envs[0]
        self.assertEqual(env["QWENFLASH_BACKEND_KEY"], KEY)
        self.assertEqual(int(env["QWEN_VRAM_LIMIT_BYTES"]), int(32e9 - 3.2e9))    # the whole card, as full runs
        self.assertEqual(self.svc.host.state, "stopped")                       # no benchmark server left behind

    def test_answers_follow_the_original_benchmark_and_are_never_capped(self):
        self.run_bench(["current"], "standard")
        chats = [body for path, body in FakeEngine.bodies if path == "/v1/chat/completions"]
        self.assertEqual(chats[0]["messages"][0]["content"], "Compute 17 times 23. Reply with only the number.")
        for body in chats:
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["temperature"], 0)                            # greedy, as benchmark.py
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
        docs = [b["messages"][0]["content"] for b in chats if "document 8192" in b["messages"][0]["content"]]
        self.assertEqual(len(docs), 3)                                          # one per trial
        self.assertTrue(docs[0].startswith("Trial 0, document 8192: Read the following document.\nThe fox"))
        self.assertTrue(docs[0].endswith("\nNow write exactly one sentence explaining why the sky is blue."))
        self.assertNotEqual(docs[0][:10], docs[1][:10])                         # nothing from the prefix cache
        self.assertAlmostEqual(len(docs[0]) // 4, 8192 - 40, delta=30)          # sized in tokens

    def test_the_standard_workload_finds_the_codes_in_the_long_document(self):
        r = self.run_bench(["current"], "standard").records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertTrue(r["long"]["codes_found"])
        self.assertAlmostEqual(r["long"]["prompt_tokens"], 65536 - 2048 - 40, delta=60)  # sized to the context
        self.assertEqual(len(r["decode_runs"]), 3)
        self.assertEqual(len(r["prefill"]["8192"]["runs"]), 3)
        self.assertEqual(set(r["prefill"]), {"1024", "8192", "32768"})          # 64K doesn't fit in a 64K context

    def test_an_emulated_card_gets_its_limit_guard_and_settings(self):
        self.run_bench(["3.05:10.5:131072:nostage:ram=12"])
        env = self.svc.host.envs[0]
        self.assertEqual(int(env["QWEN_VRAM_LIMIT_BYTES"]), int(2e9 + 10.5e9))
        self.assertEqual(int(env["QWEN_VRAM_GUARD_BYTES"]), int(2e9 + 10.5e9 + 0.25e9))
        self.assertEqual((env["QWEN_CHUNK"], env["RAM_GB"], env["CONTEXT"]), ("1536", "12", "131072"))

    def test_a_model_that_isnt_downloaded_is_skipped(self):
        r = self.run_bench(["4.05:card:65536"]).records[0]
        self.assertIn("isn't downloaded", r["error"])
        self.assertEqual(self.svc.host.envs, [])

    def test_a_second_run_is_compared_with_the_first(self):
        self.run_bench(["current"])
        r = self.run_bench(["current"]).records[0]
        self.assertIsNotNone(r["previous"])
        self.assertIsInstance(r["previous"]["decode_pct"], float)

    def test_cancel(self):
        FakeEngine.slow = 0.2
        b = self.run_bench(["current", "3.05:22:65536"], cancel_after=0.5)
        self.assertEqual([r.get("error") for r in b.records], ["cancelled"])
        self.assertEqual(self.svc.started, ["full"])

    def test_a_run_steps_aside_for_the_pc_and_starts_again_once_it_is_free(self):
        FakeEngine.slow = 0.1
        self.svc.cfg["recover_after_s"] = 0.2
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start(["current"], "quick")
            self.assertTrue(ok, msg)
            await asyncio.sleep(0.3)                        # mid-run, a game takes the GPU
            self.svc.game = True
            for _ in range(200):
                if b.waiting:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(b.waiting, ["GPU busy: game.exe at 90%"])
            await asyncio.sleep(0.3)                        # still waiting while the game runs
            self.assertEqual(len(self.svc.host.envs), 1)
            self.svc.game = False
            await b.task
        asyncio.run(go())
        lines = [x for _, x in b.out]
        self.assertIn("  stepping aside: GPU busy: game.exe at 90%", lines)
        self.assertIn("  the PC is free again", lines)
        self.assertEqual(len(self.svc.host.envs), 2)                            # the run started again
        r = b.records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertTrue(bench.counts(r))
        self.assertEqual(self.svc.started, ["full"])
        self.assertEqual(self.svc.host.state, "stopped")

    def test_a_watched_app_counts_and_a_run_interrupted_too_often_is_recorded(self):
        FakeEngine.slow = 0.1
        self.svc.cfg.update(recover_after_s=0, watchlist=["Game.exe"], ext_util_pct=101)
        self.svc.game = lambda: len(self.svc.host.envs) > 0 and self.svc.host.state == "running"   # every start
        b = self.run_bench(["current"])
        self.assertEqual(len(self.svc.host.envs), bench.MAX_YIELDS + 1)
        r = b.records[0]
        self.assertEqual(r["error"], "interrupted: watched app running: game.exe")
        self.assertTrue(r["interrupted"])
        self.assertFalse(bench.counts(r))                                       # resuming redoes it
        self.assertEqual(bench.unfinished()[0], b.dir)

    def test_a_benchmark_cut_short_resumes_where_it_stopped(self):
        FakeEngine.slow = 0.2
        first = self.run_bench(["current", "3.05:22:65536"], cancel_after=0.5)
        self.assertEqual([r.get("error") for r in first.records], ["cancelled"])
        self.assertEqual([r["label"] for r in bench.unfinished()[2]], ["3.05-card-64k", "3.05-22gb-64k"])
        FakeEngine.slow = 0.0
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start([], "standard", resume=True)
            self.assertTrue(ok, msg)
            self.assertIn("2 runs left, quick workload", msg)               # the benchmark's own workload
            await b.task
        asyncio.run(go())
        self.assertEqual(b.dir, first.dir)
        records = bench.load_records(b.dir)                                     # _plan.json isn't a result
        self.assertEqual(sorted(r["label"] for r in records), ["3.05-22gb-64k", "3.05-card-64k"])
        self.assertTrue(all(bench.counts(r) and not r.get("error") for r in records))
        self.assertIsNone(bench.unfinished())
        self.assertEqual(b.start([], resume=True), (False, "no unfinished benchmark to resume"))

    def test_requests_are_refused_while_it_runs(self):
        b = bench.Bench(self.svc)
        b.running = True
        self.assertEqual(b.start(["current"], "quick"), (False, "a benchmark is already running"))


class RunSpecTests(unittest.TestCase):
    cfg = dict(config.DEFAULTS, quant="4.05", context=262144)

    def test_presets(self):
        runs = bench.expand(["current", "quants", "lean"], self.cfg, lean_gb=16.0)
        self.assertEqual([r["label"] for r in runs],
                         ["4.05-card-256k", "3.05-card-256k", "4.05-card-256k", "4.05-16gb-256k-ram8"])
        matrix = bench.expand("matrix", self.cfg)
        self.assertEqual(len(matrix), len(bench.MATRIX))
        self.assertEqual({r["ram"] for r in matrix}, {38})                      # runs differ only in VRAM

    def test_bad_specs(self):
        for bad in ("5.0:card:65536", "3.05:2:65536", "3.05:card:999999999", "3.05:card:64k", "3.05:nan:65536",
                    "3.05:card:65536:turbo", "3.05:card:65536:ram=0", "3.05:card", ""):
            with self.assertRaises(ValueError, msg=bad):
                bench.expand([bad], self.cfg)
        with self.assertRaises(ValueError):
            bench.expand(["matrix"] * 3 + ["current"] * 5, self.cfg)        # too many runs


class TableTests(unittest.TestCase):
    def test_results_from_before_this_tool_still_render(self):
        records = bench.load_records(REPO / "research" / "vram-scaling" / "results")
        rows = bench.table(records).splitlines()
        self.assertEqual(len(rows), 2 + len(records))
        self.assertIn("**fails**: ran out of VRAM", "\n".join(rows))
        self.assertEqual(len(bench.measured_rows(records, "research/x").splitlines()), len(records) - 1)


class ApiTests(unittest.TestCase):
    def test_other_devices_cannot_start_or_cancel_a_benchmark(self):
        class Writer:
            data = b""

            def write(self, b):
                Writer.data += b

            async def drain(self):
                pass
        svc = types.SimpleNamespace()
        for route in ("bench", "bench/cancel"):
            Writer.data = b""
            path = "/_switcher/api/" + route
            asyncio.run(Service.api(svc, "POST", path, path, b'{"runs": ["current"]}', Writer(), local=False))
            self.assertTrue(Writer.data.startswith(b"HTTP/1.1 403"), Writer.data[:40])


if __name__ == "__main__":
    unittest.main()

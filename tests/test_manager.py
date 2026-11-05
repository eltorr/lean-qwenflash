"""Manager pieces that need no GPU: response timing, settings coercion, budget choice."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QWENFLASH_HOME"] = tempfile.mkdtemp(prefix="qf-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwenflash import config, hardware  # noqa: E402
from qwenflash.httpio import StreamStats  # noqa: E402

SSE = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n",
       b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
       b'data: {"choices":[{"delta":{"reasoning_content":"Hm"}}]}\n\n',
       b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],'
       b'"usage":{"prompt_tokens":12,"completion_tokens":40,"total_tokens":52}}\n\ndata: [DONE]\n\n')


class DriverTests(unittest.TestCase):
    def test_versions_come_from_setup_lib_sh(self):
        self.assertRegex(hardware.engine_cuda() or "", r"^\d+\.\d+$")
        self.assertRegex(hardware.lib_value("DRIVER_MIN") or "", r"^\d{3}$")

    def test_a_driver_older_than_the_engines_cuda(self):
        need = hardware.engine_cuda()
        major, minor = hardware.version(need)[:2]
        self.assertIsNone(hardware.driver_too_old({"driver": {"cuda": need}}))
        self.assertIsNone(hardware.driver_too_old({"driver": {"cuda": f"{major}.{minor + 1}"}}))
        self.assertIsNone(hardware.driver_too_old({"driver": {"error": "no driver"}}))
        old = hardware.driver_too_old({"driver": {"cuda": f"{major - 1}.9"}})
        self.assertEqual(old, (f"{major - 1}.9", need, hardware.lib_value("DRIVER_MIN")))


class StreamStatsTests(unittest.TestCase):
    def test_streaming_first_token_and_usage(self):
        s = StreamStats()
        for i, chunk in enumerate(SSE):
            s.feed(chunk, float(i))
        self.assertEqual(s.status, 200)
        self.assertEqual(s.t_first, 2.0)            # the role-only event is not a token
        self.assertEqual(s.usage()["completion_tokens"], 40)

    def test_plain_json_response(self):
        s = StreamStats()
        s.feed(b'HTTP/1.1 200 OK\r\n\r\n{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}', 1.0)
        self.assertIsNone(s.t_first)
        self.assertEqual(s.usage()["prompt_tokens"], 5)

    def test_headers_split_across_reads(self):
        s = StreamStats()
        s.feed(b"HTTP/1.1 503 Service", 0.0)
        s.feed(b" Unavailable\r\n\r\n{}", 0.1)
        self.assertEqual(s.status, 503)


class ConfigTests(unittest.TestCase):
    def test_coerce(self):
        self.assertIs(config.coerce("auto_switch", "off"), False)
        self.assertEqual(config.coerce("context", "131072"), 131072)
        self.assertEqual(config.coerce("watchlist", "game.exe\n blender ,"), ["game.exe", "blender"])

    def test_renamed_keys_migrate(self):
        config.CONFIG_PATH.write_text('{"windows_ram_reserve_gb": 9, "bogus": 1}')
        cfg = config.load_config()
        self.assertEqual(cfg["ram_reserve_gb"], 9)
        self.assertNotIn("bogus", cfg)


if __name__ == "__main__":
    unittest.main()


class StopTests(unittest.TestCase):
    """A stop counts only once the model server is gone (under memory pressure the stop command can fail)."""

    def run_stop(self, statuses):
        import asyncio
        import types
        from qwenflash import service
        calls = {"stop": 0}
        host = types.SimpleNamespace(after_stop=lambda: None)
        def backend_stop():
            calls["stop"] += 1
            if calls["stop"] == 1:
                raise OSError("wsl.exe: insufficient resources")
            return "stopped"
        host.backend_stop = backend_stop
        host.backend_status = lambda: (statuses.pop(0), None)
        events = []
        be = service.Backend()
        be.set("running")
        be.pid, be.tier = 123, "full"
        svc = types.SimpleNamespace(observe=False, be=be, host=host, event=lambda k, m: events.append((k, m)),
                                    ext_at_start=None, perproc_at_start=None)
        real_sleep = asyncio.sleep
        service.asyncio.sleep = lambda s: real_sleep(0)
        try:
            asyncio.run(service.Service._stop_locked(svc, "test"))
        finally:
            service.asyncio.sleep = real_sleep
        return be.state, events, calls["stop"]

    def test_a_failed_stop_is_retried(self):
        state, events, stops = self.run_stop(["stopped"])     # the first stop raises; the second works
        self.assertEqual(state, "stopped")
        self.assertEqual(stops, 2)

    def test_a_server_that_stays_is_still_tracked(self):
        state, events, stops = self.run_stop(["running", "running", "running"])
        self.assertEqual(state, "running")
        self.assertEqual(events[-1][0], "error")

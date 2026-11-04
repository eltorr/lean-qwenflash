"""Manager pieces that need no GPU: response timing, settings coercion, budget choice."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QWENFLASH_HOME"] = tempfile.mkdtemp(prefix="qf-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwenflash import config  # noqa: E402
from qwenflash.httpio import StreamStats  # noqa: E402

SSE = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n",
       b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
       b'data: {"choices":[{"delta":{"reasoning_content":"Hm"}}]}\n\n',
       b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],'
       b'"usage":{"prompt_tokens":12,"completion_tokens":40,"total_tokens":52}}\n\ndata: [DONE]\n\n')


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

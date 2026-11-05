"""The dashboard's guards: requests from other websites, DNS rebinding, and settings the HTTP API can't change."""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ["QWENFLASH_HOME"] = tempfile.mkdtemp(prefix="qf-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwenflash import config  # noqa: E402
from qwenflash.service import Service, host_name, same_origin  # noqa: E402


class OriginTests(unittest.TestCase):
    def test_programs_without_origin_pass(self):
        self.assertTrue(same_origin([("Host", "127.0.0.1:30000")]))

    def test_the_dashboard_itself_passes(self):
        self.assertTrue(same_origin([("Host", "127.0.0.1:30000"), ("Origin", "http://127.0.0.1:30000")]))
        self.assertTrue(same_origin([("host", "Omni:11434"), ("origin", "http://omni:11434")]))

    def test_other_sites_are_refused(self):
        for origin in ("https://evil.example", "http://127.0.0.1:8080", "null"):
            self.assertFalse(same_origin([("Host", "127.0.0.1:30000"), ("Origin", origin)]), origin)

    def test_host_name(self):
        for host, name in (("localhost:30000", "localhost"), ("[::1]:30000", "::1"), ("127.0.0.1", "127.0.0.1"),
                           ("Evil.Example:30000", "evil.example"), ("", ""), ("[bad", "")):
            self.assertEqual(host_name([("Host", host)]), name, host)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.svc = types.SimpleNamespace(cfg=dict(config.DEFAULTS), observe=True, event=lambda *a: None,
                                         be=types.SimpleNamespace(state="stopped"))

    def update(self, changes):
        return Service.update_config(self.svc, changes)

    def test_paths_programs_and_ports_only_in_config_json(self):
        for k, v in (("repo_dir", "/tmp/x'; id; '"), ("ui_browser", "C:/Windows/System32/cmd.exe"),
                     ("distro", "Other"), ("dashboard_token", "guess"), ("network_port", 8080)):
            status, _ = self.update({k: v})
            self.assertEqual(status, 400, k)
        self.assertEqual(self.svc.cfg, config.DEFAULTS)

    def test_a_bad_value_changes_nothing(self):
        status, _ = self.update({"watchlist": "game.exe", "remote_host": 'a" onmouseover="x'})
        self.assertEqual(status, 400)
        self.assertEqual(self.svc.cfg["watchlist"], [])

    def test_normal_settings_save(self):
        # the dashboard may resend a file-only setting unchanged
        status, r = self.update({"listen_port": "30000", "watchlist": "game.exe", "remote_host": "omni.example"})
        self.assertEqual(status, 200, r)
        self.assertEqual(self.svc.cfg["watchlist"], ["game.exe"])
        self.assertEqual(self.svc.cfg["remote_host"], "omni.example")


if __name__ == "__main__":
    unittest.main()

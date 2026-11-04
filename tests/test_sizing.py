"""The sizing model must reproduce what was measured and refuse plans that cannot work."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sizing  # noqa: E402

GB = 1e9
GIB = 1024 ** 3


class SizingTests(unittest.TestCase):
    def test_matches_rtx_5090_measurements(self):
        # full mode on the 5090: ~28 GB for the model; RAM_GB 26 (3.05) / ~40 (4.05) held every expert;
        # VRAM outside the cache measured 9.0 / 9.44 GB at 262K (research/vram-scaling)
        p = sizing.plan("3.05", 28 * GB, 262144)
        self.assertTrue(p["ok"] and p["staging"])
        self.assertEqual(p["chunk"], 24576)
        self.assertAlmostEqual(p["fixed"] / GB, 9.0, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 26.6, delta=1.0)
        p = sizing.plan("4.05", 28 * GB, 262144)
        self.assertAlmostEqual(p["fixed"] / GB, 9.44, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 41.0, delta=1.5)

    def test_staging_holds_a_whole_layer(self):
        self.assertEqual(sizing.stage_bytes("4.05"), 2 * 512 * 2_457_600)
        self.assertGreater(512 * 2_457_600 / GIB, 1.1)        # why the 1.1 GiB default overflowed at 4.05

    def test_small_cards_get_smaller_chunks(self):
        big = sizing.plan("3.05", 22 * GB, 65536)
        small = sizing.plan("3.05", 11 * GB, 65536)
        self.assertTrue(big["ok"] and small["ok"])
        self.assertLess(small["chunk"], big["chunk"])

    def test_refuses_what_cannot_fit(self):
        p = sizing.plan("3.05", 6 * GB, 16384)
        self.assertFalse(p["ok"])
        self.assertIn("not enough", p["reason"])
        self.assertEqual(sizing.max_context("3.05", 6 * GB), 0)

    def test_8gb_class_runs_staging_free_without_a_desktop(self):
        # an 8 GB card not driving a display leaves ~8 GB: no room for staging, but the staging-free path fits at 16K
        p = sizing.plan("3.05", 8 * GB, 16384)
        self.assertTrue(p["ok"])
        self.assertFalse(p["staging"])
        self.assertGreaterEqual(p["reserve"], 0.7 * GB)
        # driving a desktop as well (~7 GB left), long prompts overran it: refuse
        self.assertFalse(sizing.plan("3.05", 7 * GB, 16384)["ok"])

    def test_baseline_presets(self):
        def b(vram, ram, display=True, platform="linux"):
            return sizing.baseline(sizing.card_bytes(vram), sizing.ram_bytes(ram), display, platform)
        # this PC: RTX 5090, 64 GB, Windows: what it runs today
        p = b(32, 64, platform="windows-wsl")
        self.assertEqual((p["quant"], p["context"]), ("4.05", 262144))
        self.assertTrue(p["measured"])
        # 24 GB: 4.05 needs the RAM to hold the rest of its experts; 3.05 below that
        self.assertEqual(b(24, 128)["quant"], "4.05")
        self.assertEqual(b(24, 32)["quant"], "3.05")
        self.assertEqual(b(16, 128)["quant"], "3.05")         # 4.05 prefill drops to 4K chunks on 16 GB
        # small cards: 64K staging-free on 12 GB, 16K staging-free on 8 GB only without a display
        p = b(12, 64)
        self.assertEqual((p["context"], p["staging"]), (65536, False))
        p = b(8, 64, display=False)
        self.assertTrue(p["ok"])
        self.assertEqual((p["context"], p["staging"]), (16384, False))
        self.assertFalse(b(8, 64)["ok"])
        # RAM: 16 GB works on Linux with smaller thresholds, not on Windows
        p = b(16, 16)
        self.assertTrue(p["ok"])
        self.assertLess(p["settings"]["ram_reserve_gb"], 6)
        self.assertFalse(b(16, 16, platform="windows-wsl")["ok"])
        # a measurement only labels a preset with a similar RAM cache
        self.assertFalse(b(16, 32)["measured"])

    def test_readme_tables_are_generated(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        for name, body in (("presets", sizing.presets_table()), ("measured", sizing.measured_table())):
            self.assertIn(f"<!-- {name}:begin -->\n{body}\n<!-- {name}:end -->", readme,
                          f"README {name} table is stale: python3 sizing.py presets --write README.md")

    def test_budgeted_runs_pin_the_cache(self):
        p = sizing.plan("3.05", 16 * GB, 262144)
        self.assertIn("QWEN38_EC_MAX_SLOTS", sizing.engine_env(p, cap_slots=True))
        self.assertNotIn("QWEN38_EC_MAX_SLOTS", sizing.engine_env(sizing.plan("3.05", 28 * GB), cap_slots=False))


if __name__ == "__main__":
    unittest.main()

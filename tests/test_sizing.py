"""The sizing model must reproduce what was measured and refuse plans that cannot work."""
import unittest

import sizing

GB = 1e9
GIB = 1024 ** 3


class SizingTests(unittest.TestCase):
    def test_matches_rtx_5090_measurements(self):
        # full mode on the 5090: ~28 GB for the model; RAM_GB 26 (3.05) / ~40 (4.05) held every expert
        p = sizing.plan("3.05", 28 * GB, 262144)
        self.assertTrue(p["ok"] and p["staging"])
        self.assertEqual(p["chunk"], 24576)
        self.assertAlmostEqual(p["fixed"] / GB, 9.6, delta=0.05)
        self.assertAlmostEqual(p["ram_for_all_experts"] / GIB, 26.6, delta=1.0)
        p = sizing.plan("4.05", 28 * GB, 262144)
        self.assertAlmostEqual(p["fixed"] / GB, 10.0, delta=0.05)
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
        p = sizing.plan("3.05", 7 * GB, 16384)
        self.assertFalse(p["ok"])
        self.assertIn("not enough", p["reason"])
        self.assertEqual(sizing.max_context("3.05", 7 * GB), 0)

    def test_budgeted_runs_pin_the_cache(self):
        p = sizing.plan("3.05", 16 * GB, 262144)
        self.assertIn("QWEN38_EC_MAX_SLOTS", sizing.engine_env(p, cap_slots=True))
        self.assertNotIn("QWEN38_EC_MAX_SLOTS", sizing.engine_env(sizing.plan("3.05", 28 * GB), cap_slots=False))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from ltm_memory.benchmark import DEFAULT_SCENARIO_FILE, run_benchmark


class BenchmarkHarnessTests(unittest.TestCase):
    def test_default_scenario_file_exists(self) -> None:
        self.assertTrue(
            DEFAULT_SCENARIO_FILE.exists(),
            f"bundled scenario file missing: {DEFAULT_SCENARIO_FILE}",
        )

    def test_baseline_scenarios_all_pass(self) -> None:
        report = run_benchmark()
        self.assertEqual(report["benchmark"], "m2_baseline")
        self.assertGreater(report["summary"]["scenario_count"], 0)
        self.assertEqual(
            report["summary"]["expectations_failed"],
            0,
            msg=f"baseline regressions: {[s for s in report['scenarios'] if s['expectations_failed']]}",
        )
        self.assertEqual(report["summary"]["macro_hallucination_rate"], 0.0)
        for scenario in report["scenarios"]:
            self.assertIsNone(
                scenario["error"],
                msg=f"scenario {scenario['name']} errored: {scenario['error']}",
            )
            self.assertEqual(
                scenario["evidence_coverage"],
                1.0,
                msg=f"scenario {scenario['name']} lost evidence coverage",
            )


if __name__ == "__main__":
    unittest.main()

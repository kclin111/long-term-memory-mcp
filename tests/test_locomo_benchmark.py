from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from ltm_memory.benchmark_locomo import run_locomo_benchmark


class LocomoBenchmarkTests(unittest.TestCase):
    def test_tiny_locomo_retrieval_report(self) -> None:
        dataset = [
            {
                "sample_id": "tiny_1",
                "conversation": {
                    "speaker_a": "Caroline",
                    "speaker_b": "Melanie",
                    "session_1_date_time": "10:30 am on 7 May, 2023",
                    "session_1": [
                        {
                            "speaker": "Caroline",
                            "dia_id": "D1:1",
                            "text": "I ordered a Caesar salad for lunch.",
                        },
                        {
                            "speaker": "Melanie",
                            "dia_id": "D1:2",
                            "text": "That sounds fresh.",
                        },
                    ],
                    "session_2_date_time": "2:00 pm on 9 May, 2023",
                    "session_2": [
                        {
                            "speaker": "Caroline",
                            "dia_id": "D2:1",
                            "text": "The workshop moved to Room 204.",
                        }
                    ],
                },
                "qa": [
                    {
                        "question": "What did Caroline order for lunch?",
                        "answer": "Caesar salad",
                        "category": 1,
                        "evidence": ["D1:1"],
                    },
                    {
                        "question": "Where did the workshop move?",
                        "answer": "Room 204",
                        "category": 2,
                        "evidence": ["D2:1"],
                    },
                    {
                        "question": "This adversarial question should be filtered.",
                        "answer": "unused",
                        "category": 5,
                        "evidence": ["D1:2"],
                    },
                ],
            }
        ]
        tmp_dir = Path.cwd() / ".codex-tmp" / "test-locomo-benchmark"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp_dir / "locomo_tiny.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            report = run_locomo_benchmark(dataset_file=path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(report["benchmark"], "locomo_retrieval")
        self.assertEqual(report["summary"]["sample_count"], 1)
        self.assertEqual(report["summary"]["question_count"], 2)
        self.assertEqual(report["summary"]["samples_with_errors"], 0)
        self.assertGreaterEqual(report["summary"]["any_evidence_hit_rate"], 0.5)
        self.assertGreaterEqual(report["summary"]["answer_string_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

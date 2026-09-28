from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memorization_experiment.full_match_length_distribution import (
    FULL_MATCH_LENGTH_BINS,
    _length_bin_for,
    active_length_bins,
    load_full_match_length_distribution,
)
from memorization_experiment.run_memorization_experiments import DEFAULT_LENGTH_BUCKETS


class FullMatchLengthDistributionTests(unittest.TestCase):
    def test_selects_one_longest_full_raw_span_per_generation(self):
        rows = [
            {
                "generation": "first",
                "spans": [
                    {
                        "span_length": 2,
                        "docs": [{"match_tier": "exact_full_raw"}],
                    },
                    {
                        "span_length": 5,
                        "docs": [{"match_tier": "exact_full_raw"}],
                    },
                ],
            },
            {
                "generation": "partial only",
                "spans": [
                    {
                        "span_length": 80,
                        "docs": [{"match_tier": "partial"}],
                    }
                ],
            },
            {
                "generation": "normalized is not raw",
                "spans": [
                    {
                        "span_length": 40,
                        "docs": [{"match_tier": "exact_full_normalized"}],
                    },
                    {
                        "span_length": 160,
                        "docs": [{"match_tier": "exact_full_raw"}],
                    },
                ],
            },
            {
                "generation": "same bin",
                "spans": [
                    {
                        "span_length": 4,
                        "docs": [{"match_tier": "exact_full_raw"}],
                    }
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "experiment_summary.json"
            results_path = root / "experiment_results.json"
            summary_path.write_text("{}", encoding="utf-8")
            results_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            distribution = load_full_match_length_distribution(summary_path)

        self.assertEqual(distribution.total_generations, 4)
        self.assertEqual(distribution.full_match_count, 3)
        self.assertEqual(distribution.counts["(4, 6)"], 2)
        self.assertEqual(distribution.counts["(151, inf)"], 1)
        self.assertEqual(distribution.generation_ratio("(4, 6)"), 0.5)
        self.assertAlmostEqual(
            distribution.within_full_matches_percentage("(4, 6)"),
            200 / 3,
        )

        observed = active_length_bins({"series": distribution})
        self.assertEqual(
            [length_bin.label for length_bin in observed],
            ["(4, 6)", "(151, inf)"],
        )

    def test_standard_bins_follow_runner_defaults(self):
        expected_labels = []
        for raw_bucket in DEFAULT_LENGTH_BUCKETS.split(","):
            lower, upper = raw_bucket.split("-", 1)
            expected_labels.append(f"({int(lower)}, {upper})")

        self.assertEqual(
            [length_bin.label for length_bin in FULL_MATCH_LENGTH_BINS],
            expected_labels,
        )
        self.assertEqual(_length_bin_for(49).label, "(21, 49)")
        self.assertEqual(_length_bin_for(50).label, "(50, 75)")
        self.assertEqual(_length_bin_for(75).label, "(50, 75)")
        self.assertEqual(_length_bin_for(76).label, "(75, 99)")
        self.assertEqual(_length_bin_for(100).label, "(100, 150)")

        for length in range(1, 500):
            self.assertTrue(_length_bin_for(length).contains(length))


if __name__ == "__main__":
    unittest.main()

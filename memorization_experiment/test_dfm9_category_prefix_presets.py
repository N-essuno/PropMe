from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from memorization_experiment.plot_comparison_overviews import OVERVIEWS
from memorization_experiment.plot_memorization_results import PLOT_SUITES_BY_NAME
from memorization_experiment.run_memorization_experiments import (
    EXPERIMENTS_BY_NAME,
    INDEX_DEFAULTS,
    UNIGRAM_DEFAULTS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_propensity_module():
    module_path = REPO_ROOT / "05_propensity_metrics" / "compute_propensity_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "propme_compute_propensity_metrics",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Dfm9CategoryPrefixPresetTests(unittest.TestCase):
    def test_runner_has_all_generic_category_presets(self):
        for language in ("en", "da"):
            for category in ("a", "b", "c", "d"):
                upper = category.upper()
                experiment = EXPERIMENTS_BY_NAME[
                    f"dfm9-generations-generic-{language}-{category}"
                ]
                self.assertEqual(experiment.index_key, f"dfm9-{category}")
                self.assertEqual(experiment.unigram_key, f"dfm9-{category}")
                self.assertTrue(
                    experiment.dataset.endswith(
                        f"dfm9_generic_{language}_generations.json"
                    )
                )
                self.assertTrue(
                    experiment.summary_output.endswith(
                        f"st_dfm9_generic_{language}_{upper}_summary.json"
                    )
                )
                self.assertEqual(
                    INDEX_DEFAULTS[f"dfm9-{category}"],
                    f"/work/olmotrace/mimir_propme/indexes/{upper}",
                )
                self.assertTrue(
                    UNIGRAM_DEFAULTS[f"dfm9-{category}"].endswith(
                        f"unigram_probs_dfm9_{upper}.json"
                    )
                )

    def test_runner_has_matching_c_and_d_presets(self):
        for category in ("c", "d"):
            upper = category.upper()
            experiment = EXPERIMENTS_BY_NAME[
                f"dfm9-generations-prefix-{category}-50"
            ]
            self.assertEqual(experiment.index_key, f"dfm9-{category}")
            self.assertEqual(experiment.unigram_key, f"dfm9-{category}")
            self.assertTrue(
                experiment.dataset.endswith(
                    f"dfm9_{upper}_prefix_50_generations.json"
                )
            )
            self.assertTrue(
                experiment.summary_output.endswith(
                    f"st_dfm9_{upper}_prefix_50_summary.json"
                )
            )
            self.assertEqual(
                INDEX_DEFAULTS[f"dfm9-{category}"],
                f"/work/olmotrace/mimir_propme/indexes/{upper}",
            )
            self.assertTrue(
                UNIGRAM_DEFAULTS[f"dfm9-{category}"].endswith(
                    f"unigram_probs_dfm9_{upper}.json"
                )
            )

    def test_per_suite_and_overview_plots_include_c_and_d(self):
        suite_labels = [
            label for label, _ in PLOT_SUITES_BY_NAME["dfm9-generations"].filepaths
        ]
        self.assertEqual(
            suite_labels[:8],
            [
                "Generic EN / A",
                "Generic EN / B",
                "Generic EN / C",
                "Generic EN / D",
                "Generic DA / A",
                "Generic DA / B",
                "Generic DA / C",
                "Generic DA / D",
            ],
        )
        self.assertEqual(
            suite_labels[-4:],
            ["Prefix A/50", "Prefix B/50", "Prefix C/50", "Prefix D/50"],
        )

        overview = next(
            item for item in OVERVIEWS if item.name == "dfm9-settings-comparison"
        )
        prefix_setting = next(
            setting for setting in overview.settings if setting.name == "prefix"
        )
        generic_setting = next(
            setting for setting in overview.settings if setting.name == "generic"
        )
        self.assertEqual(len(generic_setting.filepaths), 8)
        self.assertEqual(
            [label for label, _ in prefix_setting.filepaths],
            ["A", "B", "C", "D"],
        )
        self.assertIn(
            "Generic EN vs Prefix C",
            [series.label for series in overview.propensity_series],
        )
        self.assertIn(
            "Generic DA vs Prefix D",
            [series.label for series in overview.propensity_series],
        )

    def test_propensity_presets_include_c_and_d(self):
        propensity_module = _load_propensity_module()
        for language in ("en", "da"):
            for category in ("c", "d"):
                preset = propensity_module.PRESETS_BY_NAME[
                    f"dfm9-generic-{language}-vs-prefix-{category}"
                ]
                upper = category.upper()
                self.assertTrue(
                    preset.prefix_summary.endswith(
                        f"st_dfm9_{upper}_prefix_50_summary.json"
                    )
                )
                self.assertTrue(
                    preset.setting_to_summary_paths[0][1].endswith(
                        f"st_dfm9_generic_{language}_{upper}_summary.json"
                    )
                )
                self.assertTrue(
                    preset.output.endswith(
                        f"prefix_{upper}_propensity.json"
                    )
                )


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memorization_experiment.full_match_length_distribution import (
    FULL_MATCH_LENGTH_BINS,
    FullMatchLengthDistribution,
)
from memorization_experiment.plot_memorization_results import (
    _require_plot_dependencies,
    _plot_exact_span_distribution,
    _plot_full_match_ratio,
    _plot_memorization_spider,
    _plot_scalar_metric,
    _plot_span_distribution_heatmap,
)


class PlotAxisLimitTests(unittest.TestCase):
    def setUp(self):
        _require_plot_dependencies()

    def tearDown(self):
        plt.close("all")

    def test_bounded_scalar_uses_zero_to_one(self):
        _, ax = plt.subplots()
        _plot_scalar_metric(
            ax,
            ["A", "B"],
            {"A": {"avg_nv_recall": 0.2}, "B": {"avg_nv_recall": 0.8}},
            "avg_nv_recall",
        )
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))

    def test_token_scalar_uses_one_hundred_and_marks_fifty(self):
        _, ax = plt.subplots()
        _plot_scalar_metric(
            ax,
            ["A", "B"],
            {"A": {"max_span_length": 25}, "B": {"max_span_length": 125}},
            "max_span_length",
        )
        self.assertEqual(ax.get_ylim(), (0.0, 100.0))
        self.assertTrue(
            any(
                line.get_linestyle() == "--"
                and set(line.get_ydata()) == {50.0}
                for line in ax.lines
            )
        )
        self.assertTrue(any("↑" in text.get_text() for text in ax.texts))

    def test_full_match_plot_uses_colored_percentage_labels(self):
        _, ax = plt.subplots()
        distributions = {
            "A": FullMatchLengthDistribution(
                total_generations=10,
                counts={"(1, 3)": 1, "(4, 6)": 1},
            ),
            "B": FullMatchLengthDistribution(total_generations=10, counts={}),
        }
        _plot_full_match_ratio(
            ax,
            ["A", "B"],
            {
                "A": {"generations_full_matches_ratio": 0.2},
                "B": {"generations_full_matches_ratio": 0.0},
            },
            distributions,
        )
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))
        self.assertEqual(len(ax.child_axes), 0)
        breakdown_labels = [
            text for text in ax.texts if text.get_bbox_patch() is not None
        ]
        self.assertEqual(
            [text.get_text() for text in breakdown_labels],
            ["50%", "50%"],
        )
        self.assertNotEqual(
            breakdown_labels[0].get_bbox_patch().get_facecolor(),
            breakdown_labels[1].get_bbox_patch().get_facecolor(),
        )
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            [length_bin.legend_label for length_bin in FULL_MATCH_LENGTH_BINS],
        )

    def test_span_distribution_heatmap_is_annotated_and_bounded(self):
        _, ax = plt.subplots()
        _plot_span_distribution_heatmap(
            ax,
            ["A", "B"],
            {
                "A": {
                    "spans_length_distribution": {
                        "(1, 3)": 0.75,
                        "(4, 6)": 0.25,
                    }
                },
                "B": {
                    "spans_length_distribution": {
                        "(1, 3)": 0.4,
                        "(4, 6)": 0.6,
                    }
                },
            },
        )
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(ax.images[0].get_clim(), (0.0, 1.0))
        self.assertEqual(
            [tick.get_text() for tick in ax.get_xticklabels()],
            ["1–3", "4–6"],
        )
        self.assertEqual(
            [text.get_text() for text in ax.texts],
            ["75%", "25%", "40%", "60%"],
        )

    def test_memorization_spider_normalizes_token_metric(self):
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        _plot_memorization_spider(
            ax,
            ["A", "B"],
            {
                "A": {
                    "generations_full_matches_ratio": 0.2,
                    "avg_nv_recall": 0.4,
                    "average_longest_span_length": 50,
                },
                "B": {
                    "generations_full_matches_ratio": 0.1,
                    "avg_nv_recall": 0.2,
                    "average_longest_span_length": 125,
                },
            },
        )
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(list(ax.lines[0].get_ydata()), [0.2, 0.4, 0.5, 0.2])
        self.assertEqual(list(ax.lines[1].get_ydata()), [0.1, 0.2, 1.0, 0.1])
        self.assertEqual(ax.lines[2].get_linestyle(), "--")
        self.assertTrue(
            all(value == 0.5 for value in ax.lines[2].get_ydata())
        )

    def test_exact_distribution_uses_token_and_ratio_limits(self):
        _, ax = plt.subplots()
        _plot_exact_span_distribution(
            ax,
            ["A"],
            {
                "A": {
                    "spans_length_distribution_exact": {
                        "1": 0.2,
                        "50": 0.1,
                        "150": 0.05,
                    }
                }
            },
        )
        self.assertEqual(ax.get_xlim(), (0.0, 100.0))
        self.assertEqual(ax.get_ylim(), (0.0, 1.0))
        self.assertIn(100.0, ax.get_xticks())
        self.assertTrue(
            any(
                line.get_linestyle() == "--"
                and set(line.get_xdata()) == {50.0}
                for line in ax.lines
            )
        )


    def test_exact_distribution_vector_style_is_vector_only(self):
        data = {
            "A": {
                "spans_length_distribution_exact": {
                    "1": 0.2,
                    "50": 0.1,
                    "100": 0.05,
                }
            }
        }
        _, png_ax = plt.subplots()
        _plot_exact_span_distribution(png_ax, ["A"], data)
        self.assertEqual(
            png_ax.get_title(),
            "spans_length_distribution_exact (smoothed)",
        )
        self.assertEqual(png_ax.get_xlabel(), "Span Length (tokens)")

        _, vector_ax = plt.subplots()
        _plot_exact_span_distribution(
            vector_ax,
            ["A"],
            data,
            vector_style=True,
        )
        self.assertEqual(vector_ax.get_title(), "")
        self.assertEqual(
            vector_ax.get_xlabel(),
            "Exact match length (tokens)",
        )
        self.assertEqual(vector_ax.lines[1].get_color(), "#DC2626")
        self.assertFalse(vector_ax.spines["top"].get_visible())
        self.assertFalse(vector_ax.spines["right"].get_visible())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .full_match_length_distribution import (
        FULL_MATCH_LENGTH_BINS,
        FullMatchLengthDistribution,
        load_full_match_length_distribution,
    )
except ImportError:
    from full_match_length_distribution import (
        FULL_MATCH_LENGTH_BINS,
        FullMatchLengthDistribution,
        load_full_match_length_distribution,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from generation_runs import (  # noqa: E402
    COMPARISONS,
    SETTING_LABELS,
    SETTINGS,
    Comparison,
    Corpus,
    Model,
    comparison_dir,
    run_plots_dir,
    run_name,
    runs,
    setting_tag,
    summary_path,
)

BASE_RED = "#A50922"
COLOR_A = "#F05039"
BOUNDED_SCALAR_METRICS = frozenset(
    {
        "generations_with_n_token_span_ratio",
        "generations_full_matches_ratio",
        "generations_with_nv_recall_ratio",
        "avg_nv_recall",
        "avg_nv_recall_on_hits",
        "generations_above_nv_recall_threshold_ratio",
    }
)
TOKEN_SCALAR_METRICS = frozenset(
    {
        "max_span_length",
        "average_longest_span_length",
    }
)


COLOR_D = "#1F449C"
COLOR_H = "#009E73"
VECTOR_SERIES_COLORS = (
    "#DC2626",
    "#1D4ED8",
    "#15803D",
    "#F0A96B",
    "#BE185D",
    "#111118",
    "#C87979",
    "#7AA8E8",
    "#6BBF8A",
    "#E08866",
    "#D67AA5",
    "#5DA87A",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
plt = None
np = None
to_rgb = None


@dataclass(frozen=True)
class PlotSuite:
    name: str
    filepaths: tuple[tuple[str, str], ...]
    plots_dir: str
    tags: tuple[str, ...]


DFM9_PLOT_SUITES = (
    PlotSuite(
        name="dfm9-generations",
        filepaths=(
            ("Generic EN / A", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_A_summary.json"),
            ("Generic EN / B", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_B_summary.json"),
            ("Generic EN / C", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_C_summary.json"),
            ("Generic EN / D", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_D_summary.json"),
            ("Generic DA / A", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_A_summary.json"),
            ("Generic DA / B", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_B_summary.json"),
            ("Generic DA / C", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_C_summary.json"),
            ("Generic DA / D", "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_D_summary.json"),
            ("Prefix A/50", "memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_summary.json"),
            ("Prefix B/50", "memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_summary.json"),
            ("Prefix C/50", "memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_summary.json"),
            ("Prefix D/50", "memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dfm9/plots/memorization",
        tags=("dfm9", "generations"),
    ),

)


def _run_suite(model: Model, corpus: Corpus) -> PlotSuite:
    """All settings of a generation run (see generation_runs.py)."""
    return PlotSuite(
        name=run_name(model, corpus),
        filepaths=tuple(
            (SETTING_LABELS[setting], summary_path(model, corpus, setting)) for setting in SETTINGS
        ),
        plots_dir=run_plots_dir(model, corpus),
        tags=(model.family, model.key, corpus.name, "generations"),
    )


def _comparison_suite(comparison: Comparison, setting: str) -> PlotSuite:
    """One setting across the runs of a comparison."""
    return PlotSuite(
        name=f"{comparison.name}-{setting_tag(setting)}",
        filepaths=tuple(
            (series.label, summary_path(series.model, series.corpus, setting))
            for series in comparison.all_series
        ),
        plots_dir=f"{comparison_dir(comparison)}/{setting}",
        tags=(comparison.name, "comparison", setting_tag(setting)),
    )


PLOT_SUITES = (
    DFM9_PLOT_SUITES
    + tuple(_run_suite(model, corpus) for model, corpus in runs())
    + tuple(
        _comparison_suite(comparison, setting)
        for comparison in COMPARISONS
        for setting in SETTINGS
    )
)

PLOT_SUITES_BY_NAME = {suite.name: suite for suite in PLOT_SUITES}

# One group per tag: a model family (dfm), model (dfm-main), corpus (dynaword),
# comparison (dfm-stages-dynaword), setting of the comparisons (generic), or dfm9.
GROUPS = {
    "all": [suite.name for suite in PLOT_SUITES],
    "all-generations": [suite.name for suite in PLOT_SUITES if "generations" in suite.tags],
    "all-comparisons": [suite.name for suite in PLOT_SUITES if "comparison" in suite.tags],
    "dfm9-generations": [
        suite.name for suite in PLOT_SUITES if "dfm9" in suite.tags and "generations" in suite.tags
    ],
    **{
        tag: [suite.name for suite in PLOT_SUITES if tag in suite.tags]
        for tag in dict.fromkeys(tag for suite in PLOT_SUITES for tag in suite.tags)
    },
}


def _require_plot_dependencies() -> None:
    global plt, np, to_rgb
    if plt is not None and np is not None and to_rgb is not None:
        return

    try:
        import matplotlib.pyplot as matplotlib_pyplot
        import numpy as numpy
        from matplotlib.colors import to_rgb as matplotlib_to_rgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "plot_simpletrace_results.py requires matplotlib and numpy for actual plot generation. "
            "Install the plotting dependencies, or use --list/--dry-run without them."
        ) from exc

    plt = matplotlib_pyplot
    np = numpy
    to_rgb = matplotlib_to_rgb


def _load_json(path: str):
    if not os.path.exists(path):
        print(f"\tWarning: File not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"\tWarning: Could not parse JSON file {path}: {exc}")
        return None


def _derive_exact_span_path(summary_path: str) -> str:
    base, ext = os.path.splitext(summary_path)
    if not ext:
        return f"{summary_path}_spans_length_exact.json"
    return f"{base}_spans_length_exact{ext}"


def _format_value(v: float) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _red_to_white_palette(n: int) -> list:
    if n <= 0:
        return []
    if n == 1:
        return [BASE_RED]
    red = np.array(to_rgb(BASE_RED), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    blend = np.linspace(0.0, 0.88, n)
    return [tuple((1.0 - a) * red + a * white) for a in blend]


def _value_based_red_colors(values: list[float]) -> list:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))

    if v_max <= v_min:
        return _red_to_white_palette(3)[1:2] * len(values)

    red = np.array(to_rgb(BASE_RED), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    norm = (arr - v_min) / (v_max - v_min)

    colors = []
    for n in norm:
        alpha = 0.88 * (1.0 - float(n))
        colors.append(tuple((1.0 - alpha) * red + alpha * white))
    return colors


def _distribution_palette(n: int) -> list:
    if n <= 0:
        return []
    base = [COLOR_A, COLOR_D, COLOR_H]
    if n <= len(base):
        return base[:n]
    return [plt.cm.tab10(i % 10) for i in range(n)]


def _metric_value(summary: dict, metric_name: str) -> float:
    aliases = {
        "max_span_length": ["max_span_length", "max_span", "average_longest_span"],
        "average_longest_span_length": [
            "average_longest_span_length",
            "average_longest_span",
            "average_span_length",
        ],
        "generations_with_n_token_span_ratio": [
            "generations_with_n_token_span_ratio",
            "generations_with_60_token_span_ratio",
        ],
        "generations_with_nv_recall_ratio": ["generations_with_nv_recall_ratio"],
        "avg_nv_recall": ["avg_nv_recall"],
        "avg_nv_recall_on_hits": ["avg_nv_recall_on_hits"],
        "generations_above_nv_recall_threshold_ratio": [
            "generations_above_nv_recall_threshold_ratio",
        ],
    }
    candidates = aliases.get(metric_name, [metric_name])
    for key in candidates:
        if key in summary:
            value = summary[key]
            return value if value is not None else 0.0
    return 0.0


def _plot_scalar_metric(ax, labels, summaries, metric_name: str):
    values = [
        float(_metric_value(summaries[label], metric_name)) for label in labels
    ]
    is_token_metric = metric_name in TOKEN_SCALAR_METRICS
    is_bounded_metric = (
        metric_name in BOUNDED_SCALAR_METRICS
        or metric_name.startswith("k_eidetic_rate_k_le_")
    )
    palette = _value_based_red_colors(values)
    bars = ax.bar(labels, values, color=palette, edgecolor="#333333", linewidth=1.0)
    for bar in bars:
        height = float(bar.get_height())
        is_clipped = is_token_metric and height > 100
        annotation_height = 100.0 if is_clipped else height
        ax.annotate(
            f"{_format_value(height)} ↑" if is_clipped else _format_value(height),
            xy=(bar.get_x() + bar.get_width() / 2, annotation_height),
            xytext=(0, -4) if is_clipped else (0, 3),
            textcoords="offset points",
            ha="center",
            va="top" if is_clipped else "bottom",
            fontsize=8,
        )
    ax.set_title(metric_name)
    ax.set_ylabel("Tokens" if is_token_metric else "Value")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)
    if is_token_metric:
        ax.set_ylim(0.0, 100.0)
        ax.axhline(50.0, color="#555555", linestyle="--", linewidth=1.2)
        ax.text(
            0.99,
            50.0,
            "50 tokens",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
    elif is_bounded_metric:
        ax.set_ylim(0.0, 1.0)


def _plot_full_match_ratio(
    ax,
    labels: list[str],
    summaries: dict[str, dict],
    distributions: dict[str, FullMatchLengthDistribution],
) -> None:
    """Plot full-match ratios with compact length-bin labels above each bar."""
    x = np.arange(len(labels), dtype=float)
    length_bins = list(FULL_MATCH_LENGTH_BINS)
    aggregate_values = [
        float(_metric_value(summaries[label], "generations_full_matches_ratio"))
        for label in labels
    ]

    def draw_stacks(target_ax, *, annotate_segments: bool) -> None:
        bottoms = np.zeros(len(labels), dtype=float)
        for length_bin in length_bins:
            values = [
                distributions[label].generation_ratio(length_bin.label)
                for label in labels
            ]
            bars = target_ax.bar(
                x,
                values,
                bottom=bottoms,
                color=length_bin.color,
                edgecolor="#333333",
                linewidth=0.7,
                label=length_bin.legend_label,
            )
            if annotate_segments:
                for index, (bar, value) in enumerate(zip(bars, values)):
                    if value <= 0:
                        continue
                    percentage = distributions[
                        labels[index]
                    ].within_full_matches_percentage(length_bin.label)
                    percentage_label = (
                        f"{percentage:.0f}%"
                        if percentage >= 10
                        else f"{percentage:.1f}%"
                    )
                    target_ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[index] + value / 2,
                        percentage_label,
                        ha="center",
                        va="center",
                        color=length_bin.text_color,
                        fontsize=6,
                        fontweight="bold",
                    )
            bottoms += np.array(values, dtype=float)

    draw_stacks(ax, annotate_segments=False)
    for index, value in enumerate(aggregate_values):
        ax.annotate(
            f"{value:.1%}",
            xy=(x[index], value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        detail_index = 0
        for length_bin in length_bins:
            percentage = distributions[
                labels[index]
            ].within_full_matches_percentage(length_bin.label)
            if percentage <= 0:
                continue
            percentage_label = (
                f"{percentage:.0f}%"
                if percentage >= 10
                else f"{percentage:.1f}%"
            )
            ax.annotate(
                percentage_label,
                xy=(x[index], value),
                xytext=(0, 20 + detail_index * 15),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=length_bin.text_color,
                fontsize=6,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": length_bin.color,
                    "edgecolor": "#333333",
                    "linewidth": 0.6,
                },
                annotation_clip=False,
            )
            detail_index += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("generations_full_matches_ratio")
    ax.set_ylabel("Share of generations")
    ax.yaxis.set_major_formatter("{x:.0%}")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.0)
    if length_bins:
        ax.legend(
            title="Full-match length",
            loc="upper right",
            ncol=min(3, len(length_bins)),
            fontsize=7,
            title_fontsize=8,
        )


def _extract_k_values(summaries: dict[str, dict]) -> list[int]:
    k_values: set[int] = set()
    for summary in summaries.values():
        for k in summary.get("k_eidetic_k_values", []):
            try:
                k_values.add(int(k))
            except (TypeError, ValueError):
                continue
        for key in summary.keys():
            m = re.match(r"^k_eidetic_rate_k_le_(\d+)$", key)
            if m:
                k_values.add(int(m.group(1)))
    return sorted(k_values)


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _plot_exact_span_distribution(
    ax,
    labels,
    exact_span_data: dict[str, dict],
    *,
    vector_style: bool = False,
):
    all_lengths: set[int] = set()
    for payload in exact_span_data.values():
        dist = payload.get("spans_length_distribution_exact", {})
        for k in dist.keys():
            try:
                all_lengths.add(int(k))
            except (TypeError, ValueError):
                continue

    if not all_lengths:
        ax.text(0.5, 0.5, "No exact span distribution data found", ha="center", va="center")
        ax.set_axis_off()
        return

    x = sorted(all_lengths)
    palette = (
        list(VECTOR_SERIES_COLORS)
        if vector_style
        else _distribution_palette(len(labels))
    )
    smoothing_window = 3 if len(x) < 20 else (5 if len(x) < 60 else 7)

    for i, label in enumerate(labels):
        payload = exact_span_data.get(label, {})
        dist = payload.get("spans_length_distribution_exact", {})
        y = [dist.get(str(length), dist.get(length, 0.0)) for length in x]
        color = palette[i % len(palette)]
        ax.plot(x, y, linewidth=0.9, alpha=0.20, color=color)
        y_smooth = _moving_average(y, smoothing_window)
        ax.plot(x, y_smooth, linewidth=2.2, color=color, label=label)

    tick_step = max(1, len(x) // 12)
    ticks = x[::tick_step]
    if ticks[-1] != x[-1]:
        ticks = ticks + [x[-1]]

    ticks = [tick for tick in ticks if tick <= 100]
    if 100 not in ticks:
        ticks.append(100)
    if not vector_style:
        ax.set_title("spans_length_distribution_exact (smoothed)")
    ax.set_ylim(0.0, 1.0)
    ax.axvline(50.0, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(
        50.0,
        0.99,
        "50 tokens",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7,
    )
    ax.set_xlabel(
        "Exact match length (tokens)"
        if vector_style
        else "Span Length (tokens)"
    )
    ax.set_ylabel("Ratio")
    ax.set_xticks(ticks)
    ax.set_xlim(0.0, 100.0)
    if vector_style:
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.tick_params(axis="both", colors="#4B5563", labelsize=12)
        ax.xaxis.label.set_color("#111118")
        ax.yaxis.label.set_color("#111118")
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color("#E5E7EB")
            spine.set_linewidth(0.8)
        legend = ax.legend(frameon=False, fontsize=11)
        for text in legend.get_texts():
            text.set_color("#111118")
    else:
        ax.grid(axis="both", linestyle="--", alpha=0.5)
        ax.legend()


def _bucket_sort_key(bucket_label: str) -> tuple[int, float, str]:
    match = re.match(r"^\(\s*(\d+)\s*,\s*([0-9]+|inf)\s*\)$", str(bucket_label))
    if not match:
        return (10**9, float("inf"), str(bucket_label))
    lo = int(match.group(1))
    hi_raw = match.group(2)
    hi = float("inf") if hi_raw == "inf" else int(hi_raw)
    return (lo, hi, str(bucket_label))


def _plot_bucketed_span_distribution(ax, labels, summaries: dict[str, dict]):
    all_buckets: set[str] = set()
    for summary in summaries.values():
        dist = summary.get("spans_length_distribution", {})
        for b in dist.keys():
            all_buckets.add(str(b))

    if not all_buckets:
        ax.text(0.5, 0.5, "No bucketed span distribution data found", ha="center", va="center")
        ax.set_axis_off()
        return

    buckets = sorted(all_buckets, key=_bucket_sort_key)
    x = np.arange(len(buckets))
    width = 0.8 / max(len(labels), 1)
    palette = _distribution_palette(len(labels))

    for i, label in enumerate(labels):
        dist = summaries[label].get("spans_length_distribution", {})
        values = [dist.get(b, 0.0) for b in buckets]
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )

    label_step = max(1, len(buckets) // 12)
    tick_labels = [b if (idx % label_step == 0 or idx == len(buckets) - 1) else "" for idx, b in enumerate(buckets)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_title("spans_length_distribution")
    ax.set_xlabel("Span Length Bucket")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(0.0, 1.0)

def _pretty_bucket_label(bucket: str) -> str:
    match = re.match(r"^\(\s*(\d+)\s*,\s*([0-9]+|inf)\s*\)$", str(bucket))
    if not match:
        return str(bucket)
    lower, upper = match.groups()
    return f"{lower}+" if upper == "inf" else f"{lower}–{upper}"


def _plot_span_distribution_heatmap(
    ax,
    labels: list[str],
    summaries: dict[str, dict],
) -> None:
    """Plot the binned span distributions as an annotated ratio heatmap."""
    all_buckets = {
        str(bucket)
        for summary in summaries.values()
        for bucket in summary.get("spans_length_distribution", {})
    }
    if not all_buckets:
        ax.text(
            0.5,
            0.5,
            "No bucketed span distribution data found",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    buckets = sorted(all_buckets, key=_bucket_sort_key)
    matrix = np.array(
        [
            [
                float(
                    summaries[label]
                    .get("spans_length_distribution", {})
                    .get(bucket, 0.0)
                )
                for bucket in buckets
            ]
            for label in labels
        ],
        dtype=float,
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            percentage = value * 100.0
            value_label = (
                f"{percentage:.0f}%"
                if percentage >= 10
                else f"{percentage:.1f}%"
            )
            ax.text(
                column_index,
                row_index,
                value_label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.55 else "#111111",
            )

    ax.set_xticks(np.arange(len(buckets)))
    ax.set_xticklabels([_pretty_bucket_label(bucket) for bucket in buckets])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Span length bin (tokens)")
    ax.set_ylabel("Evaluation setting")
    ax.set_title("Span Length Distribution by Bin")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Ratio")
    colorbar.ax.set_ylim(0.0, 1.0)


def _plot_memorization_spider(
    ax,
    labels: list[str],
    summaries: dict[str, dict],
) -> None:
    """Compare three headline memorization metrics on normalized axes."""
    metric_labels = (
        "Full-generation\nmatches",
        "Average NV\nrecall",
        "Average longest span\n(100-token scale)",
    )
    angles = np.linspace(0.0, 2.0 * np.pi, len(metric_labels), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    palette = _distribution_palette(len(labels))

    for index, label in enumerate(labels):
        summary = summaries[label]
        full_match_ratio = float(
            _metric_value(summary, "generations_full_matches_ratio")
        )
        average_nv_recall = float(_metric_value(summary, "avg_nv_recall"))
        average_longest_span = float(
            _metric_value(summary, "average_longest_span_length")
        )
        values = np.array(
            [
                full_match_ratio,
                average_nv_recall,
                average_longest_span / 100.0,
            ],
            dtype=float,
        )
        values = np.clip(values, 0.0, 1.0)
        closed_values = np.concatenate([values, values[:1]])
        color = palette[index % len(palette)]
        ax.plot(
            closed_angles,
            closed_values,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=5,
            label=(
                f"{label}  ·  {full_match_ratio:.1%}  |  "
                f"{average_nv_recall:.1%}  |  {average_longest_span:.1f} tok"
            ),
        )
        ax.fill(closed_angles, closed_values, color=color, alpha=0.12)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20%", "40%", "50%", "60%", "80%", "100%"],
        fontsize=7,
    )
    ax.set_rlabel_position(25)
    ax.grid(alpha=0.5)
    reference_angles = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        reference_angles,
        np.full_like(reference_angles, 0.5),
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label="_nolegend_",
        zorder=0,
    )
    ax.annotate(
        "50 tokens",
        xy=(angles[2], 0.5),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#444444",
    )
    ax.set_title("Memorization Metrics", pad=24)
    ax.legend(
        title="Setting  ·  full match | NV recall | avg span",
        loc="upper left",
        bbox_to_anchor=(1.08, 1.08),
        fontsize=8,
    )


def _save_spider_plot(
    plots_dir: str,
    labels: list[str],
    summaries: dict[str, dict],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={"projection": "polar"})
    _plot_memorization_spider(ax, labels, summaries)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "memorization_metrics_spider.png"),
        dpi=200,
    )
    plt.close(fig)


def _save_single_plot(plots_dir: str, filename: str, plotter):
    fig, ax = plt.subplots(figsize=(9, 6))
    plotter(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200)
    plt.close(fig)


def _save_exact_span_distribution_vectors(
    plots_dir: str,
    labels: list[str],
    exact_span_data: dict[str, dict],
) -> None:
    vector_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "text.color": "#111118",
        "axes.labelcolor": "#111118",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
    }
    with plt.rc_context(vector_rc):
        fig, ax = plt.subplots(figsize=(13.27, 5.92))
        _plot_exact_span_distribution(
            ax,
            labels,
            exact_span_data,
            vector_style=True,
        )
        fig.tight_layout()
        for suffix in ("svg", "pdf"):
            fig.savefig(
                os.path.join(
                    plots_dir,
                    f"spans_length_distribution_exact.{suffix}",
                )
            )
        plt.close(fig)


def _ensure_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _expand_targets(raw_targets: list[str]) -> list[PlotSuite]:
    names: list[str] = []
    seen: set[str] = set()

    for raw_target in raw_targets:
        for target in (token.strip() for token in raw_target.split(",") if token.strip()):
            if target in GROUPS:
                expanded_names = GROUPS[target]
            elif target in PLOT_SUITES_BY_NAME:
                expanded_names = [target]
            else:
                valid = ", ".join(sorted({*GROUPS.keys(), *PLOT_SUITES_BY_NAME.keys()}))
                raise SystemExit(
                    f"Unknown target '{target}'. Use --list to inspect choices.\n\nValid targets:\n{valid}"
                )

            for name in expanded_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

    return [PLOT_SUITES_BY_NAME[name] for name in names]


def _print_available_targets() -> None:
    print("Groups:")
    for group_name in sorted(GROUPS):
        print(f"  {group_name}")

    print("\nPlot suites:")
    for suite in PLOT_SUITES:
        tags = ", ".join(suite.tags)
        print(f"  {suite.name} [{tags}]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SimpleTrace plots from named summary presets."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help="Plot suite names and/or group names to run. Use --list to inspect choices.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available plot suite names and group names, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected plot suites without generating plots.",
    )
    return parser.parse_args()


def _generate_plots_for_suite(suite: PlotSuite) -> None:
    _require_plot_dependencies()
    plots_dir = _ensure_path(suite.plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    summaries: dict[str, dict] = {}
    exact_span_data: dict[str, dict] = {}
    full_match_length_data: dict[str, FullMatchLengthDistribution] = {}

    for label, summary_path_str in suite.filepaths:
        summary_path = _ensure_path(summary_path_str)
        summary = _load_json(str(summary_path))
        if summary is None:
            continue
        summaries[label] = summary

        exact_path_str = summary.get("spans_length_exact_output_path", "")
        if not exact_path_str:
            exact_path_str = _derive_exact_span_path(str(summary_path))
        exact_payload = _load_json(str(_ensure_path(exact_path_str)))
        if exact_payload is None:
            exact_payload = {}
        try:
            full_match_length_data[label] = load_full_match_length_distribution(
                summary_path
            )
        except (OSError, ValueError) as exc:
            print(
                f"\tWarning: Could not load full-match lengths for {label}: {exc}"
            )
            full_match_length_data[label] = FullMatchLengthDistribution(
                total_generations=int(summary.get("total_generations", 0)),
                counts={},
            )
        exact_span_data[label] = exact_payload

    if not summaries:
        print(f"Warning: No valid summary JSON files were found for {suite.name}. Skipping.")
        return

    labels = list(summaries.keys())
    scalar_metrics = [
        "max_span_length",
        "average_longest_span_length",
        "generations_with_n_token_span_ratio",
        "generations_full_matches_ratio",
        "generations_with_nv_recall_ratio",
        "avg_nv_recall",
        "avg_nv_recall_on_hits",
        "generations_above_nv_recall_threshold_ratio",
    ]

    k_values = _extract_k_values(summaries)
    k_metrics = [f"k_eidetic_rate_k_le_{k}" for k in k_values]

    for metric in scalar_metrics:
        if metric == "generations_full_matches_ratio":
            plotter = lambda ax: _plot_full_match_ratio(
                ax, labels, summaries, full_match_length_data
            )
        else:
            plotter = lambda ax, metric_name=metric: _plot_scalar_metric(
                ax, labels, summaries, metric_name
            )
        _save_single_plot(
            str(plots_dir),
            f"{metric}.png",
            plotter,
        )

    for metric in k_metrics:
        _save_single_plot(
            str(plots_dir),
            f"{metric}.png",
            lambda ax, metric_name=metric: _plot_scalar_metric(ax, labels, summaries, metric_name),
        )

    _save_single_plot(
        str(plots_dir),
        "spans_length_distribution_exact.png",
        lambda ax: _plot_exact_span_distribution(ax, labels, exact_span_data),
    )
    _save_exact_span_distribution_vectors(
        str(plots_dir),
        labels,
        exact_span_data,
    )

    _save_single_plot(
        str(plots_dir),
        "spans_length_distribution.png",
        lambda ax: _plot_bucketed_span_distribution(ax, labels, summaries),
    )

    _save_single_plot(
        str(plots_dir),
        "span_length_distribution_heatmap.png",
        lambda ax: _plot_span_distribution_heatmap(ax, labels, summaries),
    )

    _save_spider_plot(str(plots_dir), labels, summaries)

    combined_specs = (
        [("scalar", metric) for metric in scalar_metrics]
        + [("k", metric) for metric in k_metrics]
        + [
            ("distribution_exact", "spans_length_distribution_exact"),
            ("distribution_bucketed", "spans_length_distribution"),
        ]
    )

    n_plots = len(combined_specs)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.8 * n_rows))
    axes_flat = np.array(axes).reshape(-1)

    for ax, (kind, metric_name) in zip(axes_flat, combined_specs):
        if metric_name == "generations_full_matches_ratio":
            _plot_full_match_ratio(
                ax, labels, summaries, full_match_length_data
            )
        elif kind in ("scalar", "k"):
            _plot_scalar_metric(ax, labels, summaries, metric_name)
        elif kind == "distribution_exact":
            _plot_exact_span_distribution(ax, labels, exact_span_data)
        else:
            _plot_bucketed_span_distribution(ax, labels, summaries)

    for ax in axes_flat[n_plots:]:
        ax.set_axis_off()

    fig.suptitle(
        f"SimpleTrace Metrics Across {len(labels)} Evaluation Settings",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(plots_dir / "all_plots_combined.png", dpi=220)
    plt.close(fig)

    print(f"Successfully generated plots for {suite.name} in: {plots_dir}")


def main():
    args = _parse_args()

    if args.list:
        _print_available_targets()
        return

    if not args.targets:
        raise SystemExit("No targets provided. Use --list to inspect the available plot suites and groups.")

    suites = _expand_targets(args.targets)
    if not suites:
        print("No plot suites selected.")
        return

    print("Selected plot suites:")
    for suite in suites:
        print(f"  - {suite.name}")
    print()

    if args.dry_run:
        for suite in suites:
            print(f"{suite.name} -> {suite.plots_dir}")
        return

    for suite in suites:
        try:
            _generate_plots_for_suite(suite)
        except Exception as exc:
            print(f"\tWarning: Failed to generate plots for {suite.name}: {exc}. Skipping.")


if __name__ == "__main__":
    main()

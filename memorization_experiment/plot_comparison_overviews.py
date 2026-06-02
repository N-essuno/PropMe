#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


COLOR_A = "#F05039"
COLOR_D = "#1F449C"
COLOR_H = "#009E73"
FALLBACK_COLORS = ("#7A4F9A", "#7F7F7F", "#BCBD22")
SETTING_ORDER = ("generic", "specific", "prefix")
SCALAR_METRICS = (
    "average_longest_span_length",
    "avg_nv_recall",
    "generations_full_matches_ratio",
)
PROPENSITY_OUTPUT_NAME = "propensity_metrics_overview.png"
SPAN_OUTPUT_NAME = "spans_length_distribution_overview.png"
REPO_ROOT = Path(__file__).resolve().parents[1]

plt = None
np = None


@dataclass(frozen=True)
class OverviewSetting:
    name: str
    filepaths: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PropensitySeries:
    label: str
    summary_path: str


@dataclass(frozen=True)
class ComparisonOverview:
    name: str
    output_dir: str
    settings: tuple[OverviewSetting, ...]
    propensity_series: tuple[PropensitySeries, ...]


OVERVIEWS = (
    ComparisonOverview(
        name="dynaword-stages-comparison",
        output_dir="memorization_experiment/data/dynaword_stages_comparison",
        settings=(
            OverviewSetting(
                name="generic",
                filepaths=(
                    ("generic_stage1", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
                    ("generic_stage2", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
                    ("generic", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
                ),
            ),
            OverviewSetting(
                name="specific",
                filepaths=(
                    ("specific_stage1", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
                    ("specific_stage2", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
                    ("specific", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
                ),
            ),
            OverviewSetting(
                name="prefix",
                filepaths=(
                    ("prefix_stage1", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
                    ("prefix_stage2", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
                    ("prefix", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
                ),
            ),
        ),
        propensity_series=(
            PropensitySeries("Stage 1", "memorization_experiment/data/dynaword_stage1/propensity/st_dyna_propensity_metrics.json"),
            PropensitySeries("Stage 2", "memorization_experiment/data/dynaword_stage2/propensity/st_dyna_propensity_metrics.json"),
            PropensitySeries("Final", "memorization_experiment/data/dynaword/propensity/st_dyna_propensity_metrics.json"),
        ),
    ),
    ComparisonOverview(
        name="commonpile-dfm-stages-comparison",
        output_dir="memorization_experiment/data/commonpile_dfm_stages_comparison",
        settings=(
            OverviewSetting(
                name="generic",
                filepaths=(
                    ("generic_stage1", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
                    ("generic_stage2", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
                    ("generic", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
                ),
            ),
            OverviewSetting(
                name="specific",
                filepaths=(
                    ("specific_stage1", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
                    ("specific_stage2", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
                    ("specific", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
                ),
            ),
            OverviewSetting(
                name="prefix",
                filepaths=(
                    ("prefix_stage1", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
                    ("prefix_stage2", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
                    ("prefix", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
                ),
            ),
        ),
        propensity_series=(
            PropensitySeries("Stage 1", "memorization_experiment/data/commonpile_dfm_stage1/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Stage 2", "memorization_experiment/data/commonpile_dfm_stage2/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Final", "memorization_experiment/data/commonpile_dfm/propensity/st_cp_propensity_metrics.json"),
        ),
    ),
    ComparisonOverview(
        name="commonpile-dfm-dynaword-comparison",
        output_dir="memorization_experiment/data/commonpile_dfm_dynaword_comparison",
        settings=(
            OverviewSetting(
                name="generic",
                filepaths=(
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
                ),
            ),
            OverviewSetting(
                name="specific",
                filepaths=(
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
                ),
            ),
            OverviewSetting(
                name="prefix",
                filepaths=(
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
                ),
            ),
        ),
        propensity_series=(
            PropensitySeries("Common Pile", "memorization_experiment/data/commonpile_dfm/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Dynaword", "memorization_experiment/data/dynaword/propensity/st_dyna_propensity_metrics.json"),
        ),
    ),
    ComparisonOverview(
        name="dynaword-commonpile-stages-comparison",
        output_dir="memorization_experiment/data/dynaword_commonpile_stages_comparison",
        settings=(
            OverviewSetting(
                name="generic",
                filepaths=(
                    ("common_pile_stage1", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
                    ("dynaword_stage1", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
                    ("common_pile_stage2", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
                    ("dynaword_stage2", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
                ),
            ),
            OverviewSetting(
                name="specific",
                filepaths=(
                    ("common_pile_stage1", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
                    ("dynaword_stage1", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
                    ("common_pile_stage2", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
                    ("dynaword_stage2", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
                ),
            ),
            OverviewSetting(
                name="prefix",
                filepaths=(
                    ("common_pile_stage1", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
                    ("dynaword_stage1", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
                    ("common_pile_stage2", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
                    ("dynaword_stage2", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
                    ("common_pile", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
                    ("dynaword", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
                ),
            ),
        ),
        propensity_series=(
            PropensitySeries("Common Pile Stage 1", "memorization_experiment/data/commonpile_dfm_stage1/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Dynaword Stage 1", "memorization_experiment/data/dynaword_stage1/propensity/st_dyna_propensity_metrics.json"),
            PropensitySeries("Common Pile Stage 2", "memorization_experiment/data/commonpile_dfm_stage2/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Dynaword Stage 2", "memorization_experiment/data/dynaword_stage2/propensity/st_dyna_propensity_metrics.json"),
            PropensitySeries("Common Pile", "memorization_experiment/data/commonpile_dfm/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("Dynaword", "memorization_experiment/data/dynaword/propensity/st_dyna_propensity_metrics.json"),
        ),
    ),
    ComparisonOverview(
        name="commonpile-comma-dfm",
        output_dir="memorization_experiment/data/commonpile_comma_dfm",
        settings=(
            OverviewSetting(
                name="generic",
                filepaths=(
                    ("comma", "memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json"),
                    ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
                ),
            ),
            OverviewSetting(
                name="specific",
                filepaths=(
                    ("comma", "memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json"),
                    ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
                ),
            ),
            OverviewSetting(
                name="prefix",
                filepaths=(
                    ("comma", "memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json"),
                    ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
                ),
            ),
        ),
        propensity_series=(
            PropensitySeries("Comma", "memorization_experiment/data/commonpile/propensity/st_cp_propensity_metrics.json"),
            PropensitySeries("DFM Decoder", "memorization_experiment/data/commonpile_dfm/propensity/st_cp_propensity_metrics.json"),
        ),
    ),
)

OVERVIEWS_BY_NAME = {overview.name: overview for overview in OVERVIEWS}


def _require_plot_dependencies():
    global plt, np
    if plt is not None and np is not None:
        return

    tmp_cache_dir = os.path.join(tempfile.gettempdir(), "plot_comparison_overviews_cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", tmp_cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", tmp_cache_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as imported_plt
        import numpy as imported_np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib and numpy. Use the project virtualenv, "
            "for example `.venv/bin/python memorization_experiment/plot_comparison_overviews.py ...`."
        ) from exc

    plt = imported_plt
    np = imported_np


def _ensure_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"Warning: Could not parse JSON file {path}: {exc}")
        return None


def _format_value(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _pretty_setting_name(name: str) -> str:
    return name.replace("_", " ").title()


def _pretty_metric_name(name: str) -> str:
    mapping = {
        "average_longest_span_length": "Average Longest Span",
        "avg_nv_recall": "Avg NV Recall",
        "generations_full_matches_ratio": "Full Matches Ratio",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _display_label(label: str, setting_name: str) -> str:
    exact_mapping = {
        "comma": "Comma",
        "dfm_decoder": "DFM Decoder",
        "common_pile_stage1": "Common Pile Stage 1",
        "common_pile_stage2": "Common Pile Stage 2",
        "common_pile": "Common Pile",
        "dynaword_stage1": "Dynaword Stage 1",
        "dynaword_stage2": "Dynaword Stage 2",
        "dynaword": "Dynaword",
    }
    if label in exact_mapping:
        return exact_mapping[label]
    if label.endswith("_stage1"):
        return "Stage 1"
    if label.endswith("_stage2"):
        return "Stage 2"
    if label == setting_name:
        return "Final"
    mapping = {
        "commonpile_dfm": "CommonPile DFM",
        "common_pile": "Common Pile",
        "dynaword": "Dynaword",
        "generic": "Generic",
        "specific": "Specific",
        "prefix": "Prefix",
    }
    if label in mapping:
        return mapping[label]
    return label.replace("_", " ").title()


def _color_map_for_labels(labels: list[str]) -> dict[str, str]:
    palette = [COLOR_A, COLOR_D, COLOR_H, *FALLBACK_COLORS]
    return {label: palette[idx % len(palette)] for idx, label in enumerate(labels)}


def _metric_value(summary: dict, metric_name: str) -> float:
    aliases = {
        "average_longest_span_length": [
            "average_longest_span_length",
            "average_longest_span",
            "average_span_length",
        ],
        "avg_nv_recall": ["avg_nv_recall"],
        "generations_full_matches_ratio": ["generations_full_matches_ratio"],
    }
    for key in aliases.get(metric_name, [metric_name]):
        if key in summary:
            return float(summary[key])
    return 0.0


def _bucket_sort_key(bucket_label: str) -> tuple[int, float, str]:
    match = re.match(r"^\(\s*(\d+)\s*,\s*([0-9]+|inf)\s*\)$", str(bucket_label))
    if not match:
        return (10**9, float("inf"), str(bucket_label))
    lo = int(match.group(1))
    hi_raw = match.group(2)
    hi = float("inf") if hi_raw == "inf" else int(hi_raw)
    return (lo, hi, str(bucket_label))


def _extract_propensity_metrics(summary: dict) -> list[str]:
    comparisons = summary.get("comparisons", {})
    for payload in comparisons.values():
        results = payload.get("results", {})
        if results:
            return list(results.keys())
    return []


def _extract_propensity_settings(summary: dict) -> list[str]:
    non_prefix_settings = summary.get("non_prefix_settings", [])
    if non_prefix_settings:
        return [str(setting) for setting in non_prefix_settings]
    return list(summary.get("comparisons", {}).keys())


def _extract_propensity_value(summary: dict, setting: str, metric: str) -> float:
    metric_payload = summary.get("comparisons", {}).get(setting, {}).get("results", {}).get(metric, {})
    return float(metric_payload.get(f"propensity_{metric}", 0.0))


def _load_overview_payload(
    overview: ComparisonOverview,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    summaries_by_setting: dict[str, dict[str, dict]] = {}
    propensity_reports: dict[str, dict] = {}

    for setting in overview.settings:
        setting_summaries: dict[str, dict] = {}
        for label, summary_path_str in setting.filepaths:
            summary = _load_json(_ensure_path(summary_path_str))
            if summary is not None:
                setting_summaries[label] = summary
        summaries_by_setting[setting.name] = setting_summaries

    for series in overview.propensity_series:
        propensity_summary = _load_json(_ensure_path(series.summary_path))
        if propensity_summary is not None:
            propensity_reports[series.label] = propensity_summary

    return summaries_by_setting, propensity_reports


def _plot_scalar_overview(
    overview: ComparisonOverview,
    summaries_by_setting: dict[str, dict[str, dict]],
    metric_name: str,
    output_dir: Path,
) -> None:
    _require_plot_dependencies()

    ordered_settings = [setting for setting in overview.settings if summaries_by_setting.get(setting.name)]
    if not ordered_settings:
        return

    display_labels: list[str] = []
    for setting in ordered_settings:
        for label in summaries_by_setting[setting.name]:
            display_label = _display_label(label, setting.name)
            if display_label not in display_labels:
                display_labels.append(display_label)

    color_map = _color_map_for_labels(display_labels)
    x = np.arange(len(ordered_settings), dtype=float)
    width = 0.8 / max(len(display_labels), 1)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    y_max = 0.0

    for idx, display_label in enumerate(display_labels):
        values: list[float] = []
        positions: list[float] = []
        for setting_idx, setting in enumerate(ordered_settings):
            summaries = summaries_by_setting[setting.name]
            matched_label = None
            for original_label in summaries:
                if _display_label(original_label, setting.name) == display_label:
                    matched_label = original_label
                    break
            if matched_label is None:
                continue

            value = _metric_value(summaries[matched_label], metric_name)
            positions.append(x[setting_idx] + (idx - (len(display_labels) - 1) / 2) * width)
            values.append(value)
            y_max = max(y_max, value)

        if not positions:
            continue

        bars = ax.bar(
            positions,
            values,
            width=width,
            color=color_map[display_label],
            label=display_label,
            edgecolor="#333333",
            linewidth=0.9,
        )
        for bar in bars:
            height = float(bar.get_height())
            ax.annotate(
                _format_value(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_setting_name(setting.name) for setting in ordered_settings])
    ax.set_ylabel("Value")
    ax.set_title(f"{metric_name} across generic, specific, and prefix")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Series")
    if metric_name == "avg_nv_recall":
        ax.set_ylim(0.0, y_max + 0.05)
    else:
        ax.set_ylim(0.0, y_max + max(0.05, y_max * 0.12))

    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_name}_overview.png", dpi=220)
    plt.close(fig)


def _plot_distribution_overview(
    overview: ComparisonOverview,
    summaries_by_setting: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    _require_plot_dependencies()

    ordered_settings = [setting for setting in overview.settings if summaries_by_setting.get(setting.name)]
    if not ordered_settings:
        return

    display_labels: list[str] = []
    all_buckets: set[str] = set()
    for setting in ordered_settings:
        for label, summary in summaries_by_setting[setting.name].items():
            display_label = _display_label(label, setting.name)
            if display_label not in display_labels:
                display_labels.append(display_label)
            for bucket in summary.get("spans_length_distribution", {}).keys():
                all_buckets.add(str(bucket))

    if not all_buckets:
        return

    color_map = _color_map_for_labels(display_labels)
    buckets = sorted(all_buckets, key=_bucket_sort_key)
    x = np.arange(len(buckets), dtype=float)

    fig, axes = plt.subplots(1, len(ordered_settings), figsize=(6.2 * len(ordered_settings), 5.6), sharey=True)
    axes_list = list(axes if hasattr(axes, "__len__") else [axes])
    legend_handles = None
    legend_labels = None

    for ax, setting in zip(axes_list, ordered_settings):
        summaries = summaries_by_setting[setting.name]
        width = 0.8 / max(len(summaries), 1)
        for idx, (label, summary) in enumerate(summaries.items()):
            display_label = _display_label(label, setting.name)
            dist = summary.get("spans_length_distribution", {})
            values = [dist.get(bucket, 0.0) for bucket in buckets]
            offset = (idx - (len(summaries) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=color_map[display_label],
                label=display_label,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
            )

        label_step = max(1, len(buckets) // 8)
        tick_labels = [
            bucket if idx % label_step == 0 or idx == len(buckets) - 1 else ""
            for idx, bucket in enumerate(buckets)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_title(_pretty_setting_name(setting.name))
        ax.set_xlabel("Span Length Bucket")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles and legend_handles is None:
            legend_handles, legend_labels = handles, labels

    axes_list[0].set_ylabel("Ratio")
    fig.suptitle("spans_length_distribution overview", fontsize=15)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.98),
            ncol=min(len(legend_labels), 4),
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_dir / SPAN_OUTPUT_NAME, dpi=220)
    plt.close(fig)


def _plot_propensity_overview(
    overview: ComparisonOverview,
    propensity_reports: dict[str, dict],
    output_dir: Path,
) -> None:
    _require_plot_dependencies()

    ordered_series = [series for series in overview.propensity_series if propensity_reports.get(series.label)]
    if not ordered_series:
        return

    comparison_settings = ["generic", "specific"]
    metrics: list[str] = []
    for series in ordered_series:
        summary = propensity_reports[series.label]
        for metric in _extract_propensity_metrics(summary):
            if metric not in metrics:
                metrics.append(metric)

    if not metrics:
        return

    color_map = _color_map_for_labels([series.label for series in ordered_series])
    x_labels: list[str] = []
    x_positions: list[float] = []
    position_lookup: dict[tuple[str, str], float] = {}
    current_x = 0.0
    group_gap = 0.8
    metric_gap = 1.0

    for setting_name in comparison_settings:
        for metric in metrics:
            x_labels.append(f"{_pretty_setting_name(setting_name)}\n{_pretty_metric_name(metric)}")
            x_positions.append(current_x)
            position_lookup[(setting_name, metric)] = current_x
            current_x += metric_gap
        current_x += group_gap

    fig, ax = plt.subplots(figsize=(max(10.5, 2.2 * len(x_positions)), 6.0))
    y_max = 0.0

    width = 0.8 / max(len(ordered_series), 1)
    for idx, series in enumerate(ordered_series):
        positions: list[float] = []
        values: list[float] = []
        propensity_summary = propensity_reports[series.label]
        available_settings = set(_extract_propensity_settings(propensity_summary))
        for setting_name in comparison_settings:
            if setting_name not in available_settings:
                continue
            for metric in metrics:
                position = position_lookup[(setting_name, metric)]
                value = _extract_propensity_value(propensity_summary, setting_name, metric)
                positions.append(position + (idx - (len(ordered_series) - 1) / 2) * width)
                values.append(value)
                y_max = max(y_max, value)

        if not positions:
            continue

        bars = ax.bar(
            positions,
            values,
            width=width,
            color=color_map[series.label],
            label=series.label,
            edgecolor="#333333",
            linewidth=0.9,
        )
        for bar in bars:
            height = float(bar.get_height())
            ax.annotate(
                _format_value(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=16, ha="right")
    ax.set_ylabel("Propensity")
    ax.set_title("Propensity metrics overview")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, y_max + 0.05)
    ax.legend(loc="upper left", ncol=min(len(ordered_series), 4), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / PROPENSITY_OUTPUT_NAME, dpi=220)
    plt.close(fig)


def _generate_overview(overview: ComparisonOverview) -> None:
    summaries_by_setting, propensity_reports = _load_overview_payload(overview)
    output_dir = _ensure_path(overview.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in SCALAR_METRICS:
        _plot_scalar_overview(overview, summaries_by_setting, metric_name, output_dir)

    _plot_distribution_overview(overview, summaries_by_setting, output_dir)
    _plot_propensity_overview(overview, propensity_reports, output_dir)
    print(f"Generated overview plots for {overview.name} in {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined overview plots for comparison-style memorization experiments."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help="Overview names to run. If omitted, all overview families are generated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available overview targets and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list:
        for overview in OVERVIEWS:
            print(overview.name)
        return

    if args.targets:
        selected: list[ComparisonOverview] = []
        for name in args.targets:
            if name not in OVERVIEWS_BY_NAME:
                valid = ", ".join(sorted(OVERVIEWS_BY_NAME))
                raise SystemExit(f"Unknown target '{name}'. Valid targets: {valid}")
            selected.append(OVERVIEWS_BY_NAME[name])
    else:
        selected = list(OVERVIEWS)

    for overview in selected:
        _generate_overview(overview)


if __name__ == "__main__":
    main()

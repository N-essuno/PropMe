#!/usr/bin/env python3

"""Compute propensity-style metrics from prefix vs non-prefix SimpleTrace summaries.

This script supports two modes:

1) Direct mode:
   Provide summary paths explicitly via 
   --generic-summary / --specific-summary / --prefix-summary
   optionally write a JSON report, and optionally render a plot.

2) Preset mode:
   Select one or more named memorization experiment families (or groups of families) and compute propensity reports using built-in summary/output paths, similar to the preset runners used for running experiments and plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memorization_experiment" / "generation"))
from generation_runs import (  # noqa: E402
    COMPARISONS,
    NON_PREFIX_SETTINGS,
    SETTING_LABELS,
    SETTINGS,
    Comparison,
    Corpus,
    Model,
    comparison_dir,
    propensity_path,
    run_name,
    runs,
    setting_tag,
    summary_path,
)


DEFAULT_PRESET_METRICS = (
    "avg_nv_recall",
    "generations_full_matches_ratio"
)


@dataclass(frozen=True)
class PropensityPreset:
    name: str
    setting_to_summary_paths: tuple[tuple[str, str], ...]
    prefix_summary: str
    output: str
    plot_title: str
    tags: tuple[str, ...]


DFM9_PRESETS = (
    PropensityPreset(
        name="dfm9-generic-en-vs-prefix-a",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_A_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_en_vs_prefix_A_propensity.json",
        plot_title="DFM9 Generic EN vs Prefix A Propensity Metrics",
        tags=("dfm9", "generations", "generic-en", "prefix-a"),
    ),
    PropensityPreset(
        name="dfm9-generic-en-vs-prefix-b",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_B_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_en_vs_prefix_B_propensity.json",
        plot_title="DFM9 Generic EN vs Prefix B Propensity Metrics",
        tags=("dfm9", "generations", "generic-en", "prefix-b"),
    ),
    PropensityPreset(
        name="dfm9-generic-da-vs-prefix-a",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_A_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_da_vs_prefix_A_propensity.json",
        plot_title="DFM9 Generic DA vs Prefix A Propensity Metrics",
        tags=("dfm9", "generations", "generic-da", "prefix-a"),
    ),
    PropensityPreset(
        name="dfm9-generic-da-vs-prefix-b",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_B_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_da_vs_prefix_B_propensity.json",
        plot_title="DFM9 Generic DA vs Prefix B Propensity Metrics",
        tags=("dfm9", "generations", "generic-da", "prefix-b"),
    ),
    PropensityPreset(
        name="dfm9-generic-en-vs-prefix-c",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_C_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_en_vs_prefix_C_propensity.json",
        plot_title="DFM9 Generic EN vs Prefix C Propensity Metrics",
        tags=("dfm9", "generations", "generic-en", "prefix-c"),
    ),
    PropensityPreset(
        name="dfm9-generic-en-vs-prefix-d",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_D_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_en_vs_prefix_D_propensity.json",
        plot_title="DFM9 Generic EN vs Prefix D Propensity Metrics",
        tags=("dfm9", "generations", "generic-en", "prefix-d"),
    ),
    PropensityPreset(
        name="dfm9-generic-da-vs-prefix-c",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_C_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_da_vs_prefix_C_propensity.json",
        plot_title="DFM9 Generic DA vs Prefix C Propensity Metrics",
        tags=("dfm9", "generations", "generic-da", "prefix-c"),
    ),
    PropensityPreset(
        name="dfm9-generic-da-vs-prefix-d",
        setting_to_summary_paths=(
            (
                "generic",
                "memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_D_summary.json",
            ),
        ),
        prefix_summary="memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_summary.json",
        output="memorization_experiment/data/dfm9/propensity/dfm9_generic_da_vs_prefix_D_propensity.json",
        plot_title="DFM9 Generic DA vs Prefix D Propensity Metrics",
        tags=("dfm9", "generations", "generic-da", "prefix-d"),
    ),

)


def _run_preset(model: Model, corpus: Corpus) -> PropensityPreset:
    """Every non-prefix setting of a generation run (see generation_runs.py) against its prefix setting."""
    return PropensityPreset(
        name=run_name(model, corpus),
        setting_to_summary_paths=tuple(
            (setting, summary_path(model, corpus, setting)) for setting in NON_PREFIX_SETTINGS
        ),
        prefix_summary=summary_path(model, corpus, "prefix"),
        output=propensity_path(model, corpus),
        plot_title=f"{model.label} on {corpus.label} Propensity Metrics",
        tags=(model.family, model.key, corpus.name, "generations"),
    )


def _comparison_preset(comparison: Comparison, setting: str) -> PropensityPreset:
    """One setting of each compared run against the same setting of the comparison's reference run."""
    reference = comparison.reference
    return PropensityPreset(
        name=f"{comparison.name}-{setting_tag(setting)}",
        setting_to_summary_paths=tuple(
            (series.label, summary_path(series.model, series.corpus, setting))
            for series in comparison.series
        ),
        prefix_summary=summary_path(reference.model, reference.corpus, setting),
        output=f"{comparison_dir(comparison)}/{setting}/propensity_metrics.json",
        plot_title=(
            f"{comparison.title}: {SETTING_LABELS[setting]} vs {reference.label} Propensity Metrics"
        ),
        tags=(comparison.name, "comparison", setting_tag(setting)),
    )


PRESETS = (
    DFM9_PRESETS
    + tuple(_run_preset(model, corpus) for model, corpus in runs())
    + tuple(
        _comparison_preset(comparison, setting)
        for comparison in COMPARISONS
        if comparison.reference
        for setting in SETTINGS
    )
)

PRESETS_BY_NAME = {preset.name: preset for preset in PRESETS}

# One group per tag: a model family (dfm), model (dfm-main), corpus (dynaword),
# comparison (dfm-stages-dynaword), setting of the comparisons (generic), or a DFM9 tag.
GROUPS = {
    "all": [preset.name for preset in PRESETS],
    "all-generations": [preset.name for preset in PRESETS if "generations" in preset.tags],
    "all-comparisons": [preset.name for preset in PRESETS if "comparison" in preset.tags],
    "dfm9-generations": [
        preset.name
        for preset in PRESETS
        if "dfm9" in preset.tags and "generations" in preset.tags
    ],
    **{
        tag: [preset.name for preset in PRESETS if tag in preset.tags]
        for tag in dict.fromkeys(tag for preset in PRESETS for tag in preset.tags)
    },
}


DFM9_COMBINED_PROPENSITY_SERIES = (
    ("dfm9-generic-en-vs-prefix-a", "Generic EN vs Prefix A"),
    ("dfm9-generic-en-vs-prefix-b", "Generic EN vs Prefix B"),
    ("dfm9-generic-en-vs-prefix-c", "Generic EN vs Prefix C"),
    ("dfm9-generic-en-vs-prefix-d", "Generic EN vs Prefix D"),
    ("dfm9-generic-da-vs-prefix-a", "Generic DA vs Prefix A"),
    ("dfm9-generic-da-vs-prefix-b", "Generic DA vs Prefix B"),
    ("dfm9-generic-da-vs-prefix-c", "Generic DA vs Prefix C"),
    ("dfm9-generic-da-vs-prefix-d", "Generic DA vs Prefix D"),
)
DFM9_COMBINED_PROPENSITY_PLOT = (
    "memorization_experiment/data/dfm9/propensity/"
    "dfm9_propensity_comparisons.png"
)


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _parse_metrics(raw_metrics: list[str]) -> list[str]:
    metrics: list[str] = []
    for item in raw_metrics:
        for token in item.split(","):
            metric = token.strip()
            if metric:
                metrics.append(metric)
    if not metrics:
        raise ValueError("At least one metric must be provided via --metrics.")
    return list(dict.fromkeys(metrics))


def _ensure_scalar_number(summary: dict, metric: str, label: str) -> float:
    if metric not in summary:
        raise KeyError(f"Metric '{metric}' not found in {label} summary.")
    value = summary[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric}' in {label} summary must be a numeric scalar, got {type(value).__name__}."
        )
    if not math.isfinite(float(value)):
        raise ValueError(f"Metric '{metric}' in {label} summary must be finite.")
    return float(value)


def compute_propensity_value(non_prefix_value: float, prefix_value: float) -> float:
    """Return a bounded propensity score in [0, 1]."""
    scale = abs(non_prefix_value) + abs(prefix_value)
    if scale == 0.0:
        return 0.0
    return 0.5 * (1.0 + ((non_prefix_value - prefix_value) / scale))


def build_propensity_report(
    non_prefix_summary: dict,
    prefix_summary: dict,
    metrics: list[str],
    *,
    non_prefix_setting: str,
    non_prefix_summary_path: str,
    prefix_summary_path: str,
) -> dict:
    results: dict[str, dict] = {}
    delta_key = f"delta_{non_prefix_setting}_minus_prefix"
    output: dict = {
        "non_prefix_setting": non_prefix_setting,
        "non_prefix_summary_path": non_prefix_summary_path,
        "prefix_summary_path": prefix_summary_path,
        "metrics": metrics,
        "propensity_definition": (
            "propensity_m = 0.5 * (1 + (non_prefix_m - prefix_m) / (|non_prefix_m| + |prefix_m|)); "
            "when both values are 0, propensity_m = 0.0; propensity_m = 0.5 only when the values are equal "
            "and non_prefix_m > 0"
        ),
        "results": results,
    }

    for metric in metrics:
        non_prefix_value = _ensure_scalar_number(non_prefix_summary, metric, non_prefix_setting)
        prefix_value = _ensure_scalar_number(prefix_summary, metric, "prefix")
        propensity_key = f"propensity_{metric}"
        propensity_value = compute_propensity_value(non_prefix_value, prefix_value)
        metric_payload = {
            f"{non_prefix_setting}_value": non_prefix_value,
            "prefix_value": prefix_value,
            delta_key: non_prefix_value - prefix_value,
            propensity_key: propensity_value,
        }
        results[metric] = metric_payload
        output[propensity_key] = propensity_value

    return output


def build_multi_setting_propensity_report(
    prefix_summary: dict,
    metrics: list[str],
    *,
    prefix_summary_path: str,
    setting_to_summary_path: dict[str, str],
) -> dict:
    comparisons: dict[str, dict] = {}
    output: dict = {
        "prefix_summary_path": prefix_summary_path,
        "metrics": metrics,
        "non_prefix_settings": list(setting_to_summary_path.keys()),
        "propensity_definition": (
            "propensity_m = 0.5 * (1 + (non_prefix_m - prefix_m) / (|non_prefix_m| + |prefix_m|)); "
            "when both values are 0, propensity_m = 0.0; propensity_m = 0.5 only when the values are equal "
            "and non_prefix_m > 0"
        ),
        "comparisons": comparisons,
    }

    for setting_name, summary_path in setting_to_summary_path.items():
        setting_summary = _load_json(summary_path)
        report = build_propensity_report(
            setting_summary,
            prefix_summary,
            metrics,
            non_prefix_setting=setting_name,
            non_prefix_summary_path=summary_path,
            prefix_summary_path=prefix_summary_path,
        )
        comparisons[setting_name] = report

        for metric in metrics:
            output[f"propensity_{setting_name}_{metric}"] = report[f"propensity_{metric}"]

    return output


def _resolve_setting_to_summary_path(args) -> dict[str, str]:
    setting_to_summary_path: dict[str, str] = {}

    if args.generic_summary.strip():
        setting_to_summary_path["generic"] = args.generic_summary
    if args.specific_summary.strip():
        setting_to_summary_path["specific"] = args.specific_summary

    for raw_item in args.setting_summary:
        if "=" not in raw_item:
            raise ValueError(
                f"Invalid --setting-summary value '{raw_item}'. Expected LABEL=PATH."
            )
        label, summary_path = raw_item.split("=", 1)
        label = label.strip()
        summary_path = summary_path.strip()
        if not label or not summary_path:
            raise ValueError(
                f"Invalid --setting-summary value '{raw_item}'. Expected LABEL=PATH."
            )
        if label in setting_to_summary_path:
            raise ValueError(
                f"Setting '{label}' was provided twice. Use each setting label only once."
            )
        setting_to_summary_path[label] = summary_path

    legacy_setting = (args.non_prefix_setting or "").strip()
    legacy_summary = (args.non_prefix_summary or "").strip()
    if legacy_setting or legacy_summary:
        if not legacy_setting or not legacy_summary:
            raise ValueError(
                "--non-prefix-setting and --non-prefix-summary must be provided together."
            )
        if legacy_setting in setting_to_summary_path:
            raise ValueError(
                f"Setting '{legacy_setting}' was provided twice. Use either "
                f"--{legacy_setting}-summary or the legacy pair, not both."
            )
        setting_to_summary_path[legacy_setting] = legacy_summary

    if not setting_to_summary_path:
        raise ValueError(
            "Provide at least one non-prefix summary via --generic-summary, "
            "--specific-summary, --setting-summary, or the legacy "
            "--non-prefix-setting/--non-prefix-summary pair."
        )

    return setting_to_summary_path


def _extract_metric_names(summary: dict) -> list[str]:
    metrics = summary.get("metrics", [])
    if metrics:
        return [str(metric) for metric in metrics]

    comparisons = summary.get("comparisons", {})
    for payload in comparisons.values():
        results = payload.get("results", {})
        if results:
            return list(results.keys())

    raise ValueError("No metrics found in propensity summary.")


def _extract_setting_names(summary: dict) -> list[str]:
    settings = summary.get("non_prefix_settings", [])
    if settings:
        return [str(setting) for setting in settings]

    comparisons = summary.get("comparisons", {})
    if comparisons:
        return list(comparisons.keys())

    raise ValueError("No non-prefix settings found in propensity summary.")


def _extract_propensity_value(summary: dict, setting: str, metric: str) -> float:
    comparisons = summary.get("comparisons", {})
    if setting not in comparisons:
        raise KeyError(f"Setting '{setting}' not found in propensity summary.")

    metric_payload = comparisons[setting].get("results", {}).get(metric)
    if metric_payload is None:
        raise KeyError(f"Metric '{metric}' not found for setting '{setting}'.")

    propensity_key = f"propensity_{metric}"
    if propensity_key not in metric_payload:
        raise KeyError(
            f"Propensity field '{propensity_key}' not found for metric '{metric}' and setting '{setting}'."
        )

    return float(metric_payload[propensity_key])


def _derive_plot_output_path(report_output_path: str, prefix_summary_path: str) -> str:
    if report_output_path.strip():
        path = Path(report_output_path)
        if path.suffix:
            return str(path.with_name(f"{path.stem}_plot.png"))
        return f"{report_output_path}_plot.png"

    prefix_path = Path(prefix_summary_path)
    if prefix_path.suffix:
        return str(prefix_path.with_name(f"{prefix_path.stem}_propensity_plot.png"))
    return f"{prefix_summary_path}_propensity_plot.png"


def _get_plotting_modules():
    tmp_cache_dir = os.path.join(tempfile.gettempdir(), "compute_propensity_metrics_cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", tmp_cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", tmp_cache_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib and numpy (see requirements.txt)."
        ) from exc

    return plt, np


def plot_propensity_summary(summary: dict, *, title: str | None = None):
    plt, np = _get_plotting_modules()

    setting_colors = ["#1F449C", "#009E73", "#F05039", "#7A4F9A", "#BCBD22"]

    metrics = _extract_metric_names(summary)
    settings = _extract_setting_names(summary)

    x = np.arange(len(metrics), dtype=float)
    width = 0.8 / max(len(settings), 1)
    max_value = 0.0

    fig_width = max(10, 1.6 * len(metrics))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for idx, setting in enumerate(settings):
        values = [_extract_propensity_value(summary, setting, metric) for metric in metrics]
        if values:
            max_value = max(max_value, max(values))
        offset = (idx - (len(settings) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=setting,
            color=setting_colors[idx % len(setting_colors)],
            edgecolor="#333333",
            linewidth=1.0,
        )
        for bar in bars:
            h = float(bar.get_height())
            ax.annotate(
                _format_value(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(metrics) > 8 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylim(0.0, max_value + 0.05)
    ax.set_ylabel("Propensity")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Setting")
    ax.set_title(title or "Propensity Metrics by Setting")

    fig.tight_layout()
    return fig


def plot_combined_propensity_reports(
    reports_by_name: dict[str, dict],
    series: tuple[tuple[str, str], ...],
    metrics: list[str],
    *,
    title: str,
):
    """Plot one colored series per comparison and one bar per metric."""
    plt, np = _get_plotting_modules()

    colors = ("#1F449C", "#E69F00", "#009E73", "#CC79A7")
    x = np.arange(len(metrics), dtype=float)
    width = 0.82 / max(len(series), 1)
    max_value = 0.0

    fig, ax = plt.subplots(figsize=(max(10.5, 2.1 * len(metrics)), 6.2))
    for idx, (preset_name, label) in enumerate(series):
        report = reports_by_name[preset_name]
        setting_names = _extract_setting_names(report)
        if len(setting_names) != 1:
            raise ValueError(
                f"Combined propensity series '{label}' must contain exactly one "
                f"setting, found {setting_names}."
            )

        setting_name = setting_names[0]
        values = [
            _extract_propensity_value(report, setting_name, metric)
            for metric in metrics
        ]
        if values:
            max_value = max(max_value, max(values))
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors[idx % len(colors)],
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

    metric_labels = [metric.replace("_", " ").title() for metric in metrics]
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.set_ylim(0.0, min(1.05, max_value + 0.08))
    ax.set_ylabel("Propensity")
    ax.set_xlabel("Metric")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Comparison", loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def _ensure_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _expand_targets(raw_targets: list[str]) -> list[PropensityPreset]:
    names: list[str] = []
    seen: set[str] = set()

    for raw_target in raw_targets:
        for target in (token.strip() for token in raw_target.split(",") if token.strip()):
            if target in GROUPS:
                expanded_names = GROUPS[target]
            elif target in PRESETS_BY_NAME:
                expanded_names = [target]
            else:
                valid = ", ".join(sorted({*GROUPS.keys(), *PRESETS_BY_NAME.keys()}))
                raise SystemExit(
                    f"Unknown target '{target}'. Use --list to inspect choices.\n\nValid targets:\n{valid}"
                )

            for name in expanded_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

    return [PRESETS_BY_NAME[name] for name in names]


def _print_available_targets() -> None:
    print("Groups:")
    for group_name in sorted(GROUPS):
        print(f"  {group_name}")

    print("\nPropensity presets:")
    for preset in PRESETS:
        tags = ", ".join(preset.tags)
        print(f"  {preset.name} [{tags}]")

    print("\nDefault preset metrics:")
    print(f"  {', '.join(DEFAULT_PRESET_METRICS)}")


def _build_report_from_paths(
    *,
    setting_to_summary_paths: tuple[tuple[str, str], ...],
    prefix_summary_path: str,
    metrics: list[str],
) -> dict:
    prefix_summary = _load_json(prefix_summary_path)
    setting_to_summary_path = dict(setting_to_summary_paths)
    return build_multi_setting_propensity_report(
        prefix_summary,
        metrics,
        prefix_summary_path=prefix_summary_path,
        setting_to_summary_path=setting_to_summary_path,
    )


def _preset_confidence_intervals(preset: PropensityPreset, metrics: list[str], args: argparse.Namespace) -> dict:
    """95% bootstrap CIs (bootstrap_ci.py) of a preset's settings, propensities or comparisons."""
    import bootstrap_ci

    rounds, seed = args.bootstrap_samples, args.ci_seed
    out = {"method": "percentile bootstrap, 95%", "bootstrap_samples": rounds, "seed": seed}
    if "comparison" in preset.tags:
        # Each series against the reference run on the same setting.
        out["reference"] = bootstrap_ci.setting_ci(preset.prefix_summary, rounds, seed)
        out["series"] = {
            label: {
                **bootstrap_ci.setting_ci(path, rounds, seed),
                "vs_reference": bootstrap_ci.comparison_ci(path, preset.prefix_summary, metrics, rounds, seed),
            }
            for label, path in preset.setting_to_summary_paths
        }
    else:
        out["prefix"] = bootstrap_ci.setting_ci(preset.prefix_summary, rounds, seed)
        out["settings"] = {
            label: {
                **bootstrap_ci.setting_ci(path, rounds, seed),
                "propensity": bootstrap_ci.propensity_ci(path, preset.prefix_summary, metrics, rounds, seed),
            }
            for label, path in preset.setting_to_summary_paths
        }
    return out


def _write_report(report: dict, output_path: str) -> Path:
    rendered = json.dumps(report, indent=4)
    output_file = _ensure_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered + "\n")
    return output_file


def _write_plot(report: dict, *, plot_output_path: str, plot_title: str | None) -> Path:
    fig = plot_propensity_summary(report, title=plot_title)
    plt, _ = _get_plotting_modules()
    plot_output_file = _ensure_path(plot_output_path)
    plot_output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_output_file


def _has_direct_summary_inputs(args: argparse.Namespace) -> bool:
    return any(
        [
            args.generic_summary.strip(),
            args.specific_summary.strip(),
            bool(args.setting_summary),
            args.non_prefix_setting.strip(),
            args.non_prefix_summary.strip(),
            args.prefix_summary.strip(),
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute propensity-style metrics either from explicit summary paths "
            "or from named memorization experiment presets."
        )
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help="Preset names and/or group names to run. Use --list to inspect choices.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available preset names and group names, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected presets and derived output paths without computing reports.",
    )
    parser.add_argument(
        "--generic-summary",
        default="",
        help="Direct mode: path to the SimpleTrace summary JSON for the generic prompt setting.",
    )
    parser.add_argument(
        "--specific-summary",
        default="",
        help="Direct mode: path to the SimpleTrace summary JSON for the specific prompt setting.",
    )
    parser.add_argument(
        "--setting-summary",
        action="append",
        default=[],
        help=(
            "Direct mode: custom non-prefix setting in the form LABEL=PATH. "
            "Repeat to compare multiple custom settings against --prefix-summary."
        ),
    )
    parser.add_argument(
        "--non-prefix-setting",
        default="",
        help="Direct mode legacy interface: label of the non-prefix setting to compare against prefix.",
    )
    parser.add_argument(
        "--non-prefix-summary",
        default="",
        help="Direct mode legacy interface: path to the SimpleTrace summary JSON for the non-prefix prompt setting.",
    )
    parser.add_argument(
        "--prefix-summary",
        default="",
        help="Direct mode: path to the SimpleTrace summary JSON for the prefix prompt setting.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[],
        help=(
            "Metric names to use. In direct mode this is required. "
            f"In preset mode it defaults to: {', '.join(DEFAULT_PRESET_METRICS)}"
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Direct mode: optional output JSON path. If omitted, the report is printed to stdout only.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, also render a grouped bar plot of the propensity metrics.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help=(
            "Preset mode: add 95%% bootstrap confidence intervals of every setting's metrics, of the "
            "propensities and of the comparisons to each report (see bootstrap_ci.py)."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap rounds for --ci.",
    )
    parser.add_argument(
        "--ci-seed",
        type=int,
        default=42,
        help="Seed of the bootstrap resampling for --ci.",
    )
    parser.add_argument(
        "--plot-output",
        default="",
        help="Direct mode: optional output PNG path for the propensity plot.",
    )
    parser.add_argument(
        "--plot-title",
        default="",
        help="Direct mode: optional chart title override for the propensity plot.",
    )
    return parser


def _run_direct_mode(args: argparse.Namespace) -> int:
    if not args.prefix_summary.strip():
        raise SystemExit("--prefix-summary is required in direct mode.")
    if not args.metrics:
        raise SystemExit("--metrics is required in direct mode.")

    metrics = _parse_metrics(args.metrics)
    prefix_summary = _load_json(args.prefix_summary)
    setting_to_summary_path = _resolve_setting_to_summary_path(args)

    report = build_multi_setting_propensity_report(
        prefix_summary,
        metrics,
        prefix_summary_path=args.prefix_summary,
        setting_to_summary_path=setting_to_summary_path,
    )

    rendered = json.dumps(report, indent=4)
    print(rendered)

    if args.output.strip():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")

    if args.plot:
        fig = plot_propensity_summary(report, title=args.plot_title.strip() or None)
        plt, _ = _get_plotting_modules()
        plot_output_path = args.plot_output.strip() or _derive_plot_output_path(
            args.output,
            args.prefix_summary,
        )
        plot_output_file = Path(plot_output_path)
        plot_output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(plot_output_file)

    return 0


def _run_preset_mode(args: argparse.Namespace) -> int:
    if args.output.strip() or args.plot_output.strip() or args.plot_title.strip():
        raise SystemExit(
            "Preset mode does not support --output, --plot-output, or --plot-title. "
            "Use direct mode with explicit summary paths for ad hoc outputs."
        )

    if not args.targets:
        raise SystemExit("No targets provided. Use --list to inspect the available presets and groups.")

    presets = _expand_targets(args.targets)
    if not presets:
        print("No propensity presets selected.")
        return 0

    metrics = _parse_metrics(args.metrics) if args.metrics else list(DEFAULT_PRESET_METRICS)

    print("Selected propensity presets:")
    for preset in presets:
        print(f"  - {preset.name}")
    print()

    failures: list[str] = []
    reports_by_name: dict[str, dict] = {}
    selected_names = {preset.name for preset in presets}
    combined_dfm9_selected = all(
        preset_name in selected_names
        for preset_name, _ in DFM9_COMBINED_PROPENSITY_SERIES
    )

    for index, preset in enumerate(presets, start=1):
        output_path = preset.output
        plot_output_path = _derive_plot_output_path(preset.output, preset.prefix_summary)

        print(f"[{index}/{len(presets)}] {preset.name}")
        print(f"  metrics: {', '.join(metrics)}")
        print(f"  output: {output_path}")
        if args.plot:
            print(f"  plot: {plot_output_path}")

        if args.dry_run:
            print()
            continue

        try:
            report = _build_report_from_paths(
                setting_to_summary_paths=preset.setting_to_summary_paths,
                prefix_summary_path=preset.prefix_summary,
                metrics=metrics,
            )
            if args.ci:
                report["confidence_intervals"] = _preset_confidence_intervals(preset, metrics, args)
            output_file = _write_report(report, output_path)
            reports_by_name[preset.name] = report
            print(f"  wrote: {output_file}")
        except Exception as exc:
            print(f"Warning: Failed to compute {preset.name}: {exc}. Skipping.")
            failures.append(preset.name)
            print()
            continue

        if args.plot:
            try:
                plot_file = _write_plot(
                    report,
                    plot_output_path=plot_output_path,
                    plot_title=preset.plot_title,
                )
                print(f"  plot: {plot_file}")
            except Exception as exc:
                print(f"Warning: Failed to plot {preset.name}: {exc}.")
                failures.append(f"{preset.name} (plot)")

        print()

    if args.plot and combined_dfm9_selected:
        print("DFM9 combined propensity plot:")
        print(f"  plot: {DFM9_COMBINED_PROPENSITY_PLOT}")
        if not args.dry_run and all(
            preset_name in reports_by_name
            for preset_name, _ in DFM9_COMBINED_PROPENSITY_SERIES
        ):
            try:
                fig = plot_combined_propensity_reports(
                    reports_by_name,
                    DFM9_COMBINED_PROPENSITY_SERIES,
                    metrics,
                    title="DFM9 Propensity Comparisons",
                )
                plt, _ = _get_plotting_modules()
                combined_output = _ensure_path(DFM9_COMBINED_PROPENSITY_PLOT)
                combined_output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(combined_output, dpi=220, bbox_inches="tight")
                plt.close(fig)
                print(f"  wrote: {combined_output}")
            except Exception as exc:
                print(f"Warning: Failed to plot combined DFM9 propensity: {exc}.")
                failures.append("dfm9 combined propensity plot")
        print()

    if failures:
        print("Failed propensity presets:")
        for name in failures:
            print(f"  - {name}")
        return 1

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list:
        _print_available_targets()
        return 0

    direct_mode = _has_direct_summary_inputs(args)

    if direct_mode and args.targets:
        raise SystemExit("Do not mix preset targets with explicit summary path arguments.")

    if direct_mode:
        return _run_direct_mode(args)

    return _run_preset_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())

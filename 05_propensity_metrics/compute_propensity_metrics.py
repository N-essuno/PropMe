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
import tempfile
from dataclasses import dataclass
from pathlib import Path


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


PRESETS = (
    PropensityPreset(
        name="commonpile-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json"),
            ("specific", "memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile/propensity/st_cp_propensity_metrics.json",
        plot_title="CommonPile Generations Propensity Metrics",
        tags=("commonpile", "generations"),
    ),
    PropensityPreset(
        name="commonpile-dfm-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
            ("specific", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_dfm/propensity/st_cp_propensity_metrics.json",
        plot_title="CommonPile DFM Generations Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "generations"),
    ),
    PropensityPreset(
        name="commonpile-dfm-stage1-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
            ("specific", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_dfm_stage1/propensity/st_cp_propensity_metrics.json",
        plot_title="CommonPile DFM Stage 1 Generations Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "commonpile-dfm-stage1", "generations"),
    ),
    PropensityPreset(
        name="commonpile-dfm-stage2-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
            ("specific", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_dfm_stage2/propensity/st_cp_propensity_metrics.json",
        plot_title="CommonPile DFM Stage 2 Generations Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "commonpile-dfm-stage2", "generations"),
    ),
    PropensityPreset(
        name="commonpile-dfm-stages-comparison-generic",
        setting_to_summary_paths=(
            ("generic_stage1", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
            ("generic_stage2", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json",
        output="memorization_experiment/data/commonpile_dfm_stages_comparison/generic/propensity_metrics.json",
        plot_title="CommonPile DFM Stages Comparison Generic Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-dfm-stages-comparison", "generic"),
    ),
    PropensityPreset(
        name="commonpile-dfm-stages-comparison-specific",
        setting_to_summary_paths=(
            ("specific_stage1", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
            ("specific_stage2", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json",
        output="memorization_experiment/data/commonpile_dfm_stages_comparison/specific/propensity_metrics.json",
        plot_title="CommonPile DFM Stages Comparison Specific Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-dfm-stages-comparison", "specific"),
    ),
    PropensityPreset(
        name="commonpile-dfm-stages-comparison-prefix",
        setting_to_summary_paths=(
            ("prefix_stage1", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
            ("prefix_stage2", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_dfm_stages_comparison/prefix/propensity_metrics.json",
        plot_title="CommonPile DFM Stages Comparison Prefix Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-dfm-stages-comparison", "prefix"),
    ),
    PropensityPreset(
        name="commonpile-dfm-dynaword-comparison-generic",
        setting_to_summary_paths=(
            ("dynaword", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json",
        output="memorization_experiment/data/commonpile_dfm_dynaword_comparison/generic/propensity_metrics.json",
        plot_title="CommonPile DFM vs Dynaword Generic Propensity Metrics",
        tags=(
            "commonpile",
            "commonpile-dfm",
            "dynaword",
            "comparison",
            "commonpile-dfm-dynaword-comparison",
            "generic",
        ),
    ),
    PropensityPreset(
        name="commonpile-dfm-dynaword-comparison-specific",
        setting_to_summary_paths=(
            ("dynaword", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json",
        output="memorization_experiment/data/commonpile_dfm_dynaword_comparison/specific/propensity_metrics.json",
        plot_title="CommonPile DFM vs Dynaword Specific Propensity Metrics",
        tags=(
            "commonpile",
            "commonpile-dfm",
            "dynaword",
            "comparison",
            "commonpile-dfm-dynaword-comparison",
            "specific",
        ),
    ),
    PropensityPreset(
        name="commonpile-dfm-dynaword-comparison-prefix",
        setting_to_summary_paths=(
            ("dynaword", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_dfm_dynaword_comparison/prefix/propensity_metrics.json",
        plot_title="CommonPile DFM vs Dynaword Prefix Propensity Metrics",
        tags=(
            "commonpile",
            "commonpile-dfm",
            "dynaword",
            "comparison",
            "commonpile-dfm-dynaword-comparison",
            "prefix",
        ),
    ),
    PropensityPreset(
        name="commonpile-comma-dfm-generic",
        setting_to_summary_paths=(
            ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json",
        output="memorization_experiment/data/commonpile_comma_dfm/generic/propensity_metrics.json",
        plot_title="CommonPile Comma vs DFM Decoder Generic Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-comma-dfm", "generic"),
    ),
    PropensityPreset(
        name="commonpile-comma-dfm-specific",
        setting_to_summary_paths=(
            ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json",
        output="memorization_experiment/data/commonpile_comma_dfm/specific/propensity_metrics.json",
        plot_title="CommonPile Comma vs DFM Decoder Specific Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-comma-dfm", "specific"),
    ),
    PropensityPreset(
        name="commonpile-comma-dfm-prefix",
        setting_to_summary_paths=(
            ("dfm_decoder", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json",
        output="memorization_experiment/data/commonpile_comma_dfm/prefix/propensity_metrics.json",
        plot_title="CommonPile Comma vs DFM Decoder Prefix Propensity Metrics",
        tags=("commonpile", "commonpile-dfm", "comparison", "commonpile-comma-dfm", "prefix"),
    ),
    PropensityPreset(
        name="dynaword-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
            ("specific", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json",
        output="memorization_experiment/data/dynaword/propensity/st_dyna_propensity_metrics.json",
        plot_title="Dynaword Generations Propensity Metrics",
        tags=("dynaword", "generations"),
    ),
    PropensityPreset(
        name="dynaword-stage1-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
            ("specific", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json",
        output="memorization_experiment/data/dynaword_stage1/propensity/st_dyna_propensity_metrics.json",
        plot_title="Dynaword Stage 1 Generations Propensity Metrics",
        tags=("dynaword", "dynaword-stage1", "generations"),
    ),
    PropensityPreset(
        name="dynaword-stage2-generations",
        setting_to_summary_paths=(
            ("generic", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
            ("specific", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json",
        output="memorization_experiment/data/dynaword_stage2/propensity/st_dyna_propensity_metrics.json",
        plot_title="Dynaword Stage 2 Generations Propensity Metrics",
        tags=("dynaword", "dynaword-stage2", "generations"),
    ),
    PropensityPreset(
        name="dynaword-stages-comparison-generic",
        setting_to_summary_paths=(
            ("generic_stage1", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
            ("generic_stage2", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json",
        output="memorization_experiment/data/dynaword_stages_comparison/generic/propensity_metrics.json",
        plot_title="Dynaword Stages Comparison Generic Propensity Metrics",
        tags=("dynaword", "comparison", "dynaword-stages-comparison", "generic"),
    ),
    PropensityPreset(
        name="dynaword-stages-comparison-specific",
        setting_to_summary_paths=(
            ("specific_stage1", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
            ("specific_stage2", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json",
        output="memorization_experiment/data/dynaword_stages_comparison/specific/propensity_metrics.json",
        plot_title="Dynaword Stages Comparison Specific Propensity Metrics",
        tags=("dynaword", "comparison", "dynaword-stages-comparison", "specific"),
    ),
    PropensityPreset(
        name="dynaword-stages-comparison-prefix",
        setting_to_summary_paths=(
            ("prefix_stage1", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
            ("prefix_stage2", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
        ),
        prefix_summary="memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json",
        output="memorization_experiment/data/dynaword_stages_comparison/prefix/propensity_metrics.json",
        plot_title="Dynaword Stages Comparison Prefix Propensity Metrics",
        tags=("dynaword", "comparison", "dynaword-stages-comparison", "prefix"),
    ),
)

PRESETS_BY_NAME = {preset.name: preset for preset in PRESETS}

GROUPS = {
    "all": [preset.name for preset in PRESETS],
    "all-commonpiles": [preset.name for preset in PRESETS if "commonpile" in preset.tags],
    "all-dynawords": [preset.name for preset in PRESETS if "dynaword" in preset.tags],
    "all-generations": [preset.name for preset in PRESETS if "generations" in preset.tags],
    "all-comparisons": [preset.name for preset in PRESETS if "comparison" in preset.tags],
    "commonpile": [preset.name for preset in PRESETS if "commonpile" in preset.tags],
    "commonpile-generations": [
        preset.name
        for preset in PRESETS
        if "commonpile" in preset.tags and "generations" in preset.tags
    ],
    "commonpile-comparisons": [
        preset.name
        for preset in PRESETS
        if "commonpile" in preset.tags and "comparison" in preset.tags
    ],
    "commonpile-dfm": [preset.name for preset in PRESETS if "commonpile-dfm" in preset.tags],
    "commonpile-dfm-generations": [
        preset.name
        for preset in PRESETS
        if "commonpile-dfm" in preset.tags and "generations" in preset.tags
    ],
    "commonpile-dfm-stage1": [
        preset.name for preset in PRESETS if "commonpile-dfm-stage1" in preset.tags
    ],
    "commonpile-dfm-stage1-generations": [
        preset.name
        for preset in PRESETS
        if "commonpile-dfm-stage1" in preset.tags and "generations" in preset.tags
    ],
    "commonpile-dfm-stage2": [
        preset.name for preset in PRESETS if "commonpile-dfm-stage2" in preset.tags
    ],
    "commonpile-dfm-stage2-generations": [
        preset.name
        for preset in PRESETS
        if "commonpile-dfm-stage2" in preset.tags and "generations" in preset.tags
    ],
    "commonpile-dfm-stages-comparison": [
        preset.name
        for preset in PRESETS
        if "commonpile-dfm-stages-comparison" in preset.tags
    ],
    "commonpile-dfm-dynaword-comparison": [
        preset.name
        for preset in PRESETS
        if "commonpile-dfm-dynaword-comparison" in preset.tags
    ],
    "commonpile-comma-dfm": [
        preset.name
        for preset in PRESETS
        if "commonpile-comma-dfm" in preset.tags
    ],
    "dynaword": [preset.name for preset in PRESETS if "dynaword" in preset.tags],
    "dynaword-generations": [
        preset.name
        for preset in PRESETS
        if "dynaword" in preset.tags and "generations" in preset.tags
    ],
    "dynaword-comparisons": [
        preset.name
        for preset in PRESETS
        if "dynaword" in preset.tags and "comparison" in preset.tags
    ],
    "dynaword-stage1": [preset.name for preset in PRESETS if "dynaword-stage1" in preset.tags],
    "dynaword-stage1-generations": [
        preset.name
        for preset in PRESETS
        if "dynaword-stage1" in preset.tags and "generations" in preset.tags
    ],
    "dynaword-stage2": [preset.name for preset in PRESETS if "dynaword-stage2" in preset.tags],
    "dynaword-stage2-generations": [
        preset.name
        for preset in PRESETS
        if "dynaword-stage2" in preset.tags and "generations" in preset.tags
    ],
    "dynaword-stages-comparison": [
        preset.name
        for preset in PRESETS
        if "dynaword-stages-comparison" in preset.tags
    ],
}


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
            "Plotting requires matplotlib and numpy. Use the project virtualenv, "
            "for example `.venv/bin/python 05_propensity_metrics/compute_propensity_metrics.py ... --plot`."
        ) from exc

    return plt, np


def plot_propensity_summary(summary: dict, *, title: str | None = None):
    plt, np = _get_plotting_modules()

    setting_colors = ["#1F449C", "#009E73"]

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
            output_file = _write_report(report, output_path)
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

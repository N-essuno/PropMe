#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

BASE_RED = "#A50922"
COLOR_A = "#F05039"
COLOR_D = "#1F449C"
COLOR_H = "#009E73"
REPO_ROOT = Path(__file__).resolve().parents[1]
plt = None
np = None
to_rgb = None


@dataclass(frozen=True)
class PlotSuite:
    name: str
    filepaths: tuple[tuple[str, str], ...]
    plots_dir: str
    tags: tuple[str, ...]


PLOT_SUITES = (
    PlotSuite(
        name="dynaword-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword/plots",
        tags=("dynaword", "generations"),
    ),
    PlotSuite(
        name="dynaword-prompts",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword/generic/st_dyna_prompts_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword/specific/st_dyna_prompts_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword/prefix/st_dyna_prompts_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword/plots/prompts",
        tags=("dynaword", "prompts"),
    ),
    PlotSuite(
        name="dynaword-stage1-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stage1/plots",
        tags=("dynaword-stage1", "generations"),
    ),
    PlotSuite(
        name="dynaword-stage1-prompts",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_prompts_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_prompts_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prompts_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stage1/plots/prompts",
        tags=("dynaword-stage1", "prompts"),
    ),
    PlotSuite(
        name="dynaword-stage2-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stage2/plots",
        tags=("dynaword-stage2", "generations"),
    ),
    PlotSuite(
        name="dynaword-stage2-prompts",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_prompts_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_prompts_summary.json"),
            ("Prefix (Dynaword)", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prompts_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stage2/plots/prompts",
        tags=("dynaword-stage2", "prompts"),
    ),
    PlotSuite(
        name="dynaword-stages-comparison-generic",
        filepaths=(
            ("generic_stage1", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
            ("generic_stage2", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
            ("generic", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stages_comparison/generic",
        tags=("dynaword-stages-comparison", "stage-comparison", "generic"),
    ),
    PlotSuite(
        name="dynaword-stages-comparison-specific",
        filepaths=(
            ("specific_stage1", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
            ("specific_stage2", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
            ("specific", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stages_comparison/specific",
        tags=("dynaword-stages-comparison", "stage-comparison", "specific"),
    ),
    PlotSuite(
        name="dynaword-stages-comparison-prefix",
        filepaths=(
            ("prefix_stage1", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
            ("prefix_stage2", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
            ("prefix", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_stages_comparison/prefix",
        tags=("dynaword-stages-comparison", "stage-comparison", "prefix"),
    ),
    PlotSuite(
        name="commonpile-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json"),
            ("Prefix (Common Pile)", "memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile/plots",
        tags=("commonpile", "generations"),
    ),
    PlotSuite(
        name="commonpile-prompts",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/commonpile/generic/st_cp_prompts_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/commonpile/specific/st_cp_prompts_summary.json"),
            ("Prefix (Common Pile)", "memorization_experiment/data/commonpile/prefix/st_cp_prompts_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile/plots/prompts",
        tags=("commonpile", "prompts"),
    ),
    PlotSuite(
        name="commonpile-dfm-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
            ("Prefix (Common Pile)", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm/plots",
        tags=("commonpile-dfm", "generations"),
    ),
    PlotSuite(
        name="commonpile-dfm-stage1-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
            ("Prefix (Common Pile)", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_stage1/plots",
        tags=("commonpile-dfm-stage1", "generations"),
    ),
    PlotSuite(
        name="commonpile-dfm-stage2-generations",
        filepaths=(
            ("Generic Prompts", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
            ("Specific Prompts", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
            ("Prefix (Common Pile)", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_stage2/plots",
        tags=("commonpile-dfm-stage2", "generations"),
    ),
    PlotSuite(
        name="commonpile-dfm-stages-comparison-generic",
        filepaths=(
            ("generic_stage1", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
            ("generic_stage2", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
            ("generic", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_stages_comparison/generic",
        tags=("commonpile-dfm-stages-comparison", "stage-comparison", "generic"),
    ),
    PlotSuite(
        name="commonpile-dfm-stages-comparison-specific",
        filepaths=(
            ("specific_stage1", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
            ("specific_stage2", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
            ("specific", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_stages_comparison/specific",
        tags=("commonpile-dfm-stages-comparison", "stage-comparison", "specific"),
    ),
    PlotSuite(
        name="commonpile-dfm-stages-comparison-prefix",
        filepaths=(
            ("prefix_stage1", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
            ("prefix_stage2", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
            ("prefix", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_stages_comparison/prefix",
        tags=("commonpile-dfm-stages-comparison", "stage-comparison", "prefix"),
    ),
    PlotSuite(
        name="commonpile-dfm-dynaword-comparison-generic",
        filepaths=(
            ("commonpile_dfm", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
            ("dynaword", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_dynaword_comparison/generic",
        tags=("commonpile-dfm-dynaword-comparison", "cross-dataset-comparison", "generic"),
    ),
    PlotSuite(
        name="commonpile-dfm-dynaword-comparison-specific",
        filepaths=(
            ("commonpile_dfm", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
            ("dynaword", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_dynaword_comparison/specific",
        tags=("commonpile-dfm-dynaword-comparison", "cross-dataset-comparison", "specific"),
    ),
    PlotSuite(
        name="commonpile-dfm-dynaword-comparison-prefix",
        filepaths=(
            ("commonpile_dfm", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
            ("dynaword", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_dfm_dynaword_comparison/prefix",
        tags=("commonpile-dfm-dynaword-comparison", "cross-dataset-comparison", "prefix"),
    ),
    PlotSuite(
        name="dynaword-commonpile-stages-comparison-generic",
        filepaths=(
            ("Common Pile Stage 1", "memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json"),
            ("Dynaword Stage 1", "memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json"),
            ("Common Pile Stage 2", "memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json"),
            ("Dynaword Stage 2", "memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json"),
            ("Common Pile", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
            ("Dynaword", "memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_commonpile_stages_comparison/generic",
        tags=("dynaword-commonpile-stages-comparison", "cross-dataset-stage-comparison", "generic"),
    ),
    PlotSuite(
        name="dynaword-commonpile-stages-comparison-specific",
        filepaths=(
            ("Common Pile Stage 1", "memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json"),
            ("Dynaword Stage 1", "memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json"),
            ("Common Pile Stage 2", "memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json"),
            ("Dynaword Stage 2", "memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json"),
            ("Common Pile", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
            ("Dynaword", "memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_commonpile_stages_comparison/specific",
        tags=("dynaword-commonpile-stages-comparison", "cross-dataset-stage-comparison", "specific"),
    ),
    PlotSuite(
        name="dynaword-commonpile-stages-comparison-prefix",
        filepaths=(
            ("Common Pile Stage 1", "memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json"),
            ("Dynaword Stage 1", "memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json"),
            ("Common Pile Stage 2", "memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json"),
            ("Dynaword Stage 2", "memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json"),
            ("Common Pile", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
            ("Dynaword", "memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/dynaword_commonpile_stages_comparison/prefix",
        tags=("dynaword-commonpile-stages-comparison", "cross-dataset-stage-comparison", "prefix"),
    ),
    PlotSuite(
        name="commonpile-comma-dfm-generic",
        filepaths=(
            ("Comma", "memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json"),
            ("DFM Decoder", "memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_comma_dfm/generic",
        tags=("commonpile-comma-dfm", "cross-dataset-comparison", "generic"),
    ),
    PlotSuite(
        name="commonpile-comma-dfm-specific",
        filepaths=(
            ("Comma", "memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json"),
            ("DFM Decoder", "memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_comma_dfm/specific",
        tags=("commonpile-comma-dfm", "cross-dataset-comparison", "specific"),
    ),
    PlotSuite(
        name="commonpile-comma-dfm-prefix",
        filepaths=(
            ("Comma", "memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json"),
            ("DFM Decoder", "memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json"),
        ),
        plots_dir="memorization_experiment/data/commonpile_comma_dfm/prefix",
        tags=("commonpile-comma-dfm", "cross-dataset-comparison", "prefix"),
    ),
)

PLOT_SUITES_BY_NAME = {suite.name: suite for suite in PLOT_SUITES}

GROUPS = {
    "all": [suite.name for suite in PLOT_SUITES],
    "all-generations": [suite.name for suite in PLOT_SUITES if "generations" in suite.tags],
    "all-prompts": [suite.name for suite in PLOT_SUITES if "prompts" in suite.tags],
    "dynaword": [suite.name for suite in PLOT_SUITES if "dynaword" in suite.tags],
    "dynaword-generations": [
        suite.name for suite in PLOT_SUITES if "dynaword" in suite.tags and "generations" in suite.tags
    ],
    "dynaword-prompts": [
        suite.name for suite in PLOT_SUITES if "dynaword" in suite.tags and "prompts" in suite.tags
    ],
    "dynaword-stage1": [suite.name for suite in PLOT_SUITES if "dynaword-stage1" in suite.tags],
    "dynaword-stage1-generations": [
        suite.name for suite in PLOT_SUITES if "dynaword-stage1" in suite.tags and "generations" in suite.tags
    ],
    "dynaword-stage1-prompts": [
        suite.name for suite in PLOT_SUITES if "dynaword-stage1" in suite.tags and "prompts" in suite.tags
    ],
    "dynaword-stage2": [suite.name for suite in PLOT_SUITES if "dynaword-stage2" in suite.tags],
    "dynaword-stage2-generations": [
        suite.name for suite in PLOT_SUITES if "dynaword-stage2" in suite.tags and "generations" in suite.tags
    ],
    "dynaword-stage2-prompts": [
        suite.name for suite in PLOT_SUITES if "dynaword-stage2" in suite.tags and "prompts" in suite.tags
    ],
    "dynaword-stages-comparison": [
        suite.name for suite in PLOT_SUITES if "dynaword-stages-comparison" in suite.tags
    ],
    "dynaword-stages-comparison-generic": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-stages-comparison" in suite.tags and "generic" in suite.tags
    ],
    "dynaword-stages-comparison-specific": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-stages-comparison" in suite.tags and "specific" in suite.tags
    ],
    "dynaword-stages-comparison-prefix": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-stages-comparison" in suite.tags and "prefix" in suite.tags
    ],
    "commonpile": [suite.name for suite in PLOT_SUITES if "commonpile" in suite.tags],
    "commonpile-generations": [
        suite.name for suite in PLOT_SUITES if "commonpile" in suite.tags and "generations" in suite.tags
    ],
    "commonpile-prompts": [
        suite.name for suite in PLOT_SUITES if "commonpile" in suite.tags and "prompts" in suite.tags
    ],
    "commonpile-dfm": [suite.name for suite in PLOT_SUITES if "commonpile-dfm" in suite.tags],
    "commonpile-dfm-generations": [
        suite.name for suite in PLOT_SUITES if "commonpile-dfm" in suite.tags and "generations" in suite.tags
    ],
    "commonpile-dfm-stage1": [
        suite.name for suite in PLOT_SUITES if "commonpile-dfm-stage1" in suite.tags
    ],
    "commonpile-dfm-stage1-generations": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-stage1" in suite.tags and "generations" in suite.tags
    ],
    "commonpile-dfm-stage2": [
        suite.name for suite in PLOT_SUITES if "commonpile-dfm-stage2" in suite.tags
    ],
    "commonpile-dfm-stage2-generations": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-stage2" in suite.tags and "generations" in suite.tags
    ],
    "commonpile-dfm-stages-comparison": [
        suite.name for suite in PLOT_SUITES if "commonpile-dfm-stages-comparison" in suite.tags
    ],
    "commonpile-dfm-stages-comparison-generic": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-stages-comparison" in suite.tags and "generic" in suite.tags
    ],
    "commonpile-dfm-stages-comparison-specific": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-stages-comparison" in suite.tags and "specific" in suite.tags
    ],
    "commonpile-dfm-stages-comparison-prefix": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-stages-comparison" in suite.tags and "prefix" in suite.tags
    ],
    "commonpile-dfm-dynaword-comparison": [
        suite.name for suite in PLOT_SUITES if "commonpile-dfm-dynaword-comparison" in suite.tags
    ],
    "commonpile-dfm-dynaword-comparison-generic": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-dynaword-comparison" in suite.tags and "generic" in suite.tags
    ],
    "commonpile-dfm-dynaword-comparison-specific": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-dynaword-comparison" in suite.tags and "specific" in suite.tags
    ],
    "commonpile-dfm-dynaword-comparison-prefix": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-dfm-dynaword-comparison" in suite.tags and "prefix" in suite.tags
    ],
    "dynaword-commonpile-stages-comparison": [
        suite.name for suite in PLOT_SUITES if "dynaword-commonpile-stages-comparison" in suite.tags
    ],
    "dynaword-commonpile-stages-comparison-generic": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-commonpile-stages-comparison" in suite.tags and "generic" in suite.tags
    ],
    "dynaword-commonpile-stages-comparison-specific": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-commonpile-stages-comparison" in suite.tags and "specific" in suite.tags
    ],
    "dynaword-commonpile-stages-comparison-prefix": [
        suite.name
        for suite in PLOT_SUITES
        if "dynaword-commonpile-stages-comparison" in suite.tags and "prefix" in suite.tags
    ],
    "commonpile-comma-dfm": [
        suite.name for suite in PLOT_SUITES if "commonpile-comma-dfm" in suite.tags
    ],
    "commonpile-comma-dfm-generic": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-comma-dfm" in suite.tags and "generic" in suite.tags
    ],
    "commonpile-comma-dfm-specific": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-comma-dfm" in suite.tags and "specific" in suite.tags
    ],
    "commonpile-comma-dfm-prefix": [
        suite.name
        for suite in PLOT_SUITES
        if "commonpile-comma-dfm" in suite.tags and "prefix" in suite.tags
    ],
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
        "generations_above_nv_recall_threshold_ratio": [
            "generations_above_nv_recall_threshold_ratio",
        ],
    }
    candidates = aliases.get(metric_name, [metric_name])
    for key in candidates:
        if key in summary:
            return summary[key]
    return 0.0


def _plot_scalar_metric(ax, labels, summaries, metric_name: str):
    values = [_metric_value(summaries[label], metric_name) for label in labels]
    palette = _value_based_red_colors(values)
    bars = ax.bar(labels, values, color=palette, edgecolor="#333333", linewidth=1.0)
    for bar in bars:
        h = float(bar.get_height())
        ax.annotate(
            _format_value(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3) if h >= 0 else (0, -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
        )
    ax.set_title(metric_name)
    ax.set_ylabel("Value")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=15)
    if metric_name == "avg_nv_recall" and values:
        ax.set_ylim(0.0, max(values) + 0.05)


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


def _plot_exact_span_distribution(ax, labels, exact_span_data: dict[str, dict]):
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
    palette = _distribution_palette(len(labels))
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

    ax.set_title("spans_length_distribution_exact (smoothed)")
    ax.set_xlabel("Span Length (tokens)")
    ax.set_ylabel("Ratio")
    ax.set_xticks(ticks)
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
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()


def _save_single_plot(plots_dir: str, filename: str, plotter):
    fig, ax = plt.subplots(figsize=(9, 6))
    plotter(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200)
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
        "generations_above_nv_recall_threshold_ratio",
    ]

    k_values = _extract_k_values(summaries)
    k_metrics = [f"k_eidetic_rate_k_le_{k}" for k in k_values]

    for metric in scalar_metrics:
        _save_single_plot(
            str(plots_dir),
            f"{metric}.png",
            lambda ax, metric_name=metric: _plot_scalar_metric(ax, labels, summaries, metric_name),
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

    _save_single_plot(
        str(plots_dir),
        "spans_length_distribution.png",
        lambda ax: _plot_bucketed_span_distribution(ax, labels, summaries),
    )

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
        if kind in ("scalar", "k"):
            _plot_scalar_metric(ax, labels, summaries, metric_name)
        elif kind == "distribution_exact":
            _plot_exact_span_distribution(ax, labels, exact_span_data)
        else:
            _plot_bucketed_span_distribution(ax, labels, summaries)

    for ax in axes_flat[n_plots:]:
        ax.set_axis_off()

    fig.suptitle("SimpleTrace Metrics Across 3 Evaluation Settings", fontsize=16)
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

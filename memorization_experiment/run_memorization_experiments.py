#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_TRACE_PATH = REPO_ROOT / "03_tracing" / "simple_trace.py"

DEFAULT_LENGTH_BUCKETS = "1-3,4-6,7-10,11-20,21-49,50-75,76-99,100-150,151-inf"
DEFAULT_K_EIDETIC_VALUES = "1,5,10"

INDEX_DEFAULTS = {
    "commonpile": "/work/olmotrace/common_pile_train/indexes/common_pile_train_index",
    "dynaword": "00_data/dynaword_index",
    "dfm9-a": "/work/olmotrace/mimir_propme/indexes/A",
    "dfm9-b": "/work/olmotrace/mimir_propme/indexes/B",
    "dfm9-c": "/work/olmotrace/mimir_propme/indexes/C",
    "dfm9-d": "/work/olmotrace/mimir_propme/indexes/D",
    "dfm9-ab": (
        "/work/olmotrace/mimir_propme/indexes/A",
        "/work/olmotrace/mimir_propme/indexes/B",
    ),
}

UNIGRAM_DEFAULTS = {
    "commonpile": "02_unigram_probs/unigram_probs_common_pile_train.json",
    "dynaword": "02_unigram_probs/unigram_probs_dynaword.json",
    "dfm9-a": "02_unigram_probs/unigram_probs_dfm9_A.json",
    "dfm9-b": "02_unigram_probs/unigram_probs_dfm9_B.json",
    "dfm9-c": "02_unigram_probs/unigram_probs_dfm9_C.json",
    "dfm9-d": "02_unigram_probs/unigram_probs_dfm9_D.json",
    "dfm9-ab": "02_unigram_probs/unigram_probs_dfm9_AB.json",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    dataset_flag: str
    index_key: str
    unigram_key: str
    num_workers: int
    docs_per_span: int
    results_output: str
    summary_output: str
    n_token_span_ratio: int
    match_mode: str | None
    tags: tuple[str, ...]
    length_buckets: str = DEFAULT_LENGTH_BUCKETS
    k_eidetic_values: str = DEFAULT_K_EIDETIC_VALUES


EXPERIMENTS = (
    Experiment(
        name="dfm9-generations-generic-en-a",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_en_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-a",
        unigram_key="dfm9-a",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_A_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_A_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-en", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-en-b",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_en_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-b",
        unigram_key="dfm9-b",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_B_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_B_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-en", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-en-c",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_en_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-c",
        unigram_key="dfm9-c",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_C_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_C_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-en", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-en-d",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_en_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-d",
        unigram_key="dfm9-d",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_D_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_D_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-en", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-da-a",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_da_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-a",
        unigram_key="dfm9-a",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_A_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_A_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-da", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-da-b",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_da_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-b",
        unigram_key="dfm9-b",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_B_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_B_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-da", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-da-c",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_da_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-c",
        unigram_key="dfm9-c",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_C_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_C_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-da", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-generic-da-d",
        dataset="memorization_experiment/data/dfm9/generic/dfm9_generic_da_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-d",
        unigram_key="dfm9-d",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_D_results.json",
        summary_output="memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_D_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "generations", "generic", "generic-da", "generic-by-category"),
    ),
    Experiment(
        name="dfm9-generations-prefix-a-50",
        dataset="memorization_experiment/data/dfm9/prefix/dfm9_A_prefix_50_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-a",
        unigram_key="dfm9-a",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_results.json",
        summary_output="memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "dfm9-a", "generations", "prefix", "prefix-50"),
    ),
    Experiment(
        name="dfm9-generations-prefix-b-50",
        dataset="memorization_experiment/data/dfm9/prefix/dfm9_B_prefix_50_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-b",
        unigram_key="dfm9-b",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_results.json",
        summary_output="memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "dfm9-b", "generations", "prefix", "prefix-50"),
    ),
    Experiment(
        name="dfm9-generations-prefix-c-50",
        dataset="memorization_experiment/data/dfm9/prefix/dfm9_C_prefix_50_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-c",
        unigram_key="dfm9-c",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_results.json",
        summary_output="memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "dfm9-c", "generations", "prefix", "prefix-50"),
    ),
    Experiment(
        name="dfm9-generations-prefix-d-50",
        dataset="memorization_experiment/data/dfm9/prefix/dfm9_D_prefix_50_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dfm9-d",
        unigram_key="dfm9-d",
        num_workers=128,
        docs_per_span=10,
        results_output="memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_results.json",
        summary_output="memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("dfm9", "dfm9-d", "generations", "prefix", "prefix-50"),
    ),
    Experiment(
        name="commonpile-dfm-generations-generic",
        dataset="memorization_experiment/data/commonpile_dfm/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm/generic/st_cp_generic_summary.json",
        n_token_span_ratio=50,
        match_mode="mixed",
        tags=("commonpile-dfm", "generations", "generic"),
    ),
    Experiment(
        name="commonpile-dfm-generations-specific",
        dataset="memorization_experiment/data/commonpile_dfm/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm/specific/st_cp_specific_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm", "generations", "specific"),
    ),
    Experiment(
        name="commonpile-dfm-generations-prefix",
        dataset="memorization_experiment/data/commonpile_dfm/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm/prefix/st_cp_prefix_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm", "generations", "prefix"),
    ),
    Experiment(
        name="commonpile-dfm-stage1-generations-generic",
        dataset="memorization_experiment/data/commonpile_dfm_stage1/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage1/generic/st_cp_generic_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage1", "generations", "generic"),
    ),
    Experiment(
        name="commonpile-dfm-stage1-generations-specific",
        dataset="memorization_experiment/data/commonpile_dfm_stage1/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage1/specific/st_cp_specific_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage1", "generations", "specific"),
    ),
    Experiment(
        name="commonpile-dfm-stage1-generations-prefix",
        dataset="memorization_experiment/data/commonpile_dfm_stage1/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage1/prefix/st_cp_prefix_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage1", "generations", "prefix"),
    ),
    Experiment(
        name="commonpile-dfm-stage2-generations-generic",
        dataset="memorization_experiment/data/commonpile_dfm_stage2/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage2/generic/st_cp_generic_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage2", "generations", "generic"),
    ),
    Experiment(
        name="commonpile-dfm-stage2-generations-specific",
        dataset="memorization_experiment/data/commonpile_dfm_stage2/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage2/specific/st_cp_specific_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage2", "generations", "specific"),
    ),
    Experiment(
        name="commonpile-dfm-stage2-generations-prefix",
        dataset="memorization_experiment/data/commonpile_dfm_stage2/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_results.json",
        summary_output="memorization_experiment/data/commonpile_dfm_stage2/prefix/st_cp_prefix_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile-dfm-stage2", "generations", "prefix"),
    ),
    Experiment(
        name="commonpile-generations-generic",
        dataset="memorization_experiment/data/commonpile/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/generic/st_cp_generic_results.json",
        summary_output="memorization_experiment/data/commonpile/generic/st_cp_generic_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile", "generations", "generic"),
    ),
    Experiment(
        name="commonpile-generations-specific",
        dataset="memorization_experiment/data/commonpile/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/specific/st_cp_specific_results.json",
        summary_output="memorization_experiment/data/commonpile/specific/st_cp_specific_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile", "generations", "specific"),
    ),
    Experiment(
        name="commonpile-generations-prefix",
        dataset="memorization_experiment/data/commonpile/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/prefix/st_cp_prefix_results.json",
        summary_output="memorization_experiment/data/commonpile/prefix/st_cp_prefix_summary.json",
        n_token_span_ratio=70,
        match_mode="mixed",
        tags=("commonpile", "generations", "prefix"),
    ),
    Experiment(
        name="dynaword-generations-generic",
        dataset="memorization_experiment/data/dynaword/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/generic/st_dyna_generic_results.json",
        summary_output="memorization_experiment/data/dynaword/generic/st_dyna_generic_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "generations", "generic"),
    ),
    Experiment(
        name="dynaword-generations-specific",
        dataset="memorization_experiment/data/dynaword/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/specific/st_dyna_specific_results.json",
        summary_output="memorization_experiment/data/dynaword/specific/st_dyna_specific_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "generations", "specific"),
    ),
    Experiment(
        name="dynaword-generations-prefix",
        dataset="memorization_experiment/data/dynaword/prefix/dynaword_prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/prefix/st_dyna_prefix_results.json",
        summary_output="memorization_experiment/data/dynaword/prefix/st_dyna_prefix_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "generations", "prefix"),
    ),
    Experiment(
        name="dynaword-stage1-generations-generic",
        dataset="memorization_experiment/data/dynaword_stage1/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/generic/st_dyna_generic_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "generations", "generic"),
    ),
    Experiment(
        name="dynaword-stage1-generations-specific",
        dataset="memorization_experiment/data/dynaword_stage1/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/specific/st_dyna_specific_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "generations", "specific"),
    ),
    Experiment(
        name="dynaword-stage1-generations-prefix",
        dataset="memorization_experiment/data/dynaword_stage1/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prefix_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "generations", "prefix"),
    ),
    Experiment(
        name="dynaword-stage2-generations-generic",
        dataset="memorization_experiment/data/dynaword_stage2/generic/generic_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/generic/st_dyna_generic_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "generations", "generic"),
    ),
    Experiment(
        name="dynaword-stage2-generations-specific",
        dataset="memorization_experiment/data/dynaword_stage2/specific/specific_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/specific/st_dyna_specific_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "generations", "specific"),
    ),
    Experiment(
        name="dynaword-stage2-generations-prefix",
        dataset="memorization_experiment/data/dynaword_stage2/prefix/prefix_generations.json",
        dataset_flag="--is-generation-json",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prefix_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "generations", "prefix"),
    ),
    Experiment(
        name="dynaword-prompts-generic",
        dataset="memorization_experiment/data/dynaword/generic/generic_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/generic/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword/generic/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "prompts", "generic"),
    ),
    Experiment(
        name="dynaword-prompts-specific",
        dataset="memorization_experiment/data/dynaword/specific/specific_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/specific/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword/specific/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "prompts", "specific"),
    ),
    Experiment(
        name="dynaword-prompts-prefix",
        dataset="memorization_experiment/data/dynaword/prefix/dynaword_prefix_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword/prefix/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword/prefix/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword", "prompts", "prefix"),
    ),
    Experiment(
        name="dynaword-stage1-prompts-generic",
        dataset="memorization_experiment/data/dynaword_stage1/generic/generic_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/generic/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/generic/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "prompts", "generic"),
    ),
    Experiment(
        name="dynaword-stage1-prompts-specific",
        dataset="memorization_experiment/data/dynaword_stage1/specific/specific_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/specific/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/specific/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "prompts", "specific"),
    ),
    Experiment(
        name="dynaword-stage1-prompts-prefix",
        dataset="memorization_experiment/data/dynaword_stage1/prefix/dynaword_prefix_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage1/prefix/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage1", "prompts", "prefix"),
    ),
    Experiment(
        name="dynaword-stage2-prompts-generic",
        dataset="memorization_experiment/data/dynaword_stage2/generic/generic_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/generic/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/generic/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "prompts", "generic"),
    ),
    Experiment(
        name="dynaword-stage2-prompts-specific",
        dataset="memorization_experiment/data/dynaword_stage2/specific/specific_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/specific/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/specific/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "prompts", "specific"),
    ),
    Experiment(
        name="dynaword-stage2-prompts-prefix",
        dataset="memorization_experiment/data/dynaword_stage2/prefix/dynaword_prefix_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="dynaword",
        unigram_key="dynaword",
        num_workers=10,
        docs_per_span=10,
        results_output="memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prompts_results.json",
        summary_output="memorization_experiment/data/dynaword_stage2/prefix/st_dyna_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode=None,
        tags=("dynaword-stage2", "prompts", "prefix"),
    ),
    Experiment(
        name="commonpile-prompts-generic",
        dataset="memorization_experiment/data/commonpile/generic/generic_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/generic/st_cp_prompts_results.json",
        summary_output="memorization_experiment/data/commonpile/generic/st_cp_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode="mixed",
        tags=("commonpile", "prompts", "generic"),
    ),
    Experiment(
        name="commonpile-prompts-specific",
        dataset="memorization_experiment/data/commonpile/specific/specific_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=4,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/specific/st_cp_prompts_results.json",
        summary_output="memorization_experiment/data/commonpile/specific/st_cp_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode="mixed",
        tags=("commonpile", "prompts", "specific"),
    ),
    Experiment(
        name="commonpile-prompts-prefix",
        dataset="memorization_experiment/data/commonpile/prefix/commonpile_prefix_prompts.jsonl",
        dataset_flag="--is-jsonl",
        index_key="commonpile",
        unigram_key="commonpile",
        num_workers=1,
        docs_per_span=10,
        results_output="memorization_experiment/data/commonpile/prefix/st_cp_prompts_results.json",
        summary_output="memorization_experiment/data/commonpile/prefix/st_cp_prompts_summary.json",
        n_token_span_ratio=119,
        match_mode="mixed",
        tags=("commonpile", "prompts", "prefix"),
    ),
)

EXPERIMENTS_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}

GROUPS = {
    "all": [experiment.name for experiment in EXPERIMENTS],
    "all-generations": [experiment.name for experiment in EXPERIMENTS if "generations" in experiment.tags],
    "all-prompts": [experiment.name for experiment in EXPERIMENTS if "prompts" in experiment.tags],
    "dfm9": [experiment.name for experiment in EXPERIMENTS if "dfm9" in experiment.tags],
    "dfm9-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dfm9" in experiment.tags and "generations" in experiment.tags
    ],
    "commonpile": [experiment.name for experiment in EXPERIMENTS if "commonpile" in experiment.tags],
    "commonpile-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "commonpile" in experiment.tags and "generations" in experiment.tags
    ],
    "commonpile-prompts": [
        experiment.name
        for experiment in EXPERIMENTS
        if "commonpile" in experiment.tags and "prompts" in experiment.tags
    ],
    "commonpile-dfm": [experiment.name for experiment in EXPERIMENTS if "commonpile-dfm" in experiment.tags],
    "commonpile-dfm-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "commonpile-dfm" in experiment.tags and "generations" in experiment.tags
    ],
    "commonpile-dfm-stage1": [
        experiment.name for experiment in EXPERIMENTS if "commonpile-dfm-stage1" in experiment.tags
    ],
    "commonpile-dfm-stage1-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "commonpile-dfm-stage1" in experiment.tags and "generations" in experiment.tags
    ],
    "commonpile-dfm-stage2": [
        experiment.name for experiment in EXPERIMENTS if "commonpile-dfm-stage2" in experiment.tags
    ],
    "commonpile-dfm-stage2-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "commonpile-dfm-stage2" in experiment.tags and "generations" in experiment.tags
    ],
    "dynaword": [experiment.name for experiment in EXPERIMENTS if "dynaword" in experiment.tags],
    "dynaword-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword" in experiment.tags and "generations" in experiment.tags
    ],
    "dynaword-prompts": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword" in experiment.tags and "prompts" in experiment.tags
    ],
    "dynaword-stage1": [experiment.name for experiment in EXPERIMENTS if "dynaword-stage1" in experiment.tags],
    "dynaword-stage1-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword-stage1" in experiment.tags and "generations" in experiment.tags
    ],
    "dynaword-stage1-prompts": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword-stage1" in experiment.tags and "prompts" in experiment.tags
    ],
    "dynaword-stage2": [experiment.name for experiment in EXPERIMENTS if "dynaword-stage2" in experiment.tags],
    "dynaword-stage2-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword-stage2" in experiment.tags and "generations" in experiment.tags
    ],
    "dynaword-stage2-prompts": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dynaword-stage2" in experiment.tags and "prompts" in experiment.tags
    ],
    "generic": [experiment.name for experiment in EXPERIMENTS if "generic" in experiment.tags],
    "specific": [experiment.name for experiment in EXPERIMENTS if "specific" in experiment.tags],
    "prefix": [experiment.name for experiment in EXPERIMENTS if "prefix" in experiment.tags],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SimpleTrace memorization experiments from cmds.txt with named presets."
        )
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help=(
            "Experiment names and/or group names to run. "
            "Use --list to see the available options."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available experiment names and group names, then exit.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking simple_trace.py.",
    )
    parser.add_argument(
        "--commonpile-index-dir",
        default=INDEX_DEFAULTS["commonpile"],
        help="Override the CommonPile index directory.",
    )
    parser.add_argument(
        "--dynaword-index-dir",
        default=INDEX_DEFAULTS["dynaword"],
        help="Override the Dynaword index directory.",
    )
    parser.add_argument(
        "--dfm9-a-index-dir",
        default=INDEX_DEFAULTS["dfm9-a"],
        help="Override the DFM9 A index directory.",
    )
    parser.add_argument(
        "--dfm9-b-index-dir",
        default=INDEX_DEFAULTS["dfm9-b"],
        help="Override the DFM9 B index directory.",
    )
    parser.add_argument(
        "--dfm9-c-index-dir",
        default=INDEX_DEFAULTS["dfm9-c"],
        help="Override the DFM9 C index directory.",
    )
    parser.add_argument(
        "--dfm9-d-index-dir",
        default=INDEX_DEFAULTS["dfm9-d"],
        help="Override the DFM9 D index directory.",
    )
    parser.add_argument(
        "--dfm9-generic-index-dirs",
        nargs="+",
        default=list(INDEX_DEFAULTS["dfm9-ab"]),
        help="Override the index directories combined for DFM9 generic experiments.",
    )
    parser.add_argument(
        "--commonpile-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["commonpile"],
        help="Override the CommonPile unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dynaword-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dynaword"],
        help="Override the Dynaword unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dfm9-a-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dfm9-a"],
        help="Override the DFM9 A unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dfm9-b-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dfm9-b"],
        help="Override the DFM9 B unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dfm9-c-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dfm9-c"],
        help="Override the DFM9 C unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dfm9-d-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dfm9-d"],
        help="Override the DFM9 D unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--dfm9-generic-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dfm9-ab"],
        help="Override the combined A+B unigram probabilities JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional --limit forwarded to simple_trace.py for every selected run.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override --num-workers for every selected run.",
    )
    parser.add_argument(
        "--docs-per-span",
        type=int,
        default=None,
        help="Override --docs-per-span for every selected run.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("text", "mixed"),
        default=None,
        help="Override --match-mode for every selected run.",
    )
    parser.add_argument(
        "--enable-print",
        action="store_true",
        help="Forward --enable-print to simple_trace.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Deprecated: experiments now continue on errors by default.",
    )
    return parser.parse_args()


def resolve_index_dir(index_key: str, args: argparse.Namespace) -> str | list[str]:
    if index_key == "commonpile":
        return args.commonpile_index_dir
    if index_key == "dynaword":
        return args.dynaword_index_dir
    if index_key == "dfm9-a":
        return args.dfm9_a_index_dir
    if index_key == "dfm9-b":
        return args.dfm9_b_index_dir
    if index_key == "dfm9-c":
        return args.dfm9_c_index_dir
    if index_key == "dfm9-d":
        return args.dfm9_d_index_dir
    if index_key == "dfm9-ab":
        return args.dfm9_generic_index_dirs
    raise ValueError(f"Unknown index key: {index_key}")


def resolve_unigram_path(unigram_key: str, args: argparse.Namespace) -> str:
    if unigram_key == "commonpile":
        return args.commonpile_unigram_probs_path
    if unigram_key == "dynaword":
        return args.dynaword_unigram_probs_path
    if unigram_key == "dfm9-a":
        return args.dfm9_a_unigram_probs_path
    if unigram_key == "dfm9-b":
        return args.dfm9_b_unigram_probs_path
    if unigram_key == "dfm9-c":
        return args.dfm9_c_unigram_probs_path
    if unigram_key == "dfm9-d":
        return args.dfm9_d_unigram_probs_path
    if unigram_key == "dfm9-ab":
        return args.dfm9_generic_unigram_probs_path
    raise ValueError(f"Unknown unigram key: {unigram_key}")


def expand_targets(raw_targets: list[str]) -> list[Experiment]:
    names: list[str] = []
    seen: set[str] = set()

    for raw_target in raw_targets:
        for target in (token.strip() for token in raw_target.split(",") if token.strip()):
            if target in GROUPS:
                expanded_names = GROUPS[target]
            elif target in EXPERIMENTS_BY_NAME:
                expanded_names = [target]
            else:
                valid = ", ".join(sorted({*GROUPS.keys(), *EXPERIMENTS_BY_NAME.keys()}))
                raise SystemExit(f"Unknown target '{target}'. Use --list to inspect choices.\n\nValid targets:\n{valid}")

            for name in expanded_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

    return [EXPERIMENTS_BY_NAME[name] for name in names]


def build_command(experiment: Experiment, args: argparse.Namespace) -> list[str]:
    resolved_index_dirs = resolve_index_dir(experiment.index_key, args)
    index_dirs = (
        [resolved_index_dirs]
        if isinstance(resolved_index_dirs, str)
        else list(resolved_index_dirs)
    )
    unigram_path = resolve_unigram_path(experiment.unigram_key, args)
    num_workers = args.num_workers if args.num_workers is not None else experiment.num_workers
    docs_per_span = args.docs_per_span if args.docs_per_span is not None else experiment.docs_per_span
    match_mode = args.match_mode if args.match_mode is not None else experiment.match_mode

    command = [
        args.python,
        str(SIMPLE_TRACE_PATH),
        "--dataset",
        experiment.dataset,
        experiment.dataset_flag,
        "--index-dir",
        *index_dirs,
        "--unigram-probs-path",
        unigram_path,
        "--num-workers",
        str(num_workers),
        "--docs-per-span",
        str(docs_per_span),
        "--results-output",
        experiment.results_output,
        "--summary-output",
        experiment.summary_output,
        "--length-buckets",
        experiment.length_buckets,
        "--k-eidetic-values",
        experiment.k_eidetic_values,
        "--n-token-span-ratio",
        str(experiment.n_token_span_ratio),
    ]

    if match_mode:
        command.extend(["--match-mode", match_mode])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.enable_print:
        command.append("--enable-print")

    return command


def ensure_inputs_exist(experiment: Experiment, args: argparse.Namespace) -> None:
    dataset_path = REPO_ROOT / experiment.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset for {experiment.name}: {dataset_path}")

    resolved_index_dirs = resolve_index_dir(experiment.index_key, args)
    index_dirs = (
        [resolved_index_dirs]
        if isinstance(resolved_index_dirs, str)
        else list(resolved_index_dirs)
    )
    for raw_index_path in index_dirs:
        index_path = Path(raw_index_path)
        if not index_path.is_absolute():
            index_path = REPO_ROOT / index_path
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing index directory for {experiment.name}: {index_path}"
            )

    unigram_path = Path(resolve_unigram_path(experiment.unigram_key, args))
    if not unigram_path.is_absolute():
        unigram_path = REPO_ROOT / unigram_path
    if not unigram_path.exists():
        raise FileNotFoundError(f"Missing unigram file for {experiment.name}: {unigram_path}")

    for output_path in (experiment.results_output, experiment.summary_output):
        (REPO_ROOT / output_path).parent.mkdir(parents=True, exist_ok=True)


def print_available_targets() -> None:
    print("Groups:")
    for group_name in sorted(GROUPS):
        print(f"  {group_name}")

    print("\nExperiments:")
    for experiment in EXPERIMENTS:
        tags = ", ".join(experiment.tags)
        print(f"  {experiment.name} [{tags}]")


def main() -> int:
    args = parse_args()

    if args.list:
        print_available_targets()
        return 0

    if not args.targets:
        raise SystemExit("No targets provided. Use --list to inspect the available groups and experiment names.")

    experiments = expand_targets(args.targets)
    if not experiments:
        print("No experiments selected.")
        return 0

    print("Selected experiments:")
    for experiment in experiments:
        print(f"  - {experiment.name}")
    print()

    failures: list[tuple[str, int]] = []

    for index, experiment in enumerate(experiments, start=1):
        command = build_command(experiment, args)
        print(f"[{index}/{len(experiments)}] {experiment.name}")
        print(shlex.join(command))

        if args.dry_run:
            print()
            continue

        try:
            ensure_inputs_exist(experiment, args)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Skipping.", file=sys.stderr)
            failures.append((experiment.name, 1))
        except subprocess.CalledProcessError as exc:
            print(
                f"Warning: Experiment {experiment.name} failed with exit code {exc.returncode}. Skipping.",
                file=sys.stderr,
            )
            failures.append((experiment.name, exc.returncode))
        except Exception as exc:
            print(f"Warning: Failed to run {experiment.name}: {exc}. Skipping.", file=sys.stderr)
            failures.append((experiment.name, 1))

        print()

    if failures:
        print("Failed experiments:", file=sys.stderr)
        for name, returncode in failures:
            print(f"  - {name} (exit code {returncode})", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

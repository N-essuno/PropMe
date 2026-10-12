#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "generation"))
from generation_runs import (  # noqa: E402
    SETTINGS,
    Corpus,
    Model,
    generation_path,
    results_path,
    run_name,
    runs,
    setting_tag,
    summary_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLE_TRACE_PATH = REPO_ROOT / "03_tracing" / "simple_trace.py"
# Root of the large data (indexes/, raw/), outside the repository by default.
INDEXES_ROOT = str(Path(os.environ.get("PROPME_DATA_ROOT", REPO_ROOT / "propme_data")) / "indexes")
# Indexes of the DFM9 category experiments.
DFM9_INDEXES_ROOT = str(Path(os.environ.get("PROPME_DFM9_ROOT", REPO_ROOT / "dfm9_data")) / "indexes")

DEFAULT_LENGTH_BUCKETS = "1-3,4-6,7-10,11-20,21-49,50-75,76-99,100-150,151-inf"
DEFAULT_K_EIDETIC_VALUES = "1,5,10"

INDEX_DEFAULTS = {
    "commonpile": f"{INDEXES_ROOT}/commonpile_index/common_pile_train_index",
    "dynaword": f"{INDEXES_ROOT}/dynaword_index",
    "dolma3": f"{INDEXES_ROOT}/dolma3_index_link",
    "dfm9-a": f"{DFM9_INDEXES_ROOT}/A",
    "dfm9-b": f"{DFM9_INDEXES_ROOT}/B",
    "dfm9-c": f"{DFM9_INDEXES_ROOT}/C",
    "dfm9-d": f"{DFM9_INDEXES_ROOT}/D",
    "dfm9-ab": (
        f"{DFM9_INDEXES_ROOT}/A",
        f"{DFM9_INDEXES_ROOT}/B",
    ),
}

UNIGRAM_DEFAULTS = {
    "commonpile": "02_unigram_probs/unigram_probs_common_pile_train.json",
    "dynaword": "02_unigram_probs/unigram_probs_dynaword.json",
    "dolma3": "02_unigram_probs/unigram_probs_dolma3_link.json",
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
    # Forwarded to simple_trace.py when set; otherwise its defaults apply.
    find_threads: int | None = None
    max_page_table_gb: float | None = None


DFM9_EXPERIMENTS = (
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

)


# simple_trace.py --max-page-table-gb per corpus index (per worker; 32 workers).
MAX_PAGE_TABLE_GB = {"dynaword": 10, "commonpile": 10, "dolma3": 4}


def _generation_experiment(model: Model, corpus: Corpus, setting: str) -> Experiment:
    """Trace one setting of a generation run (see generation_runs.py) against the run's corpus index."""
    return Experiment(
        name=f"{run_name(model, corpus)}-{setting_tag(setting)}",
        dataset=generation_path(model, corpus, setting),
        dataset_flag="--is-generation-json",
        index_key=corpus.name,
        unigram_key=corpus.name,
        num_workers=32,
        docs_per_span=10,
        results_output=results_path(model, corpus, setting),
        summary_output=summary_path(model, corpus, setting),
        n_token_span_ratio=50,
        match_mode="mixed",
        find_threads=64,
        max_page_table_gb=MAX_PAGE_TABLE_GB[corpus.name],
        tags=(
            model.family,
            model.key,
            run_name(model, corpus),
            corpus.name,
            "generations",
            setting_tag(setting),
        ),
    )


EXPERIMENTS = DFM9_EXPERIMENTS + tuple(
    _generation_experiment(model, corpus, setting)
    for model, corpus in runs()
    for setting in SETTINGS
)

EXPERIMENTS_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}

# One group per tag: a model family (dfm), model (dfm-main), run (dfm-main-dynaword),
# corpus (dynaword), setting (minimal-cue), or a DFM9 tag.
GROUPS = {
    "all": [experiment.name for experiment in EXPERIMENTS],
    "all-generations": [experiment.name for experiment in EXPERIMENTS if "generations" in experiment.tags],
    "dfm9-generations": [
        experiment.name
        for experiment in EXPERIMENTS
        if "dfm9" in experiment.tags and "generations" in experiment.tags
    ],
    **{
        tag: [experiment.name for experiment in EXPERIMENTS if tag in experiment.tags]
        for tag in dict.fromkeys(tag for experiment in EXPERIMENTS for tag in experiment.tags)
    },
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
        "--dolma3-index-dir",
        default=INDEX_DEFAULTS["dolma3"],
        help="Override the Dolma3 index directory.",
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
        "--dolma3-unigram-probs-path",
        default=UNIGRAM_DEFAULTS["dolma3"],
        help="Override the Dolma3 unigram probabilities JSON path.",
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
        "--shard",
        default=None,
        metavar="K/N",
        help=(
            "Trace only shard K of N of every selected run (every N-th generation, starting with the "
            "K-th), into <output>_shardKofN files. Run the N shards (e.g. on N servers), then --merge-shards N."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="DIR",
        help=(
            "Write results and summaries under DIR instead of memorization_experiment/data, keeping the "
            "layout below it (e.g. DIR/propme/<model>/<corpus>/...), so other runs are not overwritten."
        ),
    )
    parser.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Combine the N --shard runs of every selected run into its usual results and summary "
            "(no tracing; the k-eidetic statistics still query the index)."
        ),
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
        "--find-threads",
        type=int,
        default=None,
        help="Override --find-threads (index query threads per worker) for every selected run.",
    )
    parser.add_argument(
        "--max-page-table-gb",
        type=float,
        default=None,
        help="Override --max-page-table-gb (per-worker page-table cap) for every selected run.",
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
    if index_key == "dolma3":
        return args.dolma3_index_dir
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
    if unigram_key == "dolma3":
        return args.dolma3_unigram_probs_path
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


DATA_PREFIX = "memorization_experiment/data/"


def experiment_outputs(experiment: Experiment, args: argparse.Namespace) -> tuple[str, str]:
    """Results and summary paths of an experiment, moved under --output-root when given."""
    paths = (experiment.results_output, experiment.summary_output)
    if not args.output_root:
        return paths
    root = args.output_root.rstrip("/")
    return tuple(f"{root}/{p.removeprefix(DATA_PREFIX)}" for p in paths)


def shard_output_path(path: str, k: int, n: int) -> str:
    """st_generic_1000_results.json -> st_generic_1000_shard1of4_results.json."""
    base, suffix = path.rsplit("_", 1)
    return f"{base}_shard{k}of{n}_{suffix}"


def parse_shard(raw: str) -> tuple[int, int]:
    try:
        k, n = (int(part) for part in raw.split("/"))
    except ValueError:
        raise SystemExit(f"--shard must look like K/N, got {raw!r}") from None
    if not 1 <= k <= n:
        raise SystemExit(f"--shard needs 1 <= K <= N, got {raw!r}")
    return k, n


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
    find_threads = args.find_threads if args.find_threads is not None else experiment.find_threads
    max_page_table_gb = (
        args.max_page_table_gb if args.max_page_table_gb is not None else experiment.max_page_table_gb
    )
    results_output, summary_output = experiment_outputs(experiment, args)
    if args.shard:
        k, n = parse_shard(args.shard)
        results_output = shard_output_path(results_output, k, n)
        summary_output = shard_output_path(summary_output, k, n)

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
        results_output,
        "--summary-output",
        summary_output,
        "--length-buckets",
        experiment.length_buckets,
        "--k-eidetic-values",
        experiment.k_eidetic_values,
        "--n-token-span-ratio",
        str(experiment.n_token_span_ratio),
    ]

    if match_mode:
        command.extend(["--match-mode", match_mode])
    if find_threads is not None:
        command.extend(["--find-threads", str(find_threads)])
    if max_page_table_gb is not None:
        command.extend(["--max-page-table-gb", f"{max_page_table_gb:g}"])
    if args.shard:
        command.extend(["--shard", args.shard])
    if args.merge_shards:
        command.extend([
            "--merge-results",
            *(shard_output_path(experiment_outputs(experiment, args)[0], k, args.merge_shards)
              for k in range(1, args.merge_shards + 1)),
        ])
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

    if args.merge_shards:
        for k in range(1, args.merge_shards + 1):
            shard_results = REPO_ROOT / shard_output_path(experiment_outputs(experiment, args)[0], k, args.merge_shards)
            if not shard_results.exists():
                raise FileNotFoundError(f"Missing shard results for {experiment.name}: {shard_results}")

    for output_path in experiment_outputs(experiment, args):
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
    if args.shard and args.merge_shards:
        raise SystemExit("--shard and --merge-shards are mutually exclusive")
    if args.shard:
        parse_shard(args.shard)
    if args.merge_shards is not None and args.merge_shards < 1:
        raise SystemExit("--merge-shards must be >= 1")

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

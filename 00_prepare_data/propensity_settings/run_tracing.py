#!/usr/bin/env python3
"""Trace the Tatoeba samples (the generic prompt sets) against the English and Danish indexes.

For each (set, index) pair this runs two passes:

1. SimpleTrace (`03_tracing/simple_trace.py`, mixed mode) for span-level
   overlap: longest matched spans, full-sentence matches in retrieved docs,
   NV recall.
2. A verbatim lookup that counts how often each whole sentence occurs in the
   index (see `verbatim_token_variants`).

Results are aggregated into `tracing_overview.json` and `tracing_overview.md`.
Run from the repository root after build_samples.py:

    python 00_prepare_data/propensity_settings/run_tracing.py --num-workers 8

English sets go against Common Pile and Dolma3, Danish sets against Dynaword
(override with --indexes). Dolma3 defaults to the combined symlinked index;
`--indexes dolma3_split` loads the separate split indexes instead. Pairs whose outputs exist are skipped unless
--overwrite.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]
SIMPLE_TRACE_PATH = REPO_ROOT / "03_tracing" / "simple_trace.py"
TRACES_DIR = DATA_DIR / "traces"
# README and metrics of the prompt sets; the prompt files are in PROMPT_DATA_DIR (see prompt_set_path).
PROMPT_SETS_DIR = DATA_DIR / "prompt_sets"
PROMPT_DATA_DIR = REPO_ROOT / "memorization_experiment" / "data"
# Prompt-set corpus suffix -> dataset folder in PROMPT_DATA_DIR.
PROMPT_SET_DATASETS = {"cp": "commonpile", "d3": "dolma3", "dw": "dynaword2"}
# Prompt-free runs (unconditional, minimal_cue_<lang>).
PROMPT_FREE_DIR = PROMPT_DATA_DIR / "prompt_free"
GENERATION_LOGS_DIR = REPO_ROOT / "logs" / "generation"
TOKENIZER_MODEL = "meta-llama/Llama-2-7b-hf"
NEWLINE_TOKEN = "<0x0A>"

# Root of the large data (indexes/, raw/), outside the repository by default.
DATA_ROOT = Path(os.environ.get("PROPME_DATA_ROOT", REPO_ROOT / "propme_data"))
INDEXES_ROOT = str(DATA_ROOT / "indexes")
# The separate Dolma3 split indexes behind the symlinked dolma3_index_link, as
# listed in scripts/test_simpletrace_dolma3.py. Loaded together they form the
# same corpus, so both Dolma3 entries share one unigram table.
DOLMA3_SPLIT_INDEX_DIRS = (
    *(f"{INDEXES_ROOT}/dolma3_split{i}_index" for i in range(1, 14)),
    f"{INDEXES_ROOT}/dolma3_split13_bis_index",
    *(f"{INDEXES_ROOT}/dolma3_split{i}_index" for i in range(14, 26)),
    f"{INDEXES_ROOT}/dolma3_split26_index_6shards",
)

# name -> (index directories, unigram probs path). Several directories are
# loaded as one logical multi-shard index.
INDEXES = {
    "dynaword": (
        (f"{INDEXES_ROOT}/dynaword_index",),
        "02_unigram_probs/unigram_probs_dynaword.json",
    ),
    # Dynaword 1.2.12 minus depbank, jvj, nordjyllandnews and synne: DFM's
    # training data (build_dynaword1212.py). The
    # `dynaword` index above is a later version with more subsets.
    "dynaword1212": (
        (f"{INDEXES_ROOT}/dynaword1212_index",),
        "02_unigram_probs/unigram_probs_dynaword1212.json",
    ),
    "commonpile": (
        (f"{INDEXES_ROOT}/commonpile_index/common_pile_train_index",),
        "02_unigram_probs/unigram_probs_common_pile_train.json",
    ),
    "dolma3": (
        (f"{INDEXES_ROOT}/dolma3_index_link",),
        "02_unigram_probs/unigram_probs_dolma3_link.json",
    ),
    "dolma3_split": (
        DOLMA3_SPLIT_INDEX_DIRS,
        "02_unigram_probs/unigram_probs_dolma3_link.json",
    ),
}
INDEXES_BY_LANG = {"eng": ("commonpile", "dolma3"), "dan": ("dynaword",)}
SETS = ("tatoeba_eng", "tatoeba_dan")

# Samples are single sentences (~10-50 tokens), so the buckets and the
# n-token-span threshold are finer than the defaults used for generations.
LENGTH_BUCKETS = "1-3,4-6,7-10,11-15,16-20,21-30,31-50,51-inf"
N_TOKEN_SPAN_RATIO = 10
FREQUENT_OCCURRENCES = 10


def prompt_set_path(name: str) -> Path:
    """File of a prompt set in PROMPT_DATA_DIR: generic_dw -> dynaword2/generic/generic_prompts.jsonl
    (Tatoeba, build_prompt_sets.py), specific_dw -> dynaword2/specific/specific_prompts.jsonl
    (unseen-source sentences, build_unseen_prompt_sets.py)."""
    kind, corpus, *rest = name.split("_")
    setting = "_".join([kind, *rest])
    return PROMPT_DATA_DIR / PROMPT_SET_DATASETS[corpus] / setting / f"{setting}_prompts.jsonl"


def generations_path(run: str, model: str) -> Path:
    """A model's generations for a prompt set, next to its prompts
    (generic_dw -> dynaword2/generic/generations/<model>/generic_generations.json),
    or for a prompt-free run (unconditional, minimal_cue_<lang>) in PROMPT_FREE_DIR."""
    if run == "unconditional" or run.startswith("minimal_cue_"):
        folder, stem = PROMPT_FREE_DIR / run, run
    else:
        prompts = prompt_set_path(run)
        folder, stem = prompts.parent, prompts.name.removesuffix("_prompts.jsonl")
    return folder / "generations" / model / f"{stem}_generations.json"


def build_jobs(sets: list[str], indexes: list[str] | None) -> list[tuple[str, str]]:
    jobs = []
    for set_name in sets:
        lang = set_name.rsplit("_", 1)[1]
        for index_name in indexes or INDEXES_BY_LANG[lang]:
            jobs.append((set_name, index_name))
    return jobs


def output_paths(set_name: str, index_name: str) -> dict[str, Path]:
    stem = f"{set_name}_{index_name}"
    return {
        "results": TRACES_DIR / f"st_{stem}_results.jsonl",
        "summary": TRACES_DIR / f"st_{stem}_summary.json",
        "verbatim": TRACES_DIR / f"verbatim_{stem}.jsonl",
    }


def load_set(set_name: str) -> list[dict]:
    with open(DATA_DIR / f"{set_name}_2k.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_simpletrace(set_name: str, index_name: str, args: argparse.Namespace, dataset: Path | None = None) -> None:
    """Trace a set against an index; `dataset` defaults to the set's 2k sample (any JSONL with `text` works)."""
    index_dirs, unigram_path = INDEXES[index_name]
    paths = output_paths(set_name, index_name)
    cmd = [
        sys.executable,
        str(SIMPLE_TRACE_PATH),
        "--dataset", str(dataset or DATA_DIR / f"{set_name}_2k.jsonl"),
        "--is-jsonl",
        "--text-field", "text",
        "--index-dir", *index_dirs,
        "--unigram-probs-path", unigram_path,
        "--num-workers", str(args.num_workers),
        "--find-threads", str(args.find_threads),
        "--max-page-table-gb", str(args.max_page_table_gb),
        "--docs-per-span", str(args.docs_per_span),
        "--match-mode", "mixed",
        "--results-output", str(paths["results"]),
        "--summary-output", str(paths["summary"]),
        "--length-buckets", LENGTH_BUCKETS,
        "--n-token-span-ratio", str(N_TOKEN_SPAN_RATIO),
    ]
    print(f"\n=== SimpleTrace {set_name} vs {index_name} ===\n{' '.join(cmd)}", flush=True)
    start = time.time()
    # Only the summary is useful on stdout; the full doc-id lists are huge.
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f"=== done in {time.time() - start:.0f}s ===", flush=True)


def verbatim_token_variants(text: str, enc) -> list[list[int]]:
    """Token sequences under which `text` can appear verbatim in the index.

    Llama-2's SentencePiece tokenizer marks a word-initial space on the first
    token ("▁Jeg"), which is how the sentence is tokenized at a document start
    or after a space. After a newline, quote or dash the same sentence starts
    without that marker ("J", "eg"), so an exact lookup of the plain encoding
    misses e.g. every subtitle line. The two variants start with different
    tokens, so their occurrence counts are disjoint and can be summed.
    """
    spaced = enc.encode(text, add_special_tokens=False)
    with_newline = enc.encode("\n" + text, add_special_tokens=False)
    newline_id = enc.convert_tokens_to_ids(NEWLINE_TOKEN)
    unspaced = with_newline[with_newline.index(newline_id) + 1 :]
    return [spaced] if unspaced == spaced else [spaced, unspaced]


def run_verbatim(set_name: str, index_name: str, args: argparse.Namespace) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    simple_trace = importlib.import_module("03_tracing.simple_trace")

    enc = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, add_bos_token=False, add_eos_token=False)
    index_dirs = INDEXES[index_name][0]
    engine = simple_trace.PageTableBoundedEngine(
        index_dirs[0] if len(index_dirs) == 1 else list(index_dirs),
        enc.eos_token_id,
        args.max_page_table_gb * 1e9,
    )
    records = load_set(set_name)
    variants = [verbatim_token_variants(record["text"], enc) for record in records]
    # Each lookup waits on index reads, so all (sentence, variant) lookups share
    # one pool: more of them are in flight, including both variants of a sentence.
    lookups = [ids for record_variants in variants for ids in record_variants]

    def count(ids: list[int]) -> int:
        return engine.count(input_ids=ids)["count"]

    print(f"\n=== Verbatim lookup {set_name} vs {index_name} ===", flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.verbatim_threads) as pool:
        counts = list(tqdm(pool.map(count, lookups), total=len(lookups), desc="Verbatim lookup", unit="lookup"))
    rows = []
    offset = 0
    for record, record_variants in zip(records, variants):
        record_counts = counts[offset : offset + len(record_variants)]
        offset += len(record_variants)
        rows.append({
            "id": record["id"],
            "text": record["text"],
            "n_tokens": record["n_tokens"],
            "count_spaced": record_counts[0],
            "count_unspaced": record_counts[1] if len(record_counts) > 1 else 0,
            "count": sum(record_counts),
        })
    with open(output_paths(set_name, index_name)["verbatim"], "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"=== done in {time.time() - start:.0f}s ===", flush=True)


def verbatim_stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    found = [r["count"] for r in rows if r["count"] > 0]
    n = len(rows)
    return {
        "verbatim_found_ratio": round(len(found) / n, 4),
        f"verbatim_found_ge_{FREQUENT_OCCURRENCES}x_ratio": round(
            sum(c >= FREQUENT_OCCURRENCES for c in found) / n, 4
        ),
        "verbatim_found_spaced_only_ratio": round(sum(r["count_spaced"] > 0 for r in rows) / n, 4),
        "verbatim_median_count_when_found": statistics.median(found) if found else 0,
    }


def coverage_stats(set_name: str, results_path: Path) -> dict:
    """Per-sentence fraction of tokens covered by its longest traced span."""
    n_tokens = {r["text"]: r["n_tokens"] for r in load_set(set_name)}
    coverages = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            longest = max((sp["span_length"] for sp in row["spans"]), default=0)
            coverages.append(min(1.0, longest / n_tokens[row["generation"]]))
    return {
        "avg_longest_span_coverage": round(statistics.mean(coverages), 4),
        "longest_span_coverage_ge_0.5_ratio": round(sum(c >= 0.5 for c in coverages) / len(coverages), 4),
    }


def summarize(jobs: list[tuple[str, str]]) -> None:
    length_stats = json.loads((DATA_DIR / "length_stats.json").read_text())
    rows = []
    for set_name, index_name in jobs:
        paths = output_paths(set_name, index_name)
        if not (paths["summary"].exists() and paths["verbatim"].exists()):
            continue
        s = json.loads(paths["summary"].read_text())
        rows.append(
            {
                "set": set_name,
                "index": index_name,
                "num_samples": s["total_generations"],
                "avg_tokens": length_stats[set_name]["avg_tokens"],
                **verbatim_stats(paths["verbatim"]),
                "generations_full_matches_ratio": s["generations_full_matches_ratio"],
                "generations_with_spans_ratio": round(
                    s["generations_with_spans"] / s["total_generations"], 4
                ),
                f"generations_with_{N_TOKEN_SPAN_RATIO}_token_span_ratio": s[
                    "generations_with_n_token_span_ratio"
                ],
                "average_longest_span_length": s["average_longest_span_length"],
                **coverage_stats(set_name, paths["results"]),
                "avg_nv_recall": s["avg_nv_recall"],
                "generations_with_nv_recall_ratio": s["generations_with_nv_recall_ratio"],
                "generations_above_nv_recall_0.5_ratio": s["generations_above_nv_recall_threshold_ratio"],
            }
        )

    with open(DATA_DIR / "tracing_overview.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)
    if not rows:
        return
    # Transposed (one column per set/index pair) so the table stays readable.
    keys = [k for k in rows[0] if k not in ("set", "index")]
    lines = [
        "| metric | " + " | ".join(f"{r['set']} / {r['index']}" for r in rows) + " |",
        "|---|" + "---|" * len(rows),
        *(f"| {k} | " + " | ".join(str(r[k]) for r in rows) + " |" for k in keys),
    ]
    (DATA_DIR / "tracing_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=list(INDEXES),
        default=None,
        help=(
            "Trace every set against these indexes instead of the per-language "
            "defaults. 'dolma3_split' is Dolma3 loaded from its separate split "
            "indexes rather than the combined symlinked one."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--find-threads", type=int, default=4)
    parser.add_argument(
        "--verbatim-threads",
        type=int,
        default=16,
        help="Concurrent whole-sentence lookups in the verbatim pass (one process, so no extra page tables).",
    )
    parser.add_argument("--max-page-table-gb", type=float, default=1.0)
    parser.add_argument("--docs-per-span", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the overview.")
    args = parser.parse_args()

    TRACES_DIR.mkdir(exist_ok=True)
    if not args.summary_only:
        for set_name, index_name in build_jobs(args.sets, args.indexes):
            paths = output_paths(set_name, index_name)
            if args.overwrite or not paths["verbatim"].exists():
                run_verbatim(set_name, index_name, args)
            if args.overwrite or not paths["summary"].exists():
                run_simpletrace(set_name, index_name, args)
    # Overview covers every pair traced so far, not just this invocation's.
    summarize(build_jobs(list(SETS), list(INDEXES)))


if __name__ == "__main__":
    main()

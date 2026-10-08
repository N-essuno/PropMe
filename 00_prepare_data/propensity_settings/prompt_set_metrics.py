#!/usr/bin/env python3
"""Report FMR, ALS and NVR of the generic prompt sets' prompt texts against their index.

The generic_<x> prompts are unmodified sentences of the 2k Tatoeba samples,
which run_tracing.py traced with SimpleTrace. This script takes each
prompt set's sentences from those per-sentence results and recomputes the
summary over them with simple_trace.evaluate_results, so the metrics are
defined exactly as in every other SimpleTrace summary:

    FMR  generations_full_matches_ratio  share of prompts with a retrieved doc containing the whole prompt
    ALS  average_longest_span_length     mean length (Llama-2 tokens) of each prompt's longest traced span
    NVR  avg_nv_recall                   mean near-verbatim recall over retrieved docs (and over docs with
                                         recall > 0: avg_nv_recall_on_hits)

Run from the repository root after run_tracing.py and build_prompt_sets.py:

    python 00_prepare_data/propensity_settings/prompt_set_metrics.py

Writes prompt_sets/prompt_metrics.md and prompt_sets/prompt_metrics.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

import run_tracing as rt


DATA_DIR = Path(__file__).resolve().parent
PROMPT_SETS_DIR = rt.PROMPT_SETS_DIR
INDEX_NAMES = {"commonpile": "Common Pile", "dolma3_split": "Dolma3", "dynaword": "Dynaword"}
# Corpus suffix -> (index, Tatoeba sample), as in build_prompt_sets.py.
CORPORA = {
    "cp": ("commonpile", "tatoeba_eng"),
    "d3": ("dolma3_split", "tatoeba_eng"),
    "dw": ("dynaword", "tatoeba_dan"),
}

if str(rt.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(rt.REPO_ROOT))
simple_trace = importlib.import_module("03_tracing.simple_trace")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def trace_rows(set_name: str, index_name: str) -> list[dict]:
    """Per-text SimpleTrace results of a traced set, in its input order, as evaluate_results values."""
    path = rt.output_paths(set_name, index_name)["results"]
    return [{"generation": row["generation"], "final_spans": row["spans"]} for row in load_jsonl(path)]


def prompt_results(prompts: list[dict], traced: list[dict]) -> dict[int, dict]:
    """evaluate_results input for `prompts`, one entry per prompt, looked up by text in `traced`.

    Keyed by position so repeated prompt texts are each counted, as SimpleTrace does.
    """
    by_text = {row["generation"]: row for row in traced}
    return {i: by_text[p["text"]] for i, p in enumerate(prompts)}


def summarize(results: dict[str, dict]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        summary = simple_trace.evaluate_results(
            results,
            length_buckets=simple_trace.parse_length_buckets(rt.LENGTH_BUCKETS),
            summary_output_path=str(Path(tmp) / "summary.json"),
            nv_recall_threshold=0.5,
            n_token_span_ratio=rt.N_TOKEN_SPAN_RATIO,
        )
    return {
        "num_prompts": summary["total_generations"],
        "FMR": summary["generations_full_matches_ratio"],
        "ALS": summary["average_longest_span_length"],
        "NVR": summary["avg_nv_recall"],
        "NVR_on_hits": summary["avg_nv_recall_on_hits"],
    }


def check_reconstruction(set_name: str, index_name: str, traced: list[dict]) -> None:
    """A traced set's recomputed metrics must equal its saved SimpleTrace summary."""
    saved = json.loads(rt.output_paths(set_name, index_name)["summary"].read_text())
    recomputed = summarize(dict(enumerate(traced)))
    expected = {
        "num_prompts": saved["total_generations"],
        "FMR": saved["generations_full_matches_ratio"],
        "ALS": saved["average_longest_span_length"],
        "NVR": saved["avg_nv_recall"],
        "NVR_on_hits": saved["avg_nv_recall_on_hits"],
    }
    if recomputed != expected:
        raise RuntimeError(f"{set_name} vs {index_name}: recomputed {recomputed} != saved {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    rows = []
    for corpus, (index_name, source) in CORPORA.items():
        name = f"generic_{corpus}"
        prompts = load_jsonl(rt.prompt_set_path(name))
        row = {"prompt_set": name, "index": index_name, "num_prompts": len(prompts)}
        if rt.output_paths(source, index_name)["results"].exists():
            traced = trace_rows(source, index_name)
            check_reconstruction(source, index_name, traced)
            row.update(summarize(prompt_results(prompts, traced)), source=source)
        rows.append(row)

    (PROMPT_SETS_DIR / "prompt_metrics.json").write_text(json.dumps(rows, indent=4), encoding="utf-8")
    lines = [
        "# Prompt-set overlap metrics",
        "",
        "SimpleTrace metrics of the **prompt texts themselves** against the index of each prompt set",
        "(not of model generations). Built with `../prompt_set_metrics.py` from the per-text results of the traced",
        "2k Tatoeba samples in `../traces/` (`mixed` mode, 10 docs per span).",
        "",
        "- **FMR** (`generations_full_matches_ratio`): share of prompts for which a retrieved document contains the whole prompt.",
        "- **ALS** (`average_longest_span_length`): mean length, in Llama-2 tokens, of each prompt's longest traced span.",
        "- **NVR** (`avg_nv_recall`): mean near-verbatim recall over all retrieved documents; NVR on hits averages only documents with recall > 0.",
        "",
        "| prompt set | index | prompts | FMR | ALS (tokens) | NVR | NVR on hits |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "FMR" in r:
            cells = f"{r['FMR']:.4f} | {r['ALS']:.2f} | {r['NVR']:.4f} | {r['NVR_on_hits']:.4f}"
        else:
            cells = "not traced | not traced | not traced | not traced"
        lines.append(f"| {r['prompt_set']} | {INDEX_NAMES[r['index']]} | {r['num_prompts']} | {cells} |")
    lines += [
        "",
        "Notes:",
        "",
        "- `generic_<x>` excludes full matches: prompts whose whole text occurs in the index, found by the token-level",
        "  verbatim lookup or as a document or substring of a document SimpleTrace retrieved (`../group_by_overlap.py`),",
        "  so its FMR is 0.",
        "- ALS here comes from SimpleTrace's traced spans (at least 4 tokens, rarest spans kept), not from the exact longest match",
        "  used for the overlap groups.",
    ]
    (PROMPT_SETS_DIR / "prompt_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

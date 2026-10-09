"""Exact occurrence counts in the training index of the long spans found by tracing.

For every generation with a span of >= 50 tokens (robustness.LONG), counts how often the span's
first 50 tokens and the whole span occur in the training index (infini-gram find(): occurrences,
not documents). For prefix prompts it also counts the 100-token source sequence (the 50-token
prefix and its true 50-token continuation), to relate extraction to duplication (Carlini et al.,
2023; Cooper et al., 2026).

Tokenization is SimpleTrace's: Llama-2, no BOS/EOS, so span token positions index the encoded
generation. Counts are cached next to each results file as <stem>_dups.json and read by
make_propme_report.py, which needs no index access.

    python 05_propensity_metrics/count_duplicates.py --corpus dolma3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "05_propensity_metrics"))
sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
sys.path.insert(0, str(REPO_ROOT / "memorization_experiment"))
sys.path.insert(0, str(REPO_ROOT / "memorization_experiment" / "generation"))

import robustness as rb  # noqa: E402
from generation_runs import CORPORA, MODELS_BY_KEY, SETTINGS, results_path  # noqa: E402

RUNS = {"commonpile": ("comma-2t", "dfm-main"), "dynaword": ("dfm-main",), "dolma3": ("olmo3-32b",)}
WINDOW = rb.LONG  # tokens counted at the start of each long span
DUPS_VERSION = 2
MAX_TRIM = 5  # leading tokens dropped at most when a span does not tokenize as in the matched text


def dups_path(model_key: str, corpus_name: str, setting: str) -> Path:
    results = REPO_ROOT / results_path(MODELS_BY_KEY[model_key], CORPORA[corpus_name], setting)
    return results.with_name(results.name[: -len("_results.json")] + "_dups.json")


def load_dups(model_key: str, corpus_name: str, setting: str) -> dict | None:
    """Cached counts if they exist and match the current results file, else None."""
    path = dups_path(model_key, corpus_name, setting)
    if not path.exists():
        return None
    results = REPO_ROOT / results_path(MODELS_BY_KEY[model_key], CORPORA[corpus_name], setting)
    cached = json.loads(path.read_text())
    source = {"size": results.stat().st_size, "mtime_ns": results.stat().st_mtime_ns}
    return cached if cached.get("source") == source and cached.get("version") == DUPS_VERSION else None


def source_sequences(corpus_name: str, enc) -> dict[int, list[int]]:
    """prompt_id -> token ids of the prefix and its true continuation (2 x 50 tokens)."""
    out = {}
    for prompt_id, (doc, prompt) in rb.prefix_sources(CORPORA[corpus_name]).items():
        ids = enc.encode(doc["text"], add_special_tokens=False)
        if len(ids) >= 2 * rb.TARGET_TOKENS:
            out[prompt_id] = ids[: 2 * rb.TARGET_TOKENS]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, choices=sorted(RUNS))
    parser.add_argument("--index-dir", help="default: the index run_memorization_experiments.py uses")
    parser.add_argument("--max-page-table-gb", type=float, default=8.0)
    parser.add_argument("--force", action="store_true", help="recount even if cached")
    args = parser.parse_args()

    import run_memorization_experiments as rme
    import simple_trace as st
    from transformers import AutoTokenizer

    index_dir = args.index_dir or rme.INDEX_DEFAULTS[args.corpus]
    enc = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
    t0 = time.time()
    engine = st.PageTableBoundedEngine(index_dir, enc.eos_token_id, args.max_page_table_gb * 1e9)
    print(f"opened {index_dir} in {time.time() - t0:.0f}s", flush=True)
    memo: dict[tuple[int, ...], int] = {}

    def count(ids: list[int]) -> int:
        key = tuple(ids)
        if key not in memo:
            memo[key] = int(engine.find(input_ids=list(ids))["cnt"])
        return memo[key]

    def span_counts(ids: list[int], start: int, end: int) -> tuple[int, int] | None:
        """(count of the first 50 tokens, count of the whole span). Spans matched as raw text can
        start or end mid-word (e.g. a generation continuing its prompt's last word) and then tokenize
        differently than in the matched document, so up to MAX_TRIM tokens are dropped at either end,
        fewest first, until the span is found."""
        for trim in range(2 * MAX_TRIM + 1):
            for front in range(max(0, trim - MAX_TRIM), min(trim, MAX_TRIM) + 1):
                lo, hi = start + front, end - (trim - front)
                if hi - lo < WINDOW:
                    continue
                span = count(ids[lo:hi])
                if span:
                    return count(ids[lo: lo + WINDOW]), span
        return None

    sources = None
    for model_key in RUNS[args.corpus]:
        for setting in SETTINGS:
            if not args.force and load_dups(model_key, args.corpus, setting) is not None:
                print(f"{model_key} {setting}: cached", flush=True)
                continue
            results = REPO_ROOT / results_path(MODELS_BY_KEY[model_key], CORPORA[args.corpus], setting)
            rows = {k: [] for k in ("prompt_id", "sample_idx", "longest_window", "longest_span", "min_window")}
            t0, queried, unmatched = time.time(), 0, 0
            with open(results) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    rows["prompt_id"].append(r["prompt_id"])
                    rows["sample_idx"].append(r["sample_idx"])
                    long = [s for s in r["spans"] if s["end"] - s["start"] >= WINDOW]
                    if not long:
                        for k in ("longest_window", "longest_span", "min_window"):
                            rows[k].append(None)
                        continue
                    ids = enc.encode(r["generation"])
                    pairs = [(s, span_counts(ids, s["start"], s["end"])) for s in long]
                    if any(c is None for _, c in pairs):
                        unmatched += 1
                        pairs = [(s, c) for s, c in pairs if c is not None]
                        if not pairs:
                            for k in ("longest_window", "longest_span", "min_window"):
                                rows[k].append(None)
                            continue
                    long, counts = [s for s, _ in pairs], [c for _, c in pairs]
                    windows = [c[0] for c in counts]
                    best = max(range(len(long)), key=lambda i: long[i]["end"] - long[i]["start"])
                    span = counts[best][1]
                    rows["longest_window"].append(windows[best])
                    rows["longest_span"].append(span)
                    rows["min_window"].append(min(windows))
                    queried += 1
            out = {"source": {"size": results.stat().st_size, "mtime_ns": results.stat().st_mtime_ns},
                   "version": DUPS_VERSION, "index_dir": str(index_dir), "unmatched": unmatched, **rows}
            if setting == "prefix":
                sources = sources if sources is not None else {p: count(ids) for p, ids in source_sequences(args.corpus, enc).items()}
                out["source_count"] = {str(p): c for p, c in sources.items()}
            dups_path(model_key, args.corpus, setting).write_text(json.dumps(out))
            print(f"{model_key} {setting}: {queried} generations with long spans, {unmatched} with an unmatched span, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

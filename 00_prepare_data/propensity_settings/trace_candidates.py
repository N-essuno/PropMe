#!/usr/bin/env python3
"""Verbatim filter of the candidate sentences.

Same logic as the generic sets (run_tracing.py, group_by_overlap.py),
run on candidates_<corpus>.jsonl against the index of the corpus's training
data (sources.CORPUS_INDEX: cp -> commonpile, d3 -> dolma3_split,
dw -> dynaword1212):

    groups  SimpleTrace (mixed mode, 10 docs per span), then group_by_overlap's
            groups: occurrences over both tokenizations, SimpleTrace substring
            matches, longest matched span. Groups 1-5 are not full matches.
    extra   Verbatim counts in sources.CORPUS_EXTRA_INDEXES (Dynaword
            candidates in Common Pile, which DFM also saw); any occurrence
            counts as a full match.

The candidates already have Tatoeba lengths (build_candidates.py), so there
are no cut versions.

Writes traces/ (SimpleTrace results), groups/groups_unseen_<corpus>_<index>.jsonl
and work/traced_<corpus>.jsonl: every candidate with its group, overlap
measures and extra-index occurrences.

Run from the repository root:

    python 00_prepare_data/propensity_settings/trace_candidates.py --corpora dw cp d3 --num-workers 12 --find-threads 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

import group_by_overlap as gbo
import run_tracing as rt
import sources as src

# Groups that are not full matches (as in build_prompt_sets.KEPT_GROUPS).
KEPT_GROUPS = ("1", "3", "4", "5")
TRACES_DIR = rt.TRACES_DIR
GROUPS_DIR = gbo.GROUPS_DIR
WORK_DIR = src.DATA_DIR / "work"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def simpletrace(name: str, index_name: str, records: list[dict], args: argparse.Namespace) -> None:
    """SimpleTrace `records` (deduplicated texts) unless its results exist."""
    if rt.output_paths(name, index_name)["summary"].exists() and not args.overwrite:
        print(f"SimpleTrace {name} vs {index_name}: results exist")
        return
    dataset = WORK_DIR / f"trace_input_{name}.jsonl"
    texts = list(dict.fromkeys(r["text"] for r in records))
    write_jsonl(dataset, [{"text": t} for t in texts])
    rt.run_simpletrace(name, index_name, args, dataset=dataset)


def group(name: str, index_name: str, records: list[dict], args: argparse.Namespace, enc) -> dict[str, dict]:
    """Group rows by text (group_by_overlap.group_records), cached in groups/."""
    path = GROUPS_DIR / f"groups_{name}_{index_name}.jsonl"
    if path.exists() and not args.overwrite:
        rows = read_jsonl(path)
    else:
        rows = gbo.group_records(name, index_name, records, args, enc, None,
                                 gbo.traced_substrings(name, index_name))
    return {r["text"]: r for r in rows}


def extra_counts(texts: list[str], index_name: str, args: argparse.Namespace, enc) -> dict[str, int]:
    engine = gbo.open_engine(index_name, enc, args.max_page_table_gb)
    unique = list(dict.fromkeys(texts))
    with ThreadPoolExecutor(args.threads) as pool:
        counts = list(tqdm(pool.map(lambda t: gbo.count_occurrences(t, engine, enc), unique), total=len(unique),
                           desc=f"Verbatim in {index_name}", unit="text"))
    return dict(zip(unique, counts))


def trace_corpus(corpus: str, args: argparse.Namespace, enc) -> None:
    index_name = src.CORPUS_INDEX[corpus]
    candidates = read_jsonl(src.DATA_DIR / f"candidates_{corpus}.jsonl")
    name = f"unseen_{corpus}"
    print(f"\n##### {corpus}: {len(candidates)} candidates vs {index_name} #####", flush=True)

    simpletrace(name, index_name, candidates, args)
    groups = group(name, index_name, candidates, args, enc)
    extra = {index: extra_counts([c["text"] for c in candidates], index, args, enc)
             for index in src.CORPUS_EXTRA_INDEXES[corpus]}
    for c in candidates:
        g = groups[c["text"]]
        c.update({"group": g["group"], "occurrences": g["occurrences"], "coverage_tokens": g["coverage_tokens"],
                  "coverage_chars": g["coverage_chars"]})
        c["extra_occurrences"] = {index: counts[c["text"]] for index, counts in extra.items()}
    write_jsonl(WORK_DIR / f"traced_{corpus}.jsonl", candidates)
    kept = sum(c["group"] in KEPT_GROUPS and not any(c["extra_occurrences"].values()) for c in candidates)
    print(f"{corpus}: {kept}/{len(candidates)} candidates are not full matches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", choices=list(src.CORPUS_SOURCES), default=list(src.CORPUS_SOURCES))
    parser.add_argument("--num-workers", type=int, default=12, help="SimpleTrace processes.")
    parser.add_argument("--find-threads", type=int, default=4, help="Lookup threads per SimpleTrace process.")
    parser.add_argument("--threads", type=int, default=32, help="Concurrent texts when grouping.")
    parser.add_argument("--max-page-table-gb", type=float, default=2.0)
    parser.add_argument("--docs-per-span", type=int, default=10)
    parser.add_argument("--k", type=int, default=100, help="Distinct-document threshold between groups 6 and 7.")
    parser.add_argument("--exact-doc-counts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for d in (TRACES_DIR, GROUPS_DIR, WORK_DIR):
        d.mkdir(exist_ok=True)
    enc = AutoTokenizer.from_pretrained(rt.TOKENIZER_MODEL, add_bos_token=False, add_eos_token=False)
    for corpus in args.corpora:
        trace_corpus(corpus, args, enc)


if __name__ == "__main__":
    main()

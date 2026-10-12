#!/usr/bin/env python3
"""Group the Tatoeba samples by how much of each sentence occurs in an index.

For every (set, index) pair that has a verbatim lookup (see run_tracing.py),
each sentence is assigned to one group:

    1  longest matched span < 25% of the sentence
    3  longest matched span 25-50% (50 excluded)
    4  longest matched span 50-75% (75 excluded)
    5  longest matched span >= 75%, but the full sentence does not occur
    6  full sentence occurs, in more than k distinct documents
    7  full sentence occurs, in at most k distinct documents

"Full sentence occurs" (a full match) means the whole text occurs in the
index: as a token sequence, found by the verbatim lookup (count > 0) under
either tokenization (see run_tracing.verbatim_token_variants), or as a
document or substring of a document that SimpleTrace retrieved for it (its
exact_full_raw match tier). The second catches occurrences tokenized
differently at the edges, e.g. a final "?" merged with a following quote into
'?"'. `full_match_source` says which found it; for SimpleTrace-only matches
`distinct_docs` counts the retrieved documents containing the text, a lower
bound. The SimpleTrace check needs traces/st_<set>_<index>_results.jsonl.

The longest matched span is recomputed exactly here rather than read from the
SimpleTrace results: those keep only spans of >= 4 tokens and the rarest
ceil(5% of length) of them, so they can miss the longest one. It is the
longest token span, starting at any position, that occurs verbatim in the
index (SimpleTrace's unfiltered step 1), taken over both tokenizations, each
divided by its own token count. Grouping uses this token coverage; the
character coverage of the same span is stored too.

Distinct documents are counted only for full matches, stopping once more
than k are found, so `distinct_docs` is capped at k + 1.

Run from the repository root:

    python 00_prepare_data/propensity_settings/group_by_overlap.py --threads 64

Writes groups/groups_<set>_<index>.jsonl and groups/groups_overview.{json,md}.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

import run_tracing as rt


GROUPS_DIR = rt.DATA_DIR / "groups"
GROUP_LABELS = {
    "1": "longest span < 25%",
    "3": "longest span 25-50%",
    "4": "longest span 50-75%",
    "5": "longest span >= 75%, not full",
    "6": "full match, > k docs",
    "7": "full match, <= k docs",
}
DOC_FETCH_BATCH = 128

if str(rt.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(rt.REPO_ROOT))
simple_trace = importlib.import_module("03_tracing.simple_trace")


def coverage_group(coverage: float) -> str:
    if coverage < 0.25:
        return "1"
    if coverage < 0.5:
        return "3"
    if coverage < 0.75:
        return "4"
    return "5"


def longest_match(text: str, engine, enc) -> dict:
    """Longest verbatim-indexed token span of `text`, over both tokenizations."""
    best = None
    for variant, ids in zip(("spaced", "unspaced"), rt.verbatim_token_variants(text, enc)):
        lens = simple_trace.longest_prefix_lens(engine, ids, range(len(ids)), len(ids) * 5)
        start = max(range(len(ids)), key=lambda b: lens[b])
        coverage = lens[start] / len(ids)
        if best is None or coverage > best["coverage_tokens"]:
            span_text = enc.decode(ids[start : start + lens[start]])
            best = {
                "longest_span_tokens": lens[start],
                "longest_span_start": start,
                "longest_span_variant": variant,
                "longest_span_text": span_text,
                "coverage_tokens": round(coverage, 4),
                "coverage_chars": round(min(1.0, len(span_text) / len(text)), 4),
            }
    return best


def count_distinct_docs(text: str, engine, enc, cap: int) -> int:
    """Distinct documents containing `text` (either tokenization), counted up to `cap`."""
    seen: set[tuple[int, int]] = set()
    for ids in rt.verbatim_token_variants(text, enc):
        find_res = engine.find(input_ids=ids)
        if find_res.get("cnt", 0) == 0:
            continue
        ranks = simple_trace._flatten_ranks(find_res)
        for batch_start in range(0, len(ranks), DOC_FETCH_BATCH):
            batch = ranks[batch_start : batch_start + DOC_FETCH_BATCH]
            for (shard, _), doc in zip(batch, simple_trace._get_docs_by_ranks(engine, batch, 1)):
                seen.add((shard, doc["doc_ix"]))
            if len(seen) >= cap:
                return cap
    return len(seen)


def open_engine(index_name: str, enc, max_page_table_gb: float):
    index_dirs = rt.INDEXES[index_name][0]
    return simple_trace.PageTableBoundedEngine(
        index_dirs[0] if len(index_dirs) == 1 else list(index_dirs),
        enc.eos_token_id,
        max_page_table_gb * 1e9,
    )


def count_occurrences(text: str, engine, enc) -> int:
    """Verbatim occurrences of `text` in the index, over both tokenizations."""
    return sum(engine.count(input_ids=ids)["count"] for ids in rt.verbatim_token_variants(text, enc))


def simpletrace_full_match_docs(name: str, index_name: str) -> dict[str, int] | None:
    """Text -> retrieved documents containing the whole text, from SimpleTrace results (None if not traced)."""
    path = rt.output_paths(name, index_name)["results"]
    if not path.exists():
        return None
    docs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for row in map(json.loads, f):
            ids = {
                doc["id"]
                for span in row["spans"]
                for doc in span["docs"]
                if doc["match_tier"] == simple_trace.FULL_RAW_MATCH_TIER
            }
            docs[row["generation"]] = len(ids)
    return docs


def overlap_fields(
    text: str,
    n_tokens: int,
    occurrences: int,
    engine,
    enc,
    k: int,
    exact_doc_counts: bool = False,
    substring_docs: int = 0,
) -> dict:
    """Group of `text` and the overlap measures it is based on.

    `substring_docs` is the number of documents SimpleTrace retrieved that
    contain the whole text; it makes a full match even when the verbatim
    lookup found no occurrence.
    """
    if occurrences == 0 and substring_docs > 0:
        return {
            "group": "6" if substring_docs > k else "7",
            "full_match": True,
            "full_match_source": "simpletrace",
            "distinct_docs": substring_docs,
            "longest_span_tokens": n_tokens,
            "coverage_tokens": 1.0,
            "coverage_chars": 1.0,
        }
    if occurrences > 0:
        # Occurrences bound distinct docs from above, so only texts occurring
        # more than k times can be in more than k docs.
        distinct = (
            count_distinct_docs(text, engine, enc, k + 1)
            if occurrences > k or exact_doc_counts
            else None
        )
        return {
            "group": "6" if distinct is not None and distinct > k else "7",
            "full_match": True,
            "full_match_source": "verbatim_lookup",
            "distinct_docs": distinct,
            "longest_span_tokens": n_tokens,
            "coverage_tokens": 1.0,
            "coverage_chars": 1.0,
        }
    match = longest_match(text, engine, enc)
    return {"group": coverage_group(match["coverage_tokens"]), "full_match": False, "distinct_docs": None, **match}


def group_pair(set_name: str, index_name: str, args: argparse.Namespace, enc) -> list[dict]:
    with open(rt.output_paths(set_name, index_name)["verbatim"], encoding="utf-8") as f:
        verbatim = {row["text"]: row["count"] for row in map(json.loads, f)}
    return group_records(
        set_name, index_name, rt.load_set(set_name), args, enc, verbatim, traced_substrings(set_name, index_name)
    )


def traced_substrings(name: str, index_name: str) -> dict[str, int]:
    docs = simpletrace_full_match_docs(name, index_name)
    if docs is None:
        print(f"[WARN] {name} vs {index_name} has no SimpleTrace results: full matches come from the verbatim lookup only")
        return {}
    return docs


def group_records(
    name: str,
    index_name: str,
    records: list[dict],
    args: argparse.Namespace,
    enc,
    occurrences_by_text: dict[str, int] | None = None,
    substring_docs_by_text: dict[str, int] | None = None,
) -> list[dict]:
    """Group `records` (with id, text, n_tokens) against an index and write groups_<name>_<index>.jsonl.

    Occurrences come from `occurrences_by_text` (the verbatim lookup) when
    given, otherwise they are counted here. `substring_docs_by_text` holds
    SimpleTrace's full-match document counts (see overlap_fields).
    """
    out_path = GROUPS_DIR / f"groups_{name}_{index_name}.jsonl"
    engine = open_engine(index_name, enc, args.max_page_table_gb)

    def assign(record: dict) -> dict:
        occurrences = (
            occurrences_by_text[record["text"]]
            if occurrences_by_text is not None
            else count_occurrences(record["text"], engine, enc)
        )
        return {
            "id": record["id"],
            "text": record["text"],
            "n_tokens": record["n_tokens"],
            "occurrences": occurrences,
            **overlap_fields(
                record["text"],
                record["n_tokens"],
                occurrences,
                engine,
                enc,
                args.k,
                args.exact_doc_counts,
                (substring_docs_by_text or {}).get(record["text"], 0),
            ),
        }

    print(f"\n=== Grouping {name} vs {index_name} ===", flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        rows = list(tqdm(pool.map(assign, records), total=len(records), desc="Grouping", unit="sent"))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"=== done in {time.time() - start:.0f}s ===", flush=True)
    return rows


def summarize(k: int) -> None:
    columns = []
    for set_name, index_name in rt.build_jobs(list(rt.SETS), list(rt.INDEXES)):
        path = GROUPS_DIR / f"groups_{set_name}_{index_name}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            groups = [json.loads(line)["group"] for line in f]
        counts = {g: groups.count(g) for g in GROUP_LABELS}
        columns.append({"set": set_name, "index": index_name, "num_samples": len(groups), "counts": counts})

    overview = {"k": k, "group_labels": GROUP_LABELS, "pairs": columns}
    (GROUPS_DIR / "groups_overview.json").write_text(json.dumps(overview, indent=4), encoding="utf-8")
    if not columns:
        return
    lines = [
        f"k = {k}\n",
        "| group | " + " | ".join(f"{c['set']} / {c['index']}" for c in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for g, label in GROUP_LABELS.items():
        cells = (f"{c['counts'][g]} ({100 * c['counts'][g] / c['num_samples']:.1f}%)" for c in columns)
        lines.append(f"| {g}: {label} | " + " | ".join(cells) + " |")
    (GROUPS_DIR / "groups_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=rt.SETS, default=list(rt.SETS))
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=list(rt.INDEXES),
        default=None,
        help="Indexes to group against (default: every index with a verbatim lookup for the set).",
    )
    parser.add_argument("--k", type=int, default=100, help="Distinct-document threshold between groups 6 and 7.")
    parser.add_argument(
        "--exact-doc-counts",
        action="store_true",
        help="Count distinct docs (capped at k + 1) for every full match, not only those occurring more than k times.",
    )
    parser.add_argument("--threads", type=int, default=64, help="Sentences processed concurrently (one process).")
    parser.add_argument("--max-page-table-gb", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    GROUPS_DIR.mkdir(exist_ok=True)
    if not args.summary_only:
        enc = AutoTokenizer.from_pretrained(rt.TOKENIZER_MODEL, add_bos_token=False, add_eos_token=False)
    if not args.summary_only:
        # By default, every pair that already has a verbatim lookup.
        for set_name, index_name in rt.build_jobs(args.sets, args.indexes or list(rt.INDEXES)):
            if not rt.output_paths(set_name, index_name)["verbatim"].exists():
                if args.indexes:
                    print(f"Skipping {set_name} vs {index_name}: no verbatim lookup yet (run run_tracing.py)")
                continue
            if (GROUPS_DIR / f"groups_{set_name}_{index_name}.jsonl").exists() and not args.overwrite:
                print(f"Skipping {set_name} vs {index_name}: groups exist")
                continue
            group_pair(set_name, index_name, args, enc)
    summarize(args.k)


if __name__ == "__main__":
    main()

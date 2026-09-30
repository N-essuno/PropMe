#!/usr/bin/env python3
"""Smoke-test SimpleTrace queries against the Dolma3 indexes.

Run from any directory with the project's InfiniGram and Transformers
environment active:

    python scripts/test_simpletrace_dolma3.py

By default, use all Dolma3 split indexes in the index directory. Use --single
to query the combined symlink-based index instead, or pass --index-dir to
select specific indexes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer
from infini_gram.engine import InfiniGramEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = Path("/work/pecora/propme_data/indexes")
DEFAULT_INDEX_DIRS = [
    *(str(DEFAULT_INDEX_ROOT / f"dolma3_split{i}_index") for i in range(1, 13)),
    str(DEFAULT_INDEX_ROOT / "dolma3_split13_index"),
    str(DEFAULT_INDEX_ROOT / "dolma3_split13_bis_index"),
    *(str(DEFAULT_INDEX_ROOT / f"dolma3_split{i}_index") for i in range(14, 26)),
    str(DEFAULT_INDEX_ROOT / "dolma3_split26_index_6shards"),
]
DEFAULT_SINGLE_INDEX = DEFAULT_INDEX_ROOT / "dolma3_index_link"
DEFAULT_QUERY = "Hello world."
DEFAULT_UNIGRAM_PROBS_PATH = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dolma3_link.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        nargs="+",
        default=DEFAULT_INDEX_DIRS,
        help="InfiniGram index directories (defaults to all Dolma3 split indexes).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=f"Use the combined symlink index at {DEFAULT_SINGLE_INDEX}.",
    )
    query_source = parser.add_mutually_exclusive_group()
    query_source.add_argument("--query", help="One query to trace.")
    query_source.add_argument(
        "--query-file",
        type=Path,
        help="Text file with one query per line; blank lines are skipped.",
    )
    parser.add_argument(
        "--unigram-probs-path",
        type=Path,
        default=DEFAULT_UNIGRAM_PROBS_PATH,
        help="JSON file containing token unigram probabilities.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("text", "mixed"),
        default="mixed",
        help="SimpleTrace matching mode (default: mixed).",
    )
    parser.add_argument(
        "--docs-per-span", type=int, default=3,
        help="Maximum retrieved documents per matched span.",
    )
    args = parser.parse_args()

    if args.query_file:
        if not args.query_file.is_file():
            parser.error(f"Query file not found: {args.query_file}")
        queries = args.query_file.read_text().splitlines()
        queries = [query for query in queries if query.strip()]
        if not queries:
            parser.error(f"Query file contains no non-empty queries: {args.query_file}")
    else:
        queries = [args.query if args.query is not None else DEFAULT_QUERY]

    index_dirs = [str(DEFAULT_SINGLE_INDEX)] if args.single else args.index_dir
    missing = [path for path in index_dirs if not Path(path).is_dir()]
    if missing:
        parser.error("Index directory not found: " + ", ".join(missing))
    if not args.unigram_probs_path.is_file():
        parser.error(f"Unigram probabilities file not found: {args.unigram_probs_path}")

    with args.unigram_probs_path.open() as f:
        unigram_probs = {
            int(token_id): entry["prob"]
            for token_id, entry in json.load(f).items()
        }

    # Import the repository's tracing function without requiring scripts to be
    # run from the repository root or installing PropMe as a package.
    sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
    from simple_trace import trace_generation  # noqa: E402

    print(f"Loading Llama tokenizer and {len(index_dirs)} index directories...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(
        index_dir=index_dirs,
        eos_token_id=tokenizer.eos_token_id,
        precompute_unigram_logprobs=False,
    )

    print(f"Loaded {engine.engine.get_num_shards()} shards from {len(index_dirs)} indexes.")
    for query_number, query in enumerate(queries, start=1):
        print(f"\nTracing query {query_number}/{len(queries)}: {query!r}", flush=True)
        result = trace_generation(
            query,
            engine,
            tokenizer,
            unigram_probs=unigram_probs,
            docs_per_span=args.docs_per_span,
            match_mode=args.match_mode,
        )

        spans = result["final_spans"]
        print(f"Query tokens: {len(result['gen_ids'])}; traced spans: {len(spans)}")
        for span in spans:
            print(f"\n[{span['start']}:{span['end']}] {span['text']!r}")
            print(f"Retrieved documents: {len(span['docs'])}")
            for doc in span["docs"]:
                print(f"  id={doc.get('id', '')!r} match_tier={doc.get('match_tier', '')}")


if __name__ == "__main__":
    main()

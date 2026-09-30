#!/usr/bin/env python3
"""Smoke-test one SimpleTrace query against three partial Dolma3 indexes.

Run from any directory with the project's InfiniGram and Transformers
environment active:

    python scripts/test_simpletrace_dolma3.py

Pass different partial index directories with --index-dir if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from transformers import AutoTokenizer
from infini_gram.engine import InfiniGramEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = Path("/work/pecora/propme_data/indexes")
DEFAULT_INDEX_DIRS = [
    str(DEFAULT_INDEX_ROOT / f"dolma3_split{i}_index") for i in (1, 2, 3)
]
DEFAULT_QUERY = "Hello world."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        nargs="+",
        default=DEFAULT_INDEX_DIRS,
        help="Three (or more) InfiniGram partial index directories.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text to trace.")
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

    if len(args.index_dir) < 3:
        parser.error("provide at least three partial index directories")
    missing = [path for path in args.index_dir if not Path(path).is_dir()]
    if missing:
        parser.error("Index directory not found: " + ", ".join(missing))

    # Import the repository's tracing function without requiring scripts to be
    # run from the repository root or installing PropMe as a package.
    sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
    from simple_trace import trace_generation  # noqa: E402

    print(f"Loading Llama tokenizer and {len(args.index_dir)} index directories...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(
        index_dir=args.index_dir,
        eos_token_id=tokenizer.eos_token_id,
        precompute_unigram_logprobs=False,
    )

    print(f"Tracing query: {args.query!r}", flush=True)
    result = trace_generation(
        args.query,
        engine,
        tokenizer,
        unigram_probs={},  # Neutral scores; this smoke test checks index queries.
        docs_per_span=args.docs_per_span,
        match_mode=args.match_mode,
    )

    spans = result["final_spans"]
    print(f"Loaded {engine.engine.get_num_shards()} shards from {len(args.index_dir)} indexes.")
    print(f"Query tokens: {len(result['gen_ids'])}; traced spans: {len(spans)}")
    for span in spans:
        print(f"\n[{span['start']}:{span['end']}] {span['text']!r}")
        print(f"Retrieved documents: {len(span['docs'])}")
        for doc in span["docs"]:
            print(f"  id={doc.get('id', '')!r} match_tier={doc.get('match_tier', '')}")


if __name__ == "__main__":
    main()

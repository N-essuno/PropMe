#!/usr/bin/env python3
"""Compute one unigram-probability file over the Dolma3 partial indexes.

By default this combines split1 through split25, both split13 variants, and
split26_index_6shards. The three-shard split26_index directory is omitted
because it appears to be an alternate layout of the same data. Override
--index-dir if your set of partial indexes differs.

Run from any directory with infini-gram installed:

    python scripts/compute_dolma3_unigrams.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
# Root of the large data (indexes/, raw/), outside the repository by default.
DEFAULT_INDEX_ROOT = Path(os.environ.get("PROPME_DATA_ROOT", REPO_ROOT / "propme_data")) / "indexes"
DEFAULT_INDEX_DIRS = [
    *(str(DEFAULT_INDEX_ROOT / f"dolma3_split{i}_index") for i in range(1, 13)),
    str(DEFAULT_INDEX_ROOT / "dolma3_split13_index"),
    str(DEFAULT_INDEX_ROOT / "dolma3_split13_bis_index"),
    *(str(DEFAULT_INDEX_ROOT / f"dolma3_split{i}_index") for i in range(14, 26)),
    str(DEFAULT_INDEX_ROOT / "dolma3_split26_index_6shards"),
]
DEFAULT_OUTPUT = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dolma3.json"
COMPUTE_SCRIPT = REPO_ROOT / "02_unigram_probs" / "compute_unigrams.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        nargs="+",
        default=DEFAULT_INDEX_DIRS,
        help="Partial InfiniGram indexes to aggregate (defaults to Dolma3 splits).",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path for the combined unigram probabilities.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer used to decode token IDs in the output.",
    )
    parser.add_argument("--example-token", default="a")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    missing = [path for path in args.index_dir if not Path(path).is_dir()]
    if missing:
        parser.error("Index directory not found: " + ", ".join(missing))
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    command = [
        sys.executable,
        str(COMPUTE_SCRIPT),
        "--index-dir",
        *args.index_dir,
        "--output-path",
        args.output_path,
        "--tokenizer-model",
        args.tokenizer_model,
        "--example-token",
        args.example_token,
        "--top-k",
        str(args.top_k),
    ]
    print(f"Combining {len(args.index_dir)} index directories:", flush=True)
    for index_dir in args.index_dir:
        print(f"  {index_dir}", flush=True)
    print(f"Output: {args.output_path}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

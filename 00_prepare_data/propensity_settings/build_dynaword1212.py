#!/usr/bin/env python3
"""Rebuild the Dynaword index on the data DFM was trained on.

DFM saw Danish Dynaword @ 9e230b35 (v1.2.12) minus depbank, jvj,
nordjyllandnews and synne. The existing `dynaword_index` is an unpinned,
later download that already contains the subsets added after 1.2.12, so it
cannot tell unseen text from seen text. Steps:

    export   1.2.12 parquet files (download_sources.py dynaword1212) -> one
             JSONL with `text` plus `id`, `source`, `added`, `created`
             metadata, as in the existing index (the source is what
             embedding_similarity's parse_source reads).
    index    python -m infini_gram.indexing --tokenizer llama --add_metadata
    unigram  02_unigram_probs/compute_unigrams.py (SimpleTrace needs it).

Run from the repository root:

    python 00_prepare_data/propensity_settings/build_dynaword1212.py export index unigram --cpus 40 --mem 220
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

import sources as src

REPO_ROOT = src.DATA_DIR.parents[1]
PARQUET_DIR = src.RAW_DIR / "dynaword_1212" / "data"
JSONL_DIR = src.RAW_DIR / "dynaword_1212_jsonl"
DOC_COUNTS_PATH = src.RAW_DIR / "dynaword_1212_doc_counts.json"
INDEX_DIR = src.DATA_ROOT / "indexes" / "dynaword1212_index"
UNIGRAM_PATH = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dynaword1212.json"
METADATA_FIELDS = ("id", "source", "added", "created")


def export() -> None:
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JSONL_DIR / "dynaword_1212.jsonl"
    # At 1.2.12 each subset's text is data/<s>/<s>.parquet.
    files = {p.parent.name: p for p in PARQUET_DIR.glob("*/*.parquet") if p.name != "metadata.parquet"}
    if set(files) & set(src.DFM_EXCLUDED):
        raise RuntimeError(f"Excluded subsets present in {PARQUET_DIR}")
    counts = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for subset in sorted(files):
            table = pq.read_table(files[subset])
            fields = [c for c in ("text", *METADATA_FIELDS) if c in table.column_names]
            n = 0
            for batch in table.select(fields).to_batches(max_chunksize=10_000):
                for row in batch.to_pylist():
                    if not row["text"]:
                        continue
                    row.setdefault("source", subset)
                    f.write(json.dumps({k: (str(v) if v is not None and k != "text" else v)
                                        for k, v in row.items()}, ensure_ascii=False) + "\n")
                    n += 1
            counts[subset] = n
            print(f"  {subset}: {n:,} docs", flush=True)
    # Outside JSONL_DIR: infini_gram.indexing reads every file there as data.
    DOC_COUNTS_PATH.write_text(json.dumps(counts, indent=2))
    print(f"{sum(counts.values()):,} docs from {len(counts)} subsets -> {out_path}")


def index(args) -> None:
    cmd = [
        sys.executable, "-m", "infini_gram.indexing",
        "--data_dir", str(JSONL_DIR),
        "--save_dir", str(INDEX_DIR),
        "--tokenizer", "llama",
        "--cpus", str(args.cpus),
        "--mem", str(args.mem),
        "--shards", str(args.shards),
        "--add_metadata",
        "--ulimit", "1048576",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def unigram() -> None:
    cmd = [
        sys.executable, str(REPO_ROOT / "02_unigram_probs" / "compute_unigrams.py"),
        "--index-dir", str(INDEX_DIR),
        "--output-path", str(UNIGRAM_PATH),
        "--tokenizer-model", "meta-llama/Llama-2-7b-hf",
        "--example-token", "a",
        "--top-k", "10",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("export", "index", "unigram"))
    parser.add_argument("--cpus", type=int, default=40)
    parser.add_argument("--mem", type=int, default=220, help="GiB for infini_gram.indexing.")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    for step in args.steps:
        print(f"=== {step} ===", flush=True)
        if step == "export":
            export()
        elif step == "index":
            index(args)
        else:
            unigram()


if __name__ == "__main__":
    main()

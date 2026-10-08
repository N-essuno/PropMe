#!/usr/bin/env python3
"""Smoke-test SimpleTrace queries against the Dolma3 indexes.

Run from any directory with infini-gram and transformers installed:

    python scripts/test_simpletrace_dolma3.py

By default, use all Dolma3 split indexes in the index directory. Use --single
to query the combined symlink-based index instead, or pass --index-dir to
select specific indexes. Use --num-workers and --find-threads to trace
queries in parallel worker processes, as in 03_tracing/simple_trace.py.
Use --drop-index-cache to evict the index files from the OS page cache first,
so runs comparing these settings all start from a cold cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


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
DEFAULT_SINGLE_INDEX = DEFAULT_INDEX_ROOT / "dolma3_index_link"
DEFAULT_QUERY = "Hello world."
DEFAULT_UNIGRAM_PROBS_PATH = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dolma3_link.json"
INDEX_FILE_PREFIXES = ("table", "tokenized", "offset", "metadata", "metaoff")


def drop_index_cache(index_dirs: list[str]) -> None:
    """Evict the index files' pages from the OS page cache.

    Repeated runs over the same queries read the same index pages, which
    otherwise stay cached and make later runs look much faster. Unlike
    /proc/sys/vm/drop_caches this needs no root. Symlinks are followed, so
    the combined --single index evicts the split indexes' files. Pages still
    mapped by a running process are kept.
    """
    start = time.time()
    num_files = 0
    for index_dir in index_dirs:
        for name in os.listdir(index_dir):
            if name.split(".")[0] not in INDEX_FILE_PREFIXES:
                continue
            fd = os.open(os.path.join(index_dir, name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            num_files += 1
    print(f"Evicted {num_files} index files from the page cache in {time.time() - start:.1f}s.")


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
        "--docs-per-span", type=int, default=10,
        help="Maximum retrieved documents per matched span.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of SimpleTrace worker processes (default: 1).",
    )
    parser.add_argument(
        "--find-threads", type=int, default=1,
        help="Threads per worker running index queries concurrently (default: 1).",
    )
    parser.add_argument(
        "--drop-index-cache",
        action="store_true",
        help="Evict the index files from the OS page cache before tracing, "
             "for timing comparisons on a cold cache.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.find_threads < 1:
        parser.error("--find-threads must be >= 1")

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

    # Import the repository's tracing function without requiring scripts to be
    # run from the repository root or installing PropMe as a package.
    sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
    from simple_trace import launch_simpletrace  # noqa: E402

    if args.drop_index_cache:
        drop_index_cache(index_dirs)

    # Each worker process loads its own Llama tokenizer and index engine.
    print(
        f"Tracing {len(queries)} queries over {len(index_dirs)} index directories "
        f"with {args.num_workers} workers and {args.find_threads} find threads each...",
        flush=True,
    )
    results = launch_simpletrace(
        index_dirs,
        queries,
        str(args.unigram_probs_path),
        num_workers=args.num_workers,
        docs_per_span=args.docs_per_span,
        match_mode=args.match_mode,
        find_threads=args.find_threads,
    )

    for query_number, query in enumerate(queries, start=1):
        print(f"\nQuery {query_number}/{len(queries)}: {query!r}")
        result = results.get(query)
        if result is None:
            print("Not traced (interrupted).")
            continue

        spans = result["final_spans"]
        print(f"Query tokens: {len(result['gen_ids'])}; traced spans: {len(spans)}")
        for span in spans:
            print(f"\n[{span['start']}:{span['end']}] {span['text']!r}")
            print(f"Retrieved documents: {len(span['docs'])}")
            for doc in span["docs"]:
                print(f"  id={doc.get('id', '')!r} match_tier={doc.get('match_tier', '')}")


if __name__ == "__main__":
    main()

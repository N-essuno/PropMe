"""Sample Dolma3 documents and write 50-token prefix prompts.

The default index is the combined, symlink-backed Dolma3 InfiniGram index.
Pass --split-indexes to load the separate partial index directories instead.
Run from the repository root, for example:

    python memorization_experiment/extract_prefixes/dolma3_extract_prefixes.py
"""

from __future__ import annotations

from pathlib import Path

from prefix_extraction import INDEXES_ROOT, REPO_ROOT, build_parser, extract_prefixes, validate_args


DEFAULT_INDEX_DIR = INDEXES_ROOT / "dolma3_index_link"
DEFAULT_INDEXES_ROOT = DEFAULT_INDEX_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "memorization_experiment/data/dolma3"


def split_index_dirs(root: Path) -> list[Path]:
    """Return the partial indexes represented by the combined linked index."""
    return [
        *(root / f"dolma3_split{i}_index" for i in range(1, 14)),
        root / "dolma3_split13_bis_index",
        *(root / f"dolma3_split{i}_index" for i in range(14, 26)),
        root / "dolma3_split26_index_6shards",
    ]


def main() -> None:
    parser = build_parser(__doc__, DEFAULT_OUTPUT_DIR)
    index_choice = parser.add_mutually_exclusive_group()
    index_choice.add_argument(
        "--index-dir",
        type=Path,
        nargs="+",
        help="One linked index directory or one or more separate index directories.",
    )
    index_choice.add_argument(
        "--split-indexes",
        action="store_true",
        help="Load all Dolma3 partial indexes under --indexes-root.",
    )
    parser.add_argument("--indexes-root", type=Path, default=DEFAULT_INDEXES_ROOT)
    args = parser.parse_args()

    index_dirs = (
        split_index_dirs(args.indexes_root)
        if args.split_indexes
        else args.index_dir or [DEFAULT_INDEX_DIR]
    )
    validate_args(parser, args, index_dirs)
    extract_prefixes("dolma3", index_dirs, args)


if __name__ == "__main__":
    main()

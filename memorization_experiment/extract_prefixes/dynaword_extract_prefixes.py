"""Sample Dynaword documents and write 50-token prefix prompts.

Documents from the sources in EXCLUDED_SOURCES are never sampled.
Run from the repository root, for example:

    python memorization_experiment/extract_prefixes/dynaword_extract_prefixes.py
"""

from __future__ import annotations

from pathlib import Path

from prefix_extraction import INDEXES_ROOT, REPO_ROOT, build_parser, extract_prefixes, validate_args


DEFAULT_INDEX_DIR = INDEXES_ROOT / "dynaword_index"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "memorization_experiment/data/dynaword"
EXCLUDED_SOURCES = (
    "kb_administrative_publication",
    "kb_historical_letters",
    "municipality_meetings",
    "hvadvilduhelst",
    "tidsskrift-dk",
)


def main() -> None:
    parser = build_parser(__doc__, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, nargs="+", default=[DEFAULT_INDEX_DIR])
    parser.add_argument(
        "--exclude-sources",
        nargs="*",
        default=list(EXCLUDED_SOURCES),
        help="Metadata sources to skip while sampling (pass no values to disable).",
    )
    args = parser.parse_args()
    validate_args(parser, args, args.index_dir)
    extract_prefixes("dynaword", args.index_dir, args, exclude_sources=args.exclude_sources)


if __name__ == "__main__":
    main()

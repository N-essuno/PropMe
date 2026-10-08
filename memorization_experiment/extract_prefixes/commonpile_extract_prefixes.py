"""Sample Common Pile documents and write 50-token prefix prompts.

Run from the repository root, for example:

    python memorization_experiment/extract_prefixes/commonpile_extract_prefixes.py
"""

from __future__ import annotations

from pathlib import Path

from prefix_extraction import INDEXES_ROOT, REPO_ROOT, build_parser, extract_prefixes, validate_args


DEFAULT_INDEX_DIR = INDEXES_ROOT / "commonpile_index" / "common_pile_train_index"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "memorization_experiment/data/commonpile"


def main() -> None:
    parser = build_parser(__doc__, DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, nargs="+", default=[DEFAULT_INDEX_DIR])
    args = parser.parse_args()
    validate_args(parser, args, args.index_dir)
    extract_prefixes("commonpile", args.index_dir, args)


if __name__ == "__main__":
    main()

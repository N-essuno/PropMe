"""Randomized SimpleTrace validation on the Dolma3 index.

Same protocol and outputs as validation_full_commonpile.py: full-document and
start/middle/end partial queries traced in mixed mode, plus "rescued" partial
queries whose source document id is not retrieved but whose exact text is
(useful for a heavily deduplicated web corpus). Only the defaults differ.

python 04_validation/validation_full_dolma3.py \
    --num-samples 20 \
    --docs-per-span 10
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

commonpile_validation = importlib.import_module("04_validation.validation_full_commonpile")

DOLMA3_INDEX_DIR = str(commonpile_validation.INDEXES_ROOT / "dolma3_index_link")
DOLMA3_UNIGRAM_PROBS_PATH = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dolma3_link.json"


def build_arg_parser() -> argparse.ArgumentParser:
    return commonpile_validation.build_arg_parser(
        corpus_name="Dolma3",
        default_index_dir=DOLMA3_INDEX_DIR,
        default_unigram_probs_path=str(DOLMA3_UNIGRAM_PROBS_PATH),
        output_suffix="dolma3",
    )


def main() -> None:
    commonpile_validation.main(build_arg_parser())


if __name__ == "__main__":
    main()

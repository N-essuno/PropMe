"""Print the number of documents in each DFM9 Infini-gram index."""

from __future__ import annotations

import argparse
from pathlib import Path

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer


MODEL_ID = "meta-llama/Llama-2-7b-hf"
RISK_CATEGORIES = ("A", "B", "C", "D")
DEFAULT_INDEXES_ROOT = Path("/work/olmotrace/mimir_propme/indexes")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print document counts for the DFM9 A/B/C/D indexes."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=RISK_CATEGORIES,
        default=list(RISK_CATEGORIES),
        help="One or more DFM9 risk categories to count (default: A B C D).",
    )
    parser.add_argument(
        "--indexes-root",
        type=Path,
        default=DEFAULT_INDEXES_ROOT,
        help="Directory containing category index directories A, B, C, and D.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=MODEL_ID,
        help="Tokenizer used to initialize the Infini-gram engine.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    categories = list(dict.fromkeys(args.categories))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_bos_token=False,
        add_eos_token=False,
    )

    total_docs = 0
    for category in categories:
        index_dir = args.indexes_root / category
        if not index_dir.is_dir():
            raise FileNotFoundError(f"Index directory does not exist: {index_dir}")

        engine = InfiniGramEngine(
            index_dir=str(index_dir),
            eos_token_id=tokenizer.eos_token_id,
            precompute_unigram_logprobs=False,
        )
        category_docs = engine.engine.get_total_doc_cnt()
        print(f"{category}: {category_docs}")
        total_docs += category_docs

    print(f"total: {total_docs}")


if __name__ == "__main__":
    main()

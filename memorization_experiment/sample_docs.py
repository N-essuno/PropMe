"""
Sample random documents from an InfiniGram index and save them as JSONL.

Example:
python sample_docs.py \
    --index-dir 00_data/dynaword_index/ \
    --output-path memorization_experiment/data/dynaword_sample_docs.jsonl \
    --num-docs 100 \
    --min-tokens 100 \
    --tokenizer-model meta-llama/Llama-2-7b-hf
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
from typing import Any

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    def _parse_optional_int(value: str) -> int | None:
        if value.lower() == "none":
            return None
        parsed = int(value)
        return parsed

    parser = argparse.ArgumentParser(
        description="Sample random distinct documents from an InfiniGram index."
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Path to the InfiniGram index directory.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of random distinct documents to sample.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face tokenizer model name.",
    )
    parser.add_argument(
        "--max-disp-len",
        type=_parse_optional_int,
        default=None,
        help="Maximum number of tokens to decode per sampled document. Use None for no limit (default).",
    )
    parser.add_argument(
        "--min-tokens",
        type=_parse_optional_int,
        default=None,
        help="Only sample documents with at least this many tokens. Use None for no minimum (default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser


def parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    """Parse InfiniGram metadata payload into a plain dict when possible."""
    if isinstance(raw_metadata, dict):
        # Handle both {"metadata": {...}} and already-flat metadata dicts.
        if isinstance(raw_metadata.get("metadata"), dict):
            return raw_metadata["metadata"]
        return raw_metadata

    if isinstance(raw_metadata, str):
        # Metadata is often a Python literal string produced by indexing.
        try:
            parsed = ast.literal_eval(raw_metadata)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("metadata"), dict):
                    return parsed["metadata"]
                return parsed
        except (ValueError, SyntaxError):
            # Fall back to JSON parse if possible.
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("metadata"), dict):
                        return parsed["metadata"]
                    return parsed
            except json.JSONDecodeError:
                pass

    return {}


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.num_docs < 1:
        raise ValueError("--num-docs must be >= 1")
    if args.max_disp_len is not None and args.max_disp_len < 1:
        raise ValueError("--max-disp-len must be >= 1")
    if args.min_tokens is not None and args.min_tokens < 1:
        raise ValueError("--min-tokens must be >= 1")

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(
        index_dir=args.index_dir,
        eos_token_id=tokenizer.eos_token_id,
        precompute_unigram_logprobs=False,
    )

    total_doc_cnt = engine.engine.get_total_doc_cnt()
    if total_doc_cnt < 1:
        raise RuntimeError("Index contains no documents.")

    sample_size = min(args.num_docs, total_doc_cnt)
    if args.min_tokens is None:
        sampled_doc_ixs = random.sample(range(total_doc_cnt), k=sample_size)
    else:
        eligible_doc_ixs: list[int] = []
        candidate_ixs = list(range(total_doc_cnt))
        random.shuffle(candidate_ixs)
        for doc_ix in candidate_ixs:
            doc = engine.get_doc_by_ix(doc_ix=doc_ix)
            doc_len = doc.get("doc_len")
            if isinstance(doc_len, int) and doc_len >= args.min_tokens:
                eligible_doc_ixs.append(doc_ix)
                if len(eligible_doc_ixs) >= args.num_docs:
                    break

        if len(eligible_doc_ixs) < args.num_docs:
            raise RuntimeError(
                "Found only "
                f"{len(eligible_doc_ixs)} docs with --min-tokens >= {args.min_tokens} "
                f"(requested {args.num_docs})."
            )

        sampled_doc_ixs = eligible_doc_ixs[: args.num_docs]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for doc_ix in sampled_doc_ixs:
            doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=args.max_disp_len)
            metadata = parse_metadata(doc.get("metadata"))
            record = {
                "doc_ix": doc_ix,
                "doc_len": doc.get("doc_len"),
                "disp_len": doc.get("disp_len"),
                "needle_offset": doc.get("needle_offset"),
                "text": tokenizer.decode(doc.get("token_ids", [])),
                "metadata": metadata,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Saved {sample_size} sampled documents (out of {total_doc_cnt}) to {args.output_path}"
    )


if __name__ == "__main__":
    main()

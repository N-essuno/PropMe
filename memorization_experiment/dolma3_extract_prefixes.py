"""Sample Dolma3 documents and write 50-token prefix prompts.

The default index is the combined, symlink-backed Dolma3 InfiniGram index.
Pass --split-indexes to load the separate partial index directories instead.
Run from the repository root, for example:

    python memorization_experiment/dolma3_extract_prefixes.py --num-docs 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

from sample_docs import parse_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = Path("/work/pecora/propme_data/indexes/dolma3_index_link")
DEFAULT_INDEXES_ROOT = DEFAULT_INDEX_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "memorization_experiment/data/dolma3"
MODEL_ID = "meta-llama/Llama-2-7b-hf"


def split_index_dirs(root: Path) -> list[Path]:
    """Return the partial indexes represented by the combined linked index."""
    return [
        *(root / f"dolma3_split{i}_index" for i in range(1, 14)),
        root / "dolma3_split13_bis_index",
        *(root / f"dolma3_split{i}_index" for i in range(14, 26)),
        root / "dolma3_split26_index_6shards",
    ]


def shuffled_doc_indices(total: int, seed: int):
    """Yield distinct, uniformly shuffled document IDs using only sampled-ID memory."""
    rng = random.Random(seed)
    swaps: dict[int, int] = {}
    for remaining in range(total, 0, -1):
        position = rng.randrange(remaining)
        doc_ix = swaps.get(position, position)
        last = swaps.pop(remaining - 1, remaining - 1)
        if position != remaining - 1:
            swaps[position] = last
        yield doc_ix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-docs", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--prefix-tokens", type=int, default=50)
    parser.add_argument("--tokenizer-model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_docs < 1 or args.min_tokens < 1 or args.prefix_tokens < 1:
        parser.error("--num-docs, --min-tokens and --prefix-tokens must be positive")
    index_dirs = (
        split_index_dirs(args.indexes_root)
        if args.split_indexes
        else args.index_dir or [DEFAULT_INDEX_DIR]
    )
    for index_dir in index_dirs:
        if not index_dir.is_dir():
            parser.error(f"Index directory not found: {index_dir}")
        broken_link = next(
            (path for path in index_dir.iterdir() if path.is_symlink() and not path.exists()),
            None,
        )
        if broken_link is not None:
            parser.error(
                f"Index contains a broken symlink: {broken_link} -> {broken_link.readlink()}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, add_bos_token=False, add_eos_token=False
    )
    engine = InfiniGramEngine(
        index_dir=str(index_dirs[0]) if len(index_dirs) == 1 else [str(p) for p in index_dirs],
        eos_token_id=tokenizer.eos_token_id,
        precompute_unigram_logprobs=False,
    )
    total = engine.engine.get_total_doc_cnt()
    if args.num_docs > total:
        parser.error(f"Requested {args.num_docs} documents, but the index has {total}")

    sample_path = args.output_dir / "dolma3_sample_docs.jsonl"
    prefix_path = args.output_dir / "prefix/dolma3_prefix_prompts.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    sample_temp = sample_path.with_suffix(".jsonl.tmp")
    prefix_temp = prefix_path.with_suffix(".jsonl.tmp")

    count = 0
    try:
        with sample_temp.open("w", encoding="utf-8") as sample_file, prefix_temp.open(
            "w", encoding="utf-8"
        ) as prefix_file:
            for doc_ix in shuffled_doc_indices(total, args.seed):
                doc = engine.get_doc_by_ix(doc_ix=doc_ix)
                doc_len = doc.get("doc_len")
                if not isinstance(doc_len, int) or doc_len < args.min_tokens:
                    continue

                text = tokenizer.decode(doc.get("token_ids", []))
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) < args.prefix_tokens:
                    continue

                metadata = parse_metadata(doc.get("metadata"))
                source = metadata.get("source")
                domain = source if isinstance(source, str) and source.strip() else "unknown"
                sample = {
                    "doc_ix": doc_ix,
                    "doc_len": doc_len,
                    "disp_len": doc.get("disp_len"),
                    "needle_offset": doc.get("needle_offset"),
                    "text": text,
                    "metadata": metadata,
                }
                prompt = {
                    "text": tokenizer.decode(token_ids[: args.prefix_tokens], skip_special_tokens=True),
                    "domain": domain,
                }
                sample_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                prefix_file.write(json.dumps(prompt, ensure_ascii=False) + "\n")
                count += 1
                if count == args.num_docs:
                    break

        if count != args.num_docs:
            raise RuntimeError(
                f"Found only {count} documents with at least {args.min_tokens} tokens "
                f"and a {args.prefix_tokens}-token prefix (requested {args.num_docs})"
            )
        sample_temp.replace(sample_path)
        prefix_temp.replace(prefix_path)
    finally:
        sample_temp.unlink(missing_ok=True)
        prefix_temp.unlink(missing_ok=True)

    print(f"Saved {count} sampled documents to {sample_path}")
    print(f"Saved {count} prefix prompts to {prefix_path}")


if __name__ == "__main__":
    main()

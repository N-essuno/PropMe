"""Shared sampling and prefix extraction for the dataset-specific prefix scripts.

Documents are sampled directly from InfiniGram indexes, so no separate
`sample_docs.py` step is needed. Each run writes `<name>_sample_docs.jsonl`
and `prefix/<name>_prefix_prompts.jsonl` under the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

from sample_docs import parse_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
# Root of the large data (indexes/, raw/), outside the repository by default.
INDEXES_ROOT = Path(os.environ.get("PROPME_DATA_ROOT", REPO_ROOT / "propme_data")) / "indexes"
MODEL_ID = "meta-llama/Llama-2-7b-hf"
DEFAULT_NUM_DOCS = 2000
DEFAULT_DOMAIN = "unknown"


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


def build_parser(description: str | None, default_output_dir: Path) -> argparse.ArgumentParser:
    """Return a parser with the sampling options shared by all datasets."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--prefix-tokens", type=int, default=50)
    parser.add_argument("--tokenizer-model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, index_dirs: list[Path]) -> None:
    if args.num_docs < 1 or args.min_tokens < 1 or args.prefix_tokens < 1:
        parser.error("--num-docs, --min-tokens and --prefix-tokens must be positive")
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


def extract_prefixes(
    name: str,
    index_dirs: list[Path],
    args: argparse.Namespace,
    exclude_sources: Iterable[str] = (),
) -> None:
    """Sample `args.num_docs` documents and write sample docs plus prefix prompts.

    Documents whose metadata `source` is in `exclude_sources` are skipped during
    sampling, so the requested count is still reached when sources are excluded.
    """
    excluded = frozenset(exclude_sources)
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
        raise ValueError(f"Requested {args.num_docs} documents, but the index has {total}")

    sample_path = args.output_dir / f"{name}_sample_docs.jsonl"
    prefix_path = args.output_dir / f"prefix/{name}_prefix_prompts.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    sample_temp = sample_path.with_suffix(".jsonl.tmp")
    prefix_temp = prefix_path.with_suffix(".jsonl.tmp")

    count = 0
    skipped_sources: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    try:
        with sample_temp.open("w", encoding="utf-8") as sample_file, prefix_temp.open(
            "w", encoding="utf-8"
        ) as prefix_file:
            for doc_ix in shuffled_doc_indices(total, args.seed):
                doc = engine.get_doc_by_ix(doc_ix=doc_ix)
                doc_len = doc.get("doc_len")
                if not isinstance(doc_len, int) or doc_len < args.min_tokens:
                    continue

                metadata = parse_metadata(doc.get("metadata"))
                source = metadata.get("source")
                domain = source if isinstance(source, str) and source.strip() else DEFAULT_DOMAIN
                if domain in excluded:
                    skipped_sources[domain] += 1
                    continue

                text = tokenizer.decode(doc.get("token_ids", []))
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) < args.prefix_tokens:
                    continue

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
                domains[domain] += 1
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
    if excluded:
        print(f"Excluded sources {sorted(excluded)}; skipped {dict(skipped_sources)}")
    print(f"Domain counts: {dict(domains.most_common())}")

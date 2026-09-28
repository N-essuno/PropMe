"""Sample DFM9 index documents and write category-specific prefix prompts.

The existing Common Pile and Dynaword prefix scripts consume a sampled-document
JSONL and emit ``{"text": ..., "domain": ...}`` prompts.  DFM9 consists of
separate A/B/C/D indexes, so this command performs the small sampling step
directly and writes one prompt file per category and prefix length.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
from transformers import AutoTokenizer


MODEL_ID = "meta-llama/Llama-2-7b-hf"
RISK_CATEGORIES = ("A", "B", "C", "D")
DEFAULT_INDEXES_ROOT = Path("/work/olmotrace/mimir_propme/indexes")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "memorization_experiment" / "data" / "dfm9" / "prefix"
)


@dataclass(frozen=True)
class SampledDocument:
    doc_ix: int
    token_ids: tuple[int, ...]
    metadata: dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample documents from separate DFM9 A/B/C/D Infini-gram indexes "
            "and write 50/75/100-token prefix prompt files."
        )
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=RISK_CATEGORIES,
        default=list(RISK_CATEGORIES),
        help="One or more DFM9 risk categories to process (default: A B C D).",
    )
    parser.add_argument(
        "--indexes-root",
        type=Path,
        default=DEFAULT_INDEXES_ROOT,
        help="Directory containing category index directories A, B, C, and D.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the separate prefix JSONL files.",
    )
    parser.add_argument(
        "--prefix-lengths",
        nargs="+",
        type=int,
        default=[50, 75, 100],
        help="Prefix lengths in Llama tokens (default: 50 75 100).",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=100,
        help="Number of distinct source documents to sample per category.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help=(
            "Additional minimum document length. The effective minimum is "
            "always at least the longest requested prefix."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Each category gets a stable derived seed.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=MODEL_ID,
        help="Hugging Face tokenizer used by the DFM9 indexes.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.categories = list(dict.fromkeys(args.categories))
    args.prefix_lengths = sorted(set(args.prefix_lengths))
    if args.num_docs < 1:
        raise ValueError("--num-docs must be >= 1")
    if not args.prefix_lengths or any(length < 1 for length in args.prefix_lengths):
        raise ValueError("--prefix-lengths must contain positive integers")
    if args.min_tokens is not None and args.min_tokens < 1:
        raise ValueError("--min-tokens must be >= 1")


def parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    """Parse the metadata representation returned by Infini-gram 2.x."""
    parsed: Any = raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_metadata)
            except (ValueError, SyntaxError):
                return {}
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get("metadata"), dict):
        return parsed["metadata"]
    return parsed


def metadata_domain(metadata: Mapping[str, Any], category: str) -> str:
    """Apply the existing source-as-domain behavior to nested DFM9 metadata."""
    source_metadata = metadata.get("source_metadata")
    if isinstance(source_metadata, Mapping):
        for key in ("source", "dataset", "subset"):
            value = source_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    value = metadata.get("source")
    if isinstance(value, str) and value.strip():
        return value
    cohort = metadata.get("cohort")
    if isinstance(cohort, str) and cohort.strip():
        return cohort
    return category


def category_seed(base_seed: int, category: str) -> int:
    """Derive a stable seed independent of the selected-category order."""
    return base_seed * 257 + ord(category)


def permuted_indices(total: int, rng: random.Random) -> Iterator[int]:
    """Visit every index once in pseudorandom cyclic order using O(1) memory."""
    if total < 1:
        return
    start = rng.randrange(total)
    if total == 1:
        yield 0
        return
    step = rng.randrange(1, total)
    while math.gcd(step, total) != 1:
        step = rng.randrange(1, total)
    for offset in range(total):
        yield (start + offset * step) % total


def sample_documents(
    engine: InfiniGramEngine,
    tokenizer: Any,
    *,
    category: str,
    num_docs: int,
    minimum_tokens: int,
    seed: int,
) -> list[SampledDocument]:
    total_docs = engine.engine.get_total_doc_cnt()
    if total_docs < 1:
        raise RuntimeError(f"DFM9 {category} index contains no documents")

    sampled: list[SampledDocument] = []
    rng = random.Random(category_seed(seed, category))
    progress = tqdm(
        total=num_docs,
        desc=f"DFM9 {category}: eligible documents",
        unit="doc",
    )
    try:
        for doc_ix in permuted_indices(total_docs, rng):
            document = engine.get_doc_by_ix(
                doc_ix=doc_ix,
                max_disp_len=minimum_tokens,
            )
            doc_len = document.get("doc_len")
            if not isinstance(doc_len, int) or doc_len < minimum_tokens:
                continue

            displayed_ids = document.get("token_ids")
            if not isinstance(displayed_ids, (list, tuple)):
                continue
            # Match the other extraction scripts: decode sampled text and then
            # tokenize it without special tokens before taking the prefix.
            sampled_text = tokenizer.decode(
                displayed_ids,
                skip_special_tokens=True,
            )
            token_ids = tokenizer.encode(sampled_text, add_special_tokens=False)
            if len(token_ids) < minimum_tokens:
                continue

            sampled.append(
                SampledDocument(
                    doc_ix=doc_ix,
                    token_ids=tuple(token_ids),
                    metadata=parse_metadata(document.get("metadata")),
                )
            )
            progress.update(1)
            if len(sampled) == num_docs:
                break
    finally:
        progress.close()

    if len(sampled) != num_docs:
        raise RuntimeError(
            f"DFM9 {category}: found only {len(sampled):,} documents with at "
            f"least {minimum_tokens:,} usable tokens; requested {num_docs:,}"
        )
    return sampled


def output_path(output_dir: Path, category: str, prefix_length: int) -> Path:
    return output_dir / f"dfm9_{category}_prefix_{prefix_length}_prompts.jsonl"


def write_prefix_files(
    sampled: Sequence[SampledDocument],
    tokenizer: Any,
    *,
    category: str,
    prefix_lengths: Sequence[int],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_path(output_dir, category, length) for length in prefix_lengths]
    temp_paths = [
        path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in final_paths
    ]
    handles = [path.open("w", encoding="utf-8") for path in temp_paths]
    try:
        for document in sampled:
            domain = metadata_domain(document.metadata, category)
            for length, handle in zip(prefix_lengths, handles, strict=True):
                prefix_text = tokenizer.decode(
                    document.token_ids[:length],
                    skip_special_tokens=True,
                )
                handle.write(
                    json.dumps(
                        {"text": prefix_text, "domain": domain},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    except Exception:
        for handle in handles:
            handle.close()
        for path in temp_paths:
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles:
            handle.close()
        for temp_path, final_path in zip(temp_paths, final_paths, strict=True):
            os.replace(temp_path, final_path)
    return final_paths


def load_tokenizer(model_id: str) -> Any:
    return AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        add_bos_token=False,
        add_eos_token=False,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    indexes_root = args.indexes_root.resolve()
    output_dir = args.output_dir.resolve()
    minimum_tokens = max(
        max(args.prefix_lengths),
        args.min_tokens or 0,
    )

    missing = [
        str(indexes_root / category)
        for category in args.categories
        if not (indexes_root / category).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing selected DFM9 index directories: " + ", ".join(missing)
        )

    tokenizer = load_tokenizer(args.tokenizer_model)
    for category in args.categories:
        index_dir = indexes_root / category
        print(f"Loading DFM9 {category} index from {index_dir}", flush=True)
        engine = InfiniGramEngine(
            index_dir=str(index_dir),
            eos_token_id=tokenizer.eos_token_id,
            precompute_unigram_logprobs=False,
        )
        sampled = sample_documents(
            engine,
            tokenizer,
            category=category,
            num_docs=args.num_docs,
            minimum_tokens=minimum_tokens,
            seed=args.seed,
        )
        paths = write_prefix_files(
            sampled,
            tokenizer,
            category=category,
            prefix_lengths=args.prefix_lengths,
            output_dir=output_dir,
        )
        for path in paths:
            print(f"Saved {len(sampled):,} prompts to {path}", flush=True)


if __name__ == "__main__":
    main()

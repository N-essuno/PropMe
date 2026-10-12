"""
Example run

python 02_unigram_probs/compute_unigrams.py \
    --index-dir 00_prepare_data/dummy_index \
    --output-path 02_unigram_probs/unigram_probs_dummy.json \
    --tokenizer-model meta-llama/Llama-2-7b-hf \
    --example-token a \
    --top-k 10
"""

import argparse
from collections.abc import Mapping
from datetime import datetime
import math
import json
import os
import sys
import time

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from tqdm import tqdm


def log(message: str) -> None:
    """Print a timestamped status line that stays readable beside tqdm."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    tqdm.write(f"[{timestamp}] {message}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute unigram probabilities from an InfiniGram index."
    )
    parser.add_argument(
        "--index-dir",
        nargs="+",
        required=True,
        help=(
            "Path to one or more InfiniGram index directories. Multiple paths "
            "are combined as one logical multi-shard index."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination JSON file for unigram probabilities.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face tokenizer model name.",
    )
    parser.add_argument(
        "--example-token",
        default="a",
        help="Token string to print example probability for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-probability tokens to print.",
    )
    return parser


def main() -> None:
    run_start = time.perf_counter()
    args = build_arg_parser().parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    log(f"Starting unigram calculation; index directories: {args.index_dir}")
    log(f"Output path: {args.output_path}")

    phase_start = time.perf_counter()
    log(f"Phase 1/6: loading tokenizer {args.tokenizer_model!r}.")
    enc = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_bos_token=False,
        add_eos_token=False,
    )
    log(f"Phase 1/6 complete: tokenizer loaded in {time.perf_counter() - phase_start:.1f}s.")

    index_dir = args.index_dir[0] if len(args.index_dir) == 1 else args.index_dir
    phase_start = time.perf_counter()
    log("Phase 2/6: opening InfiniGram index; this may take a while for a large index.")
    engine = InfiniGramEngine(
        index_dir=index_dir,
        eos_token_id=enc.eos_token_id,
        precompute_unigram_logprobs=True,
    )
    log(f"Phase 2/6 complete: index opened in {time.perf_counter() - phase_start:.1f}s.")

    # Compute unigram counts per shard, then aggregate. Weight tqdm by the
    # indexed token count rather than treating differently sized shards as
    # equal units; this gives a more representative ETA across the corpus.
    phase_start = time.perf_counter()
    log("Phase 3/6: reading shard count and token counts.")
    num_shards = engine.engine.get_num_shards()
    shard_token_counts = [
        int(engine.engine.get_tok_cnt(s=s)) for s in range(num_shards)
    ]
    total_index_tokens = sum(shard_token_counts)
    log(
        f"Phase 3/6 complete in {time.perf_counter() - phase_start:.1f}s: "
        f"found {num_shards} shards and {total_index_tokens:,} indexed tokens."
    )
    if total_index_tokens < 1:
        raise RuntimeError("No tokens found in index; cannot compute unigram probabilities.")

    unigram_counts: dict[int, int] = {}
    with tqdm(
        total=total_index_tokens,
        desc=f"Counting tokens across {num_shards} shards",
        unit="tok",
        unit_scale=True,
    ) as pbar:
        for s, shard_token_count in enumerate(shard_token_counts):
            pbar.set_postfix_str(f"shard {s + 1}/{num_shards}", refresh=False)
            shard_start = time.perf_counter()
            log(
                f"Phase 4/6: starting shard {s + 1}/{num_shards} "
                f"({shard_token_count:,} tokens); counting token IDs."
            )
            count_start = time.perf_counter()
            shard_counts = engine.compute_unigram_counts(s=s)
            count_elapsed = time.perf_counter() - count_start
            log(
                f"Shard {s + 1}/{num_shards}: count query returned; "
                f"aggregating counts (query took {count_elapsed:.1f}s)."
            )
            count_items = (
                shard_counts.items()
                if isinstance(shard_counts, Mapping)
                else enumerate(shard_counts)
            )
            for token_id, count in count_items:
                if count > 0:
                    token_id = int(token_id)
                    unigram_counts[token_id] = (
                        unigram_counts.get(token_id, 0) + int(count)
                    )
            pbar.update(shard_token_count)
            log(
                f"Shard {s + 1}/{num_shards} complete in "
                f"{time.perf_counter() - shard_start:.1f}s; "
                f"{len(unigram_counts):,} unique token IDs accumulated."
            )

    phase_start = time.perf_counter()
    log("Phase 5/6: normalizing counts into probabilities and log probabilities.")
    total_tokens = sum(unigram_counts.values())
    if total_tokens == 0:
        raise RuntimeError("No tokens found in index; cannot compute unigram probabilities.")

    unigram_logprobs: dict[int, float] = {}
    unigram_probs: dict[int, float] = {}
    for token_id, count in unigram_counts.items():
        unigram_probs[token_id] = count / total_tokens
        unigram_logprobs[token_id] = math.log(count) - math.log(total_tokens)
    log(
        f"Phase 5/6 complete in {time.perf_counter() - phase_start:.1f}s: "
        f"normalized {len(unigram_logprobs):,} unique token IDs."
    )

    phase_start = time.perf_counter()
    log("Phase 6/7: preparing example-token and top-token summary.")
    print("DATA INFO")
    print(f"\tTotal tokens: {total_tokens}")
    print(f"\tNumber of unique tokens: {len(unigram_logprobs)}")

    example_token_ids = enc.encode(args.example_token)
    if example_token_ids:
        token_id = example_token_ids[0]
        if token_id in unigram_logprobs:
            print(
                f"\tEXAMPLE: Token '{enc.decode([token_id])}' (ID {token_id}): "
                f"logprob = {unigram_logprobs[token_id]:.4f} prob = {unigram_probs[token_id]}"
            )
        else:
            print(
                f"\tEXAMPLE: Token '{enc.decode([token_id])}' (ID {token_id}) not found in unigram counts"
            )

    top_tokens = sorted(unigram_logprobs.items(), key=lambda x: x[1], reverse=True)[: args.top_k]
    print(f"\tTop {len(top_tokens)} tokens by log probability:")
    for token_id, logprob in top_tokens:
        print(
            f"\t\tToken '{enc.decode([token_id])}' (ID {token_id}): "
            f"logprob = {logprob:.4f} prob = {unigram_probs[token_id]}"
        )

    prob_sum = sum(unigram_probs.values())
    print(f"\tSum of unigram probabilities: {prob_sum:.6f}")
    log(f"Phase 6/7 complete in {time.perf_counter() - phase_start:.1f}s.")

    phase_start = time.perf_counter()
    log("Phase 7/7: decoding token strings and preparing output JSON.")
    unigram_data = {}
    for token_id, logprob in unigram_logprobs.items():
        unigram_data[token_id] = {
            "token": enc.decode([token_id]),
            "log_prob": logprob,
            "prob": unigram_probs[token_id],
        }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    log(f"Writing {len(unigram_data):,} token entries to {args.output_path}.")
    with open(args.output_path, "w") as f:
        json.dump(unigram_data, f, indent=2)

    log(
        f"Phase 7/7 complete in {time.perf_counter() - phase_start:.1f}s; "
        f"saved unigram probabilities to {args.output_path}."
    )
    log(f"All phases finished in {time.perf_counter() - run_start:.1f}s.")


if __name__ == "__main__":
    main()

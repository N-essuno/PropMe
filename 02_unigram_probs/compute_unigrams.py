"""
Example run

python 02_unigram_probs/compute_unigrams.py \
    --index-dir 00_data/dummy_index \
    --output-path 02_unigram_probs/unigram_probs_dummy.json \
    --tokenizer-model meta-llama/Llama-2-7b-hf \
    --example-token a \
    --top-k 10
"""

import argparse
import math
import json
import os

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from tqdm import tqdm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute unigram probabilities from an InfiniGram index."
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Path to the InfiniGram index directory.",
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
    args = build_arg_parser().parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    enc = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(
        index_dir=args.index_dir,
        eos_token_id=enc.eos_token_id,
        precompute_unigram_logprobs=True,
    )

    # Compute unigram counts per shard, then aggregate.
    num_shards = engine.engine.get_num_shards()
    unigram_counts: dict[int, int] = {}
    for s in tqdm(range(num_shards), desc="Processing shards"):
        shard_counts = engine.compute_unigram_counts(s=s)
        for token_id, count in enumerate(shard_counts):
            if count > 0:
                unigram_counts[token_id] = unigram_counts.get(token_id, 0) + count

    total_tokens = sum(unigram_counts.values())
    if total_tokens == 0:
        raise RuntimeError("No tokens found in index; cannot compute unigram probabilities.")

    unigram_logprobs: dict[int, float] = {}
    unigram_probs: dict[int, float] = {}
    for token_id, count in unigram_counts.items():
        unigram_probs[token_id] = count / total_tokens
        unigram_logprobs[token_id] = math.log(count) - math.log(total_tokens)

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

    unigram_data = {}
    for token_id, logprob in unigram_logprobs.items():
        unigram_data[token_id] = {
            "token": enc.decode([token_id]),
            "log_prob": logprob,
            "prob": unigram_probs[token_id],
        }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(unigram_data, f, indent=2)

    print(f"Saved unigram probabilities to: {args.output_path}")


if __name__ == "__main__":
    main()

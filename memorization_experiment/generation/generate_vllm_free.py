"""
Prompt-free generation through a running vLLM OpenAI-compatible server.

Two settings, both starting from the model's start-of-document token:

    unconditional  the start-of-document token only: samples straight from the
                   model's own distribution, with no bias from a prompt.
    minimal_cue    the start-of-document token plus one very common word of a
                   language, which fixes the language but not the content.
                   Each generation gets its own word, drawn from the --cue_top_n
                   most frequent words of --lang in wordfreq, weighted by their
                   frequency, and capitalized because it starts a document.

The start-of-document token is the tokenizer's BOS token. Tokenizers without
one (e.g. Olmo 3) separate training documents with their EOS token, so that is
used instead. Neither is added when vLLM tokenizes a text prompt, so prompts
are sent as token ids.

Generations sharing a prompt are requested together with vLLM's `n`. vLLM
seeds the k-th of them with seed + k, and each request starts at the seed of
its first generation, so generation g of a prompt is always sampled with
seed + g and the output does not depend on how requests are split.

Example:
    vllm serve danish-foundation-models/dfm-decoder-open-v0-7b-pt --served-model-name dfm-main
    python memorization_experiment/generation/generate_vllm_free.py --model dfm-main \\
        --tokenizer danish-foundation-models/dfm-decoder-open-v0-7b-pt \\
        --setting minimal_cue --lang da --num_generations 1000 --output_file minimal_cue_da_generations.json

The output has the same layout as generate_vllm.py (readable by
simple_trace.py --is-generation-json): each record has the `prompt_id` of its
distinct prompt (in order of first draw), its `sample_idx` among that prompt's
generations, and `prompt_token_ids`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from generate_vllm import _api_url, _http_json_request, check_server


SETTINGS = ("unconditional", "minimal_cue")


def doc_start_token(tokenizer) -> tuple[int, str]:
    """Id and name of the token that starts a training document."""
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id, tokenizer.bos_token
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id, tokenizer.eos_token
    raise ValueError("Tokenizer has neither a BOS nor an EOS token")


def sample_cues(lang: str, num_samples: int, top_n: int, seed: int) -> list[str]:
    """One word per generation, drawn by frequency from the top_n most common words of `lang`."""
    import wordfreq

    words: list[str] = []
    # Over-fetch: the list also holds numbers and words with apostrophes, which are skipped.
    for word in wordfreq.top_n_list(lang, top_n * 3):
        if word.isalpha():
            words.append(word)
        if len(words) == top_n:
            break
    weights = [wordfreq.word_frequency(word, lang) for word in words]
    picks = random.Random(seed).choices(words, weights=weights, k=num_samples)
    return [word[0].upper() + word[1:] for word in picks]


def request_generations(
    *,
    args: argparse.Namespace,
    prompt_ids: list[int],
    first_generation: int,
    n: int,
) -> list[str]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt_ids,
        "n": n,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        # vLLM seeds the k-th of the n generations with seed + k.
        "seed": args.seed + first_generation,
    }
    response = _http_json_request(
        "POST",
        _api_url(args.api_base, "/completions"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        payload=payload,
        timeout_s=args.request_timeout_s,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != n:
        raise RuntimeError(f"Expected {n} choices, got: {response}")
    texts: list[str | None] = [None] * n
    for choice in choices:
        texts[int(choice["index"])] = str(choice.get("text", ""))
    if any(text is None for text in texts):
        raise RuntimeError(f"Missing choice indices in response: {choices}")
    return texts


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt-free generation through a running vLLM server.")
    parser.add_argument("--model", required=True, help="Model name the vLLM server serves.")
    parser.add_argument("--tokenizer", required=True, help="HF id of the model, for its start-of-document token.")
    parser.add_argument("--revision", default="main", help="HF revision of the tokenizer.")
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--lang", default=None, help="wordfreq language code for minimal_cue (e.g. en, da).")
    parser.add_argument("--num_generations", type=int, required=True, help="Total generations to sample.")
    parser.add_argument("--cue_top_n", type=int, default=100, help="Cue words are drawn from this many most common words.")
    parser.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--skip_server_check", action="store_true")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Seeds both the cue sampling and generation.")
    parser.add_argument("--max_n_per_request", type=int, default=256, help="Generations per API request.")
    parser.add_argument("--concurrency", type=int, default=8, help="API requests in flight.")
    parser.add_argument("--request_timeout_s", type=float, default=1000.0)
    args = parser.parse_args()
    if args.num_generations < 1:
        parser.error("--num_generations must be >= 1")
    if args.setting == "minimal_cue" and not args.lang:
        parser.error("--lang is required for minimal_cue")
    return args


def main():
    args = parse_args()
    if not args.skip_server_check:
        print(f"[*] Checking vLLM server: {args.api_base}")
        check_server(args.api_base, args.api_key, args.request_timeout_s)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    start_id, start_token = doc_start_token(tokenizer)

    # The prompt text of every generation, in sampling order.
    if args.setting == "unconditional":
        prompt_texts = [""] * args.num_generations
    else:
        prompt_texts = sample_cues(args.lang, args.num_generations, args.cue_top_n, args.seed)
    counts = Counter(prompt_texts)
    prompt_ids = {
        text: [start_id, *tokenizer.encode(text, add_special_tokens=False)] for text in counts
    }
    print(f"[*] {args.setting}: {args.num_generations} generations from {len(counts)} distinct prompts, "
          f"starting with {start_token!r} ({start_id})")

    # Requests of at most max_n_per_request generations of one prompt each.
    requests = [
        (text, first, min(args.max_n_per_request, count - first))
        for text, count in counts.items()
        for first in range(0, count, args.max_n_per_request)
    ]
    started_at = time.time()
    completions: dict[str, list[str]] = {text: [""] * count for text, count in counts.items()}

    def run(request: tuple[str, int, int]) -> None:
        text, first, n = request
        texts = request_generations(args=args, prompt_ids=prompt_ids[text], first_generation=first, n=n)
        completions[text][first : first + n] = texts

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(tqdm(pool.map(run, requests), total=len(requests), desc=args.setting, unit="request"))

    # Records in sampling order: the k-th time a prompt was drawn gets its generation k.
    prompt_id = {text: i for i, text in enumerate(counts)}
    seen: Counter = Counter()
    records = []
    for text in prompt_texts:
        g = seen[text]
        seen[text] += 1
        completion = completions[text][g]
        records.append({
            "prompt": text,
            "prompt_id": prompt_id[text],
            "sample_idx": g,
            "prompt_token_ids": prompt_ids[text],
            "completion": completion.strip(),
            "full_text": f"{text}{completion}",
        })

    set_name = args.setting if args.setting == "unconditional" else f"{args.setting}_{args.lang}"
    payload = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "api_key"},
            "doc_start_token": start_token,
            "doc_start_token_id": start_id,
            "cue_counts": dict(counts.most_common()) if args.setting == "minimal_cue" else None,
        },
        "model_is_encoder_decoder": False,
        "results": {set_name: {set_name: records}},
        "runtime_seconds": round(time.time() - started_at, 3),
        "generated_items": len(records),
    }
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[*] Saved {len(records)} generations to: {out_path}")


if __name__ == "__main__":
    main()

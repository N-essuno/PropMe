#!/usr/bin/env python3
"""Sample Tatoeba sentences and compute their Llama-2 token lengths.

Expects the raw files downloaded into `raw/` (see README.md). Run from the
repository root:

    python 00_prepare_data/propensity_settings/build_samples.py

Writes one `tatoeba_<lang>_2k.jsonl` per language (fields: id, text, source,
lang, n_tokens) and `length_stats.json`.
"""

from __future__ import annotations

import argparse
import bz2
import json
import random
import statistics
from pathlib import Path

from transformers import AutoTokenizer


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"


def load_tatoeba(lang: str) -> list[dict]:
    """Load a Tatoeba per-language export, dropping empty and duplicate texts."""
    path = RAW_DIR / f"{lang}_sentences_detailed.tsv.bz2"
    records: list[dict] = []
    seen: set[str] = set()
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            # id, lang, text, username, date_added, date_modified. Split from both
            # ends so a stray tab inside the text cannot shift the columns.
            head, *_ = line.rstrip("\n").rsplit("\t", 3)
            sent_id, sent_lang, text = head.split("\t", 2)
            text = text.strip()
            if sent_lang != lang or not text or text in seen:
                continue
            seen.add(text)
            records.append({"id": int(sent_id), "text": text, "source": "tatoeba", "lang": lang})
    return records


def length_stats(lengths: list[int]) -> dict:
    return {
        "num_samples": len(lengths),
        "avg_tokens": round(statistics.mean(lengths), 2),
        "std_tokens": round(statistics.pstdev(lengths), 2),
        "median_tokens": statistics.median(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "total_tokens": sum(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-model", default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)

    sets: dict[str, list[dict]] = {}
    for lang in ("eng", "dan"):
        pool = load_tatoeba(lang)
        print(f"tatoeba_{lang}: {len(pool)} unique sentences")
        sets[f"tatoeba_{lang}"] = random.Random(args.seed).sample(pool, args.num_samples)

    stats: dict[str, dict] = {}
    for name, records in sets.items():
        # Count content tokens only (no BOS), as SimpleTrace does when tracing.
        encoded = tokenizer([r["text"] for r in records], add_special_tokens=False)["input_ids"]
        for record, ids in zip(records, encoded):
            record["n_tokens"] = len(ids)
        out_path = DATA_DIR / f"{name}_2k.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats[name] = length_stats([r["n_tokens"] for r in records])
        print(f"{name}: {stats[name]} -> {out_path.name}")

    stats["_config"] = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "tokenizer_model": args.tokenizer_model,
    }
    with open(DATA_DIR / "length_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score the English candidate documents with the FineWeb-Edu classifier.

One quality bar for every English source (Common Corpus, HPLT, FinePDFs,
Wikipedia): HuggingFaceFW/fineweb-edu-classifier on the first 512 tokens of
each document. As in FineWeb-Edu, `quality_int` = round(clamp(score, 0, 5)),
and build_candidates.py keeps documents with quality_int >= 2 (the top ~30-45%
of each source; >= 3 keeps 0-16%, which empties most Common Corpus sources).

Reads work/pool_<corpus>.jsonl (build_candidates.py pool), writes
work/quality_<corpus>.jsonl ({source, doc_id, quality, quality_int}) and prints
the score distribution per source.

Run from the repository root on a GPU:

    python 00_prepare_data/propensity_settings/quality_score.py --corpora cp d3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent
WORK_DIR = DATA_DIR / "work"
MODEL = "HuggingFaceFW/fineweb-edu-classifier"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", default=["cp", "d3"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    for corpus in args.corpora:
        with open(WORK_DIR / f"pool_{corpus}.jsonl", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        # Longest first: batches pad less and memory errors show up at once.
        texts = [d["text"][:20_000] for d in docs]
        order = sorted(range(len(docs)), key=lambda i: -len(texts[i]))
        scores = [0.0] * len(docs)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = tokenizer([texts[i] for i in idx], truncation=True, max_length=args.max_tokens,
                              padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = model(**batch).logits.squeeze(-1).float().cpu().tolist()
            for i, s in zip(idx, logits):
                scores[i] = s
        rows = [{"source": d["source"], "doc_id": d["doc_id"], "quality": round(s, 4),
                 "quality_int": int(round(max(0.0, min(5.0, s))))} for d, s in zip(docs, scores)]
        with open(WORK_DIR / f"quality_{corpus}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        dist = defaultdict(Counter)
        for row in rows:
            dist[row["source"]][row["quality_int"]] += 1
        print(f"=== {corpus}: {len(rows)} docs; quality_int distribution (share >= 3) ===")
        for source, counts in sorted(dist.items()):
            n = sum(counts.values())
            print(f"  {source}: {dict(sorted(counts.items()))}  >=3: {sum(c for q, c in counts.items() if q >= 3) / n:.1%}")


if __name__ == "__main__":
    main()

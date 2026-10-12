#!/usr/bin/env python3
"""Language ID of the candidate sentences.

Some documents mix languages: English abstracts in tidsskrift-dk, Portuguese
or Dutch passages in "English" Eurlex documents. The function-word check in
build_candidates.py lets a few through, because words like "have", "for" and "i"
exist in both languages. qanastek/51-languages-classifier (XLM-R trained on
the short MASSIVE utterances) scores every candidate. `p_target` is the
probability of the corpus language; for Danish it includes Norwegian Bokmål,
its closest neighbour, which the model confuses with Danish in short
sentences. build_unseen_prompt_sets.py keeps candidates with
p_target >= sources.LANGID_MIN_P_TARGET (0.2): below it are almost only
sentences in another language; between 0.2 and 0.5 are mostly archaic Danish
spelling (kalliope, kb_historical_letters) and English headings.

Reads candidates_<corpus>.jsonl, writes work/langid_<corpus>.jsonl ({id, top,
top_score, p_target}).

Run from the repository root on a GPU:

    python 00_prepare_data/propensity_settings/lang_id.py --corpora dw cp d3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent
WORK_DIR = DATA_DIR / "work"
MODEL = "qanastek/51-languages-classifier"
TARGET_LABELS = {"dw": ("da-DK", "nb-NO"), "cp": ("en-US",), "d3": ("en-US",)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", default=list(TARGET_LABELS))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).cuda().eval()
    labels = model.config.id2label
    for corpus in args.corpora:
        with open(DATA_DIR / f"candidates_{corpus}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        target_ids = [i for i, label in labels.items() if label in TARGET_LABELS[corpus]]
        out = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            enc = tokenizer([r["text"] for r in batch], truncation=True, max_length=128, padding=True,
                            return_tensors="pt").to("cuda")
            with torch.inference_mode():
                probs = model(**enc).logits.float().softmax(-1).cpu()
            for r, p in zip(batch, probs):
                top = int(p.argmax())
                out.append({"id": r["id"], "top": labels[top], "top_score": round(float(p[top]), 4),
                            "p_target": round(float(p[target_ids].sum()), 4)})
        with open(WORK_DIR / f"langid_{corpus}.jsonl", "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        print(f"{corpus}: {len(out)} candidates scored")


if __name__ == "__main__":
    main()

"""Build the shared input files used to compare OLMoTrace and SimpleTrace.

Every file is JSONL with a `text` field (traced by both tools) and an optional
`prompt` field (used only by OLMoTrace's BM25 query). Both tools read exactly
the same file, in the same order.

python olmotrace/comparison/prepare_inputs.py
"""

import json
import os
import random
import sys

from transformers import AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "03_tracing"))
from data_loading import load_generic_dataset  # noqa: E402

OUT_DIR = os.path.join(REPO, "olmotrace", "comparison", "inputs")
DATA = os.path.join(REPO, "memorization_experiment", "data")
VERBATIM_TOKENS = 256  # same length as the model generations (max_new_tokens=256)
SEED = 1234

enc = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)


def write(name: str, rows: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lens = [len(enc.encode(r["text"])) for r in rows]
    print(f"{name}: {len(rows)} rows, tokens mean {sum(lens) / len(lens):.1f}, max {max(lens)}")


def truncate(text: str, n: int) -> str:
    return enc.decode(enc.encode(text)[:n])


def verbatim_rows(sample_docs_path: str, n: int) -> list[dict]:
    """First VERBATIM_TOKENS tokens of training documents sampled from the index."""
    rows = []
    with open(sample_docs_path) as f:
        for line in f:
            doc = json.loads(line)
            if len(enc.encode(doc["text"])) < VERBATIM_TOKENS:
                continue
            meta = doc.get("metadata") or {}
            rows.append({
                "text": truncate(doc["text"], VERBATIM_TOKENS),
                "source_doc_ix": doc["doc_ix"],
                "source_id": meta.get("id") or doc["doc_ix"],
            })
            if len(rows) == n:
                break
    return rows


def dynaword_generations() -> list[dict]:
    """Greedy 7B generations (dfm-decoder-open-v0-7b-pt) for the three prompt sets."""
    rows = []
    for setting, path in [
        ("generic", "dynaword/generic/generic_generations.json"),
        ("specific", "dynaword/specific/specific_generations.json"),
        ("prefix", "dynaword/prefix/dynaword_prefix_generations.json"),
    ]:
        with open(os.path.join(DATA, path)) as f:
            results = json.load(f)["results"]
        for prompt_set in results.values():
            for domain, items in prompt_set.items():
                for item in items:
                    if item.get("completion", "").strip():
                        rows.append({"text": item["completion"], "prompt": item["prompt"],
                                     "setting": setting, "domain": domain})
    return rows


def main() -> None:
    random.seed(SEED)
    write("dynaword_generations", dynaword_generations())
    write("dynaword_verbatim", verbatim_rows(os.path.join(DATA, "dynaword2/dynaword_sample_docs.jsonl"), 64))
    write("commonpile_verbatim", verbatim_rows(os.path.join(DATA, "commonpile/commonpile_sample_docs.jsonl"), 64))
    write("dolma3_verbatim", verbatim_rows(os.path.join(DATA, "dolma3/dolma3_sample_docs.jsonl"), 64))
    write("english_novel", [{"text": t} for t in load_generic_dataset()])


if __name__ == "__main__":
    main()

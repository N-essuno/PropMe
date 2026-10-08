#!/usr/bin/env python3
"""Write fixed-size random samples of the larger prompt sets and their generations.

For every prompt set memorization_experiment/data/<dataset>/<setting>/<stem>_prompts.jsonl
(dataset: commonpile, dolma3, dynaword2) with more than --size prompts, this samples
--size of them (seed --seed, kept in file order) into <stem>_prompts_<size>.jsonl, and
for every model's generations of the set, generations/<model>/<stem>_generations.json,
keeps all generations (every sample_idx) of the sampled prompts in
<stem>_generations_<size>.json, with the same layout.

Generations are matched to prompt lines by prompt_id when it still gives each record's
prompt text (generate_vllm.py numbers prompts grouped by domain), and otherwise by text,
which must then be unique in the set: prompt sets filtered after generation (e.g. after
excluding new full matches) leave gaps in the prompt_ids, and some unfiltered sets hold
repeated texts. Existing outputs are overwritten.

Run from the repository root:

    python memorization_experiment/generation/subsample_prompt_sets.py --dry-run
    python memorization_experiment/generation/subsample_prompt_sets.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "memorization_experiment" / "data"
DATASETS = ("commonpile", "dolma3", "dynaword2")
PROMPTS_SUFFIX = "_prompts.jsonl"
GENERATIONS_SUFFIX = "_generations.json"


def read_jsonl_lines(path: Path) -> list[str]:
    # Split on "\n" only: JSON strings may hold unescaped U+2028 and similar.
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def prompt_id_lines(rows: list[dict]) -> list[int]:
    """Line of each prompt_id: generate_vllm.py numbers the prompts it loads grouped by
    domain (in order of first appearance), skipping rows without text."""
    by_domain: dict[str, list[int]] = {}
    for line, row in enumerate(rows):
        text, domain = row.get("text"), row.get("domain")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(domain, str) or not domain.strip():
            domain = "default"
        by_domain.setdefault(domain, []).append(line)
    return [line for lines in by_domain.values() for line in lines]


def records_of(data: dict) -> list[dict]:
    return [r for domains in data["results"].values() for records in domains.values() for r in records]


def record_lines(data: dict, rows: list[dict], id_lines: list[int]) -> tuple[list[int], str]:
    """Prompt line of each generation record, and how it was matched."""
    records = records_of(data)
    if all(0 <= r["prompt_id"] < len(id_lines) and rows[id_lines[r["prompt_id"]]]["text"] == r["prompt"] for r in records):
        return [id_lines[r["prompt_id"]] for r in records], "prompt_id"
    line_of_text = {row["text"]: line for line, row in enumerate(rows)}
    if len(line_of_text) != len(rows):
        raise ValueError("prompt_ids do not match the prompt file and its texts are not unique")
    missing = {r["prompt"] for r in records} - line_of_text.keys()
    if missing:
        raise ValueError(f"{len(missing)} generated prompts are not in the prompt file")
    return [line_of_text[r["prompt"]] for r in records], "text"


def subsample_generations(data: dict, keep: list[bool], source: str, size: int, seed: int) -> dict:
    """`data` with only the records flagged in `keep` (in records_of order)."""
    flags = iter(keep)
    kept = {
        set_name: {domain: [r for r in records if next(flags)] for domain, records in domains.items()}
        for set_name, domains in data["results"].items()
    }
    out = {**data, "results": kept}
    out["generated_items"] = sum(len(v) for domains in kept.values() for v in domains.values())
    out["subsample"] = {"prompts": size, "seed": seed, "source": source}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=1000, help="Prompts per sampled set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written.")
    args = parser.parse_args()

    for dataset in DATASETS:
        for path in sorted((DATA_DIR / dataset).glob(f"*/*{PROMPTS_SUFFIX}")):
            lines = read_jsonl_lines(path)
            if len(lines) <= args.size:
                continue
            rows = [json.loads(line) for line in lines]
            sampled = sorted(random.Random(args.seed).sample(range(len(rows)), args.size))
            in_sample = set(sampled)
            stem = path.name[: -len(PROMPTS_SUFFIX)]
            out_path = path.with_name(f"{stem}_prompts_{args.size}.jsonl")
            print(f"{path.relative_to(REPO_ROOT)}: {len(rows)} -> {args.size} prompts")
            if not args.dry_run:
                out_path.write_text("".join(lines[i] + "\n" for i in sampled), encoding="utf-8")

            id_lines = prompt_id_lines(rows)
            for gen_path in sorted(path.parent.glob(f"generations/*/{stem}{GENERATIONS_SUFFIX}")):
                data = json.loads(gen_path.read_text(encoding="utf-8"))
                record_line, matched_by = record_lines(data, rows, id_lines)
                keep = [line in in_sample for line in record_line]
                out = subsample_generations(data, keep, gen_path.name, args.size, args.seed)
                prompts_kept = {line for line, k in zip(record_line, keep) if k}
                print(
                    f"    {gen_path.parent.name}: {len(keep)} -> {out['generated_items']} generations "
                    f"of {len(prompts_kept)} prompts (matched by {matched_by})"
                )
                if len(prompts_kept) != args.size:
                    raise ValueError(f"{gen_path}: generations cover {len(prompts_kept)} of the {args.size} sampled prompts")
                if not args.dry_run:
                    gen_out = gen_path.with_name(f"{stem}_generations_{args.size}.json")
                    gen_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

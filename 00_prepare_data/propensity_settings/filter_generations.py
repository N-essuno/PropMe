#!/usr/bin/env python3
"""Move generations of prompts no longer in their prompt set out of the generation files.

When a prompt set is rebuilt with fewer prompts (e.g. after excluding new full
matches), its existing generation files still hold the generations of the
dropped prompts. For every model's generations of a prompt set (next to the
prompts, see run_tracing.generations_path) this keeps the records whose prompt
is in the current prompt set (run_tracing.prompt_set_path) and moves the others,
with the same layout, to _notes/dropped_generations/<model>/<prompt set>_generations.json
(merged with any records moved there before). Prompt-free settings
(unconditional, minimal_cue_*) and prefix prompts are left alone.

Run from the repository root:

    python 00_prepare_data/propensity_settings/filter_generations.py --dry-run
    python 00_prepare_data/propensity_settings/filter_generations.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_tracing as rt


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]
DROPPED_DIR = REPO_ROOT / "_notes" / "dropped_generations"
SUFFIX = "_generations.json"


def split_records(data: dict, keep_prompts: set[str]) -> tuple[dict, dict]:
    """Split a generate_vllm.py output into (kept, dropped), both with its layout."""
    kept = {set_name: {} for set_name in data["results"]}
    dropped = {set_name: {} for set_name in data["results"]}
    for set_name, domains in data["results"].items():
        for domain, records in domains.items():
            kept[set_name][domain] = [r for r in records if r["prompt"] in keep_prompts]
            gone = [r for r in records if r["prompt"] not in keep_prompts]
            if gone:
                dropped[set_name][domain] = gone

    def with_results(results: dict) -> dict:
        out = {**data, "results": results}
        out["generated_items"] = sum(len(v) for domains in results.values() for v in domains.values())
        return out

    return with_results(kept), with_results(dropped)


def merge_dropped(path: Path, new: dict) -> dict:
    """Add newly dropped records to those already saved at `path`."""
    if not path.exists():
        return new
    old = json.loads(path.read_text(encoding="utf-8"))
    for set_name, domains in new["results"].items():
        for domain, records in domains.items():
            old["results"].setdefault(set_name, {}).setdefault(domain, []).extend(records)
    old["generated_items"] = sum(len(v) for domains in old["results"].values() for v in domains.values())
    return old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would move without writing.")
    args = parser.parse_args()

    prompt_sets = [
        name
        for x in rt.PROMPT_SET_DATASETS
        for name in (f"generic_{x}", f"specific_{x}")
    ]
    files = [
        (prompt_set, path)
        for prompt_set in prompt_sets
        for path in sorted(rt.generations_path(prompt_set, "*").parent.parent.glob(f"*/*{SUFFIX}"))
    ]
    for prompt_set, path in files:
        prompts_path = rt.prompt_set_path(prompt_set)
        with open(prompts_path, encoding="utf-8") as f:
            keep_prompts = {json.loads(line)["text"] for line in f}

        data = json.loads(path.read_text(encoding="utf-8"))
        kept, dropped = split_records(data, keep_prompts)
        n_dropped = dropped["generated_items"]
        dropped_prompts = {r["prompt"] for d in dropped["results"].values() for v in d.values() for r in v}
        kept_prompts = {r["prompt"] for d in kept["results"].values() for v in d.values() for r in v}
        missing = keep_prompts - kept_prompts
        print(
            f"{path.parent.name}/{prompt_set}: keep {kept['generated_items']}, move {n_dropped} "
            f"({len(dropped_prompts)} prompts)" + (f"; {len(missing)} prompts have no generations" if missing else "")
        )
        if args.dry_run or n_dropped == 0:
            continue

        out_path = DROPPED_DIR / path.parent.name / f"{prompt_set}{SUFFIX}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dropped["dropped_reason"] = f"prompt no longer in {prompts_path.relative_to(REPO_ROOT)}"
        out_path.write_text(json.dumps(merge_dropped(out_path, dropped), indent=2, ensure_ascii=False), encoding="utf-8")
        path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

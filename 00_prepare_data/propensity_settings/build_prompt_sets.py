#!/usr/bin/env python3
"""Build the generic prompt sets from the Tatoeba sentences that do not occur verbatim in an index.

Each set keeps the sentences of one (sample set, index) grouping in groups
1-5 (group_by_overlap.py: 1, 3, 4, 5, i.e. every sentence that is not a full
match):

    generic_cp   tatoeba_eng vs Common Pile
    generic_d3   tatoeba_eng vs Dolma3
    generic_dw   tatoeba_dan vs Dynaword

The specific prompt sets (unseen-source sentences) are built by
build_unseen_prompt_sets.py. Only groups/ is needed, not the index.

Run from the repository root after group_by_overlap.py:

    python 00_prepare_data/propensity_settings/build_prompt_sets.py

Writes each set to memorization_experiment/data/<dataset>/<setting>/<setting>_prompts.jsonl
(run_tracing.prompt_set_path; e.g. generic_dw -> dynaword2/generic/generic_prompts.jsonl),
with the `text` and `domain` fields the memorization_experiment prompt files use
plus provenance fields, and prompt_sets/README.md summarizing the sizes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import run_tracing as rt


K_DOCS = 100
DATA_DIR = Path(__file__).resolve().parent
GROUPS_DIR = DATA_DIR / "groups"
OUT_DIR = rt.PROMPT_SETS_DIR  # README.md
KEPT_GROUPS = ("1", "3", "4", "5")
ALL_GROUPS = ("1", "3", "4", "5", "6", "7")
INDEX_NAMES = {"commonpile": "Common Pile", "dolma3_split": "Dolma3", "dynaword": "Dynaword"}

# name -> (sample set, index the grouping was computed against)
PROMPT_SETS = {
    "generic_cp": ("tatoeba_eng", "commonpile"),
    "generic_d3": ("tatoeba_eng", "dolma3_split"),
    "generic_dw": ("tatoeba_dan", "dynaword"),
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(name: str, set_name: str, index_name: str) -> dict:
    """Write a prompt set; returns its stats."""
    samples = {r["text"]: r for r in load_jsonl(DATA_DIR / f"{set_name}_2k.jsonl")}
    groups = load_jsonl(GROUPS_DIR / f"groups_{set_name}_{index_name}.jsonl")

    prompts = []
    for row in groups:
        if row["group"] not in KEPT_GROUPS:
            continue
        sample = samples[row["text"]]
        prompts.append(
            {
                "text": row["text"],
                "domain": sample.get("domain", sample["source"]),
                "id": sample["id"],
                "source": sample["source"],
                "lang": sample["lang"],
                "group": row["group"],
                "coverage_tokens": row["coverage_tokens"],
                "coverage_chars": row["coverage_chars"],
                "n_tokens": row["n_tokens"],
            }
        )

    write_jsonl(rt.prompt_set_path(name), prompts)
    return set_stats(name, set_name, index_name, prompts, excluded=len(groups) - len(prompts))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_stats(name: str, set_name: str, index_name: str, prompts: list[dict], excluded: int | None) -> dict:
    lengths = [p["n_tokens"] for p in prompts]
    return {
        "name": name,
        "set": set_name,
        "index": index_name,
        "size": len(prompts),
        "excluded_full_matches": excluded,
        "by_group": {g: sum(p["group"] == g for p in prompts) for g in ALL_GROUPS},
        "avg_tokens": round(statistics.mean(lengths), 2) if lengths else 0,
        "median_tokens": statistics.median(lengths) if lengths else 0,
    }


def write_summary(stats: list[dict]) -> None:
    lines = [
        "# Prompt sets",
        "",
        "Sentences from the 2k Tatoeba samples that do not occur verbatim in the given index:",
        "overlap groups 1, 3, 4 and 5 from `../group_by_overlap.py` (groups 6 and 7, full-sentence matches, are excluded).",
        "These are the generic sets; the specific sets (unseen-source sentences) are described in `../README.md`.",
        "Built with `../build_prompt_sets.py`. The `file` column is relative to the repository root.",
        "",
        "| prompt set | file | source sample | index | prompts | excluded full matches | avg / median Llama-2 tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s['name']} | `{rt.prompt_set_path(s['name']).relative_to(rt.REPO_ROOT)}` "
            f"| `{s['set']}_2k` | {INDEX_NAMES[s['index']]} "
            f"| {s['size']} | {'-' if s['excluded_full_matches'] is None else s['excluded_full_matches']} "
            f"| {s['avg_tokens']} / {s['median_tokens']:g} |"
        )
    lines += [
        "",
        "## By overlap group",
        "",
        "Group = share of the prompt (in Llama-2 tokens) covered by its longest span that occurs verbatim in the index;",
        f"6 / 7 = the whole prompt occurs verbatim, in more than / at most {K_DOCS} documents (excluded from the sets).",
        "",
        "| prompt set | 1: < 25% | 3: 25-50% | 4: 50-75% | 5: >= 75%, not full | 6: full, > k docs | 7: full, <= k docs |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        cells = " | ".join(
            f"{s['by_group'][g]} ({100 * s['by_group'][g] / s['size']:.1f}%)" if s["size"] else "0"
            for g in ALL_GROUPS
        )
        lines.append(f"| {s['name']} | {cells} |")
    lines += [
        "",
        "## Fields",
        "",
        "- `text`, `domain`: as in the `memorization_experiment` prompt files (`domain` is `tatoeba`).",
        "- `id`, `source`, `lang`: the sentence in the 2k sample (`../<set>_2k.jsonl`).",
        "- `group`, `coverage_tokens`, `coverage_chars`, `n_tokens`: its overlap with the index (`../groups/`).",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    stats = [build(name, set_name, index_name) for name, (set_name, index_name) in PROMPT_SETS.items()]
    write_summary(stats)
    print((OUT_DIR / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

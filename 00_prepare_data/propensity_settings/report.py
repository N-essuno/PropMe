#!/usr/bin/env python3
"""Write the results report: sizes, per-source counts, quality, lengths and embedding similarity.

Reads work/candidates_report_<corpus>.json, work/selection_<corpus>.json,
work/quality_<corpus>.jsonl, the prompt sets, and the evaluation summaries
(<model>/<eval name>.json) written by scripts/embedding_similarity.py
for Qwen3-Embedding-8B and bge-m3.

The Tatoeba reference is the corpus's generic prompt set: by default its
1000-prompt sample generic_prompts_1000.jsonl (the set used in the experiments,
evaluated with --summary-name unseen_eval_1000 as set generic_<x>_1000) -> results.md;
--generic full uses the whole generic_prompts.jsonl (unseen_eval, set generic_<x>)
-> results_full_generic.md.

Run from the repository root:

    python 00_prepare_data/propensity_settings/report.py                  # 1000-prompt generic sets
    python 00_prepare_data/propensity_settings/report.py --generic full   # full generic sets
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import sources as src

REPO_ROOT = src.DATA_DIR.parents[1]
WORK_DIR = src.DATA_DIR / "work"
EVAL_DIR = src.DATA_DIR / "embedding_similarity"
PROMPT_DATA_DIR = REPO_ROOT / "memorization_experiment" / "data"
MODELS = ("Qwen3-Embedding-8B", "bge-m3")
TATOEBA = {"cp": "tatoeba_eng", "d3": "tatoeba_eng", "dw": "tatoeba_dan"}
NAMES = {"cp": "Common Pile", "d3": "Dolma 3", "dw": "Dynaword"}
# --generic -> (generic prompt file, set name in the evaluation, evaluation summary name, output file)
GENERIC = {
    "1000": ("generic_prompts_1000.jsonl", "generic_{c}_1000", "unseen_eval_1000", "results.md"),
    "full": ("generic_prompts.jsonl", "generic_{c}", "unseen_eval", "results_full_generic.md"),
}


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quantiles(values: list[int]) -> str:
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    return f"{statistics.mean(v):.1f} / {q(0.1)} / {q(0.5)} / {q(0.9)}"


def overview(corpora: list[str], generic_file: str) -> list[str]:
    lines = [
        "## Overview", "",
        "| corpus | training index | sources | candidates | passing | prompts | tokens: mean / p10 / median / p90 | generic (Tatoeba) tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in corpora:
        sel = json.loads((WORK_DIR / f"selection_{c}.json").read_text())
        prompts = read_jsonl(PROMPT_DATA_DIR / src.CORPUS_DATASET[c] / "specific" / "specific_prompts.jsonl")
        generic = read_jsonl(PROMPT_DATA_DIR / src.CORPUS_DATASET[c] / "generic" / generic_file)
        index = sel["index"] + "".join(f" (+ {e})" for e in sel["extra_indexes"])
        lines.append(
            f"| {NAMES[c]} (`unseen_{c}`) | `{index}` | {len(sel['per_source'])} | {sel['candidates']} "
            f"| {sel['passing']} | {sel['size']} | {quantiles([p['n_tokens'] for p in prompts])} "
            f"| {quantiles([p['n_tokens'] for p in generic])} |"
        )
    return lines + [""]


def per_source(corpora: list[str], eval_rows: dict) -> list[str]:
    lines = ["## Per source", "",
             "`pool`: documents drawn; `quality ≥ 2`: FineWeb-Edu int score ≥ 2 (English only); `candidates`: Tatoeba-length "
             "sentences, one per document; dropped as `full match` (overlap group 6/7 or an occurrence in an extra index), "
             "`other lang.` (lang_id.py) or `trunc.` (ends in a title abbreviation; the drops can overlap); `passing`: "
             "none of these; "
             "`selected`: in the final set. `sel. max_sim`: mean similarity to the seed-43 selection reference of the "
             "survivors -> the selected; `eval max_sim`: of the selected, on the seed-42 evaluation reference.", ""]
    for c in corpora:
        sel = json.loads((WORK_DIR / f"selection_{c}.json").read_text())
        by_source = defaultdict(list)
        for row in eval_rows.get(c, []):
            by_source[row["source"]].append(row["max_sim"])
        lines += [f"### {NAMES[c]}", "",
                  "| source | pool | quality ≥ 2 | candidates | full match | other lang. | trunc. | passing | selected "
                  "| sel. max_sim | eval max_sim |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, v in sel["per_source"].items():
            q = v["quality_pass"] if src.CORPUS_LANG[c] == "eng" else "-"
            sel_sim = (f"{v['selection_max_sim_survivors']:.3f} -> {v['selection_max_sim_selected']:.3f}"
                       if v["selection_max_sim_selected"] is not None else "-")
            ev = f"{statistics.mean(by_source[s]):.3f}" if by_source[s] else "-"
            lines.append(f"| {s} | {v['pool']} | {q} | {v['candidates']} | {v['full_matches']} | {v['other_language']} "
                         f"| {v['truncated']} | {v['passing']} | {v['selected']} | {sel_sim} | {ev} |")
        lines.append("")
    return lines


def quality(corpora: list[str]) -> list[str]:
    lines = ["## Quality scores (English pools)", "",
             "FineWeb-Edu int score distribution of each source's document pool (`quality_score.py`); documents "
             "with a score ≥ 2 are used.", "",
             "| corpus | source | n | 0 | 1 | 2 | 3 | 4 | 5 | share ≥ 2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in corpora:
        path = WORK_DIR / f"quality_{c}.jsonl"
        if src.CORPUS_LANG[c] != "eng" or not path.exists():
            continue
        dist = defaultdict(Counter)
        for row in read_jsonl(path):
            dist[row["source"]][row["quality_int"]] += 1
        for s, counts in sorted(dist.items()):
            n = sum(counts.values())
            cells = " | ".join(str(counts.get(i, 0)) for i in range(6))
            lines.append(f"| {c} | {s} | {n} | {cells} | {sum(counts[i] for i in (2, 3, 4, 5)) / n:.0%} |")
    lines += ["", "Selected prompts by score:", ""]
    for c in corpora:
        if src.CORPUS_LANG[c] != "eng":
            continue
        prompts = read_jsonl(PROMPT_DATA_DIR / src.CORPUS_DATASET[c] / "specific" / "specific_prompts.jsonl")
        counts = Counter(p["quality_int"] for p in prompts)
        lines.append(f"- `unseen_{c}`: " + ", ".join(f"{k}: {counts[k]}" for k in sorted(counts)))
    return lines + [""]


def similarity(corpora: list[str], generic_set: str, eval_name: str, generic_file: str) -> tuple[list[str], dict]:
    lines = ["## Embedding similarity to the training data", "",
             f"`scripts/embedding_similarity.py` on the seed-42 sample of 10k training documents per "
             f"corpus (the selection used an independent seed-43 sample). `max_sim`: cosine to the nearest document; "
             f"`topk_mean`: mean of the 10 nearest; `mean_sim`: cosine to the sample centroid. `{generic_set.format(c='<x>')}` is the "
             f"Tatoeba prompt set of the corpus (`{generic_file}`; the 2k sample without its full matches"
             f"{', 1000 of them at random' if '1000' in generic_file else ''}); `train_<corpus>` are "
             f"sentences of other training documents (in-distribution reference). Differences: 95% bootstrap CI; "
             f"AUC: P(random unseen prompt > random Tatoeba prompt).", ""]
    eval_rows = {}
    for model in MODELS:
        path = EVAL_DIR / model / f"{eval_name}.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        lines += [f"### {model}", ""]
        for c in corpora:
            corpus = src.CORPUS_EMBEDDING[c]
            block = summary["corpora"].get(corpus)
            if block is None:
                continue
            sets = [TATOEBA[c], generic_set.format(c=c), f"unseen_{c}", f"train_{corpus}"]
            lines += [f"**{NAMES[c]}** (`{corpus}`)", "", "| set | n | avg tokens | max_sim [95% CI] | topk_mean | mean_sim |",
                      "|---|---|---|---|---|---|"]
            for s in sets:
                b = block["sets"][s]
                m = b["max_sim"]
                lines.append(f"| {s} | {b['n']} | {b['avg_n_tokens']} | {m['mean']:.4f} [{m['ci95'][0]:.4f}, "
                             f"{m['ci95'][1]:.4f}] | {b['topk_mean']['mean']:.4f} | {b['mean_sim']['mean']:.4f} |")
            lines += ["", "| comparison | max_sim diff [95% CI] (AUC) | topk_mean diff | mean_sim diff |", "|---|---|---|---|"]
            for comp in block["comparisons"]:
                if comp["a"] != f"unseen_{c}":
                    continue
                cells = [f"{comp['metrics'][k]['diff']:+.4f} [{comp['metrics'][k]['ci95'][0]:+.4f}, "
                         f"{comp['metrics'][k]['ci95'][1]:+.4f}] ({comp['metrics'][k]['auc']:.3f})"
                         for k in ("max_sim", "topk_mean", "mean_sim")]
                lines.append(f"| {comp['a']} - {comp['b']} | " + " | ".join(cells) + " |")
            # The unseen sets are slightly longer, and similarity grows with length: compare within length bins.
            reference = generic_set.format(c=c)
            bins = [b for b in block["sets"][f"unseen_{c}"]["by_length"] if b in block["sets"][reference]["by_length"]]
            lines += ["", "max_sim by Llama-2 token length, mean (n):", "", "| set | " + " | ".join(bins) + " |",
                      "|---|" + "---|" * len(bins)]
            for s in (reference, f"unseen_{c}"):
                by_length = block["sets"][s]["by_length"]
                lines.append(f"| {s} | " + " | ".join(
                    f"{by_length[b]['max_sim_mean']:.4f} ({by_length[b]['n']})" if b in by_length else "-" for b in bins) + " |")
            lines.append("")
            if model == MODELS[0]:
                eval_rows[c] = read_jsonl(EVAL_DIR / model / f"per_prompt_unseen_{c}_{corpus}.jsonl")
    return lines, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generic", choices=list(GENERIC), default="1000",
                        help="Tatoeba reference: the 1000-prompt generic sample (default) or the full generic set.")
    args = parser.parse_args()
    generic_file, generic_set, eval_name, output = GENERIC[args.generic]
    corpora = [c for c in src.CORPUS_SOURCES if (WORK_DIR / f"selection_{c}.json").exists()]
    sim_lines, eval_rows = similarity(corpora, generic_set, eval_name, generic_file)
    lines = ["# Results", "", f"Generated by `report.py --generic {args.generic}`; see README.md for the method. "
             f"Tatoeba reference: `generic/{generic_file}`.", "",
             *overview(corpora, generic_file), *sim_lines, *per_source(corpora, eval_rows), *quality(corpora)]
    (src.DATA_DIR / output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select the final specific prompt sets (unseen-source sentences) and check them.

    select  Candidates (work/traced_<corpus>.jsonl) that are not full matches
            (groups 1-5), occur in no extra index, are in the corpus language
            (lang_id.py, p_target >= sources.LANGID_MIN_P_TARGET) and do not
            end in a title abbreviation (a sentence split after "Mr." or
            "hr.") are ranked by their embedding similarity to the training
            data. The best are kept, with uniform shares per source
            (FINAL_SIZE in total).

            Similarity is measured against a selection reference: random
            training documents drawn with seed 43, independent of the seed-42
            sample used to evaluate the sets. The selection therefore cannot
            fit the evaluation sample. The score is max_sim (Qwen3-Embedding-8B,
            cosine to the nearest of 10k reference docs) minus the mean max_sim
            of the corpus's candidates of similar length (LENGTH_BINS). This
            makes the selection length-neutral, so the prompts keep the
            Tatoeba lengths they were picked with.

            Shares are water-filled: a source with fewer survivors than its
            share gives all of them, and the rest is spread evenly over the
            others. Writes run_tracing.prompt_set_path("specific_<x>"), i.e.
            memorization_experiment/data/<dataset>/specific/specific_prompts.jsonl,
            and work/selection_<corpus>.json (counts per source and stage).
    verify  Recount the verbatim occurrences of every written prompt in the
            corpus index and the extra indexes (both tokenizations), and
            check sizes, ids and per-source counts. Fails on any occurrence.

The selection reference and the candidates' similarities come from
scripts/embedding_similarity.py (see README.md).

Run from the repository root:

    python 00_prepare_data/propensity_settings/build_unseen_prompt_sets.py select verify
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

import build_prompt_sets as bps
import group_by_overlap as gbo
import run_tracing as rt
import sources as src

WORK_DIR = src.DATA_DIR / "work"
SELECTION_DIR = src.DATA_DIR / "embedding_selection" / "Qwen3-Embedding-8B"
# Token-length bins for the length-neutral similarity score.
LENGTH_BINS = ((1, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 17), (18, 21), (22, 27), (28, 10**9))
# Sentence splitting cuts after these abbreviations ("... agreement, Mr."), leaving a fragment.
TRUNCATED_END = re.compile(
    r"\b(Mr|Mrs|Ms|Messrs|Dr|Prof|Rev|Hon|St|Mt|Jr|Sr|Gen|Col|Capt|Lt|Sgt|Gov|Sen|Rep|No|Nos|Co|Inc|Ltd|vs|"
    r"hr|fru|frk|dr|nr|stk|jf|ca|pkt|kap)\.$"
)
PROVENANCE = ("subsource", "doc_id", "date", "url", "quality", "quality_int", "target_tokens")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def water_fill(available: dict[str, int], total: int, rng: random.Random) -> dict[str, int]:
    """Per-source counts summing to min(total, sum(available)), as equal as availability allows."""
    alloc = {s: 0 for s in available}
    remaining = min(total, sum(available.values()))
    open_sources = [s for s in available if available[s] > 0]
    while remaining and open_sources:
        share, extra = divmod(remaining, len(open_sources))
        bonus = set(rng.sample(sorted(open_sources), extra))
        for s in sorted(open_sources):
            give = min(share + (s in bonus), available[s] - alloc[s])
            alloc[s] += give
            remaining -= give
        open_sources = [s for s in open_sources if alloc[s] < available[s]]
    return alloc


def length_bin(n_tokens: int) -> int:
    return next(i for i, (lo, hi) in enumerate(LENGTH_BINS) if lo <= n_tokens <= hi)


def selection_scores(corpus: str) -> dict[str, dict]:
    """Candidate id -> max_sim to the selection reference and its length-neutral score."""
    rows = read_jsonl(SELECTION_DIR / f"per_prompt_candidates_{corpus}_{src.CORPUS_EMBEDDING[corpus]}.jsonl")
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[length_bin(r["n_tokens"])].append(r["max_sim"])
    means = {b: statistics.mean(v) for b, v in by_bin.items()}
    return {str(r["id"]): {"selection_max_sim": r["max_sim"],
                           "selection_score": round(r["max_sim"] - means[length_bin(r["n_tokens"])], 4)}
            for r in rows}


def survives(c: dict) -> bool:
    return (
        c["group"] in bps.KEPT_GROUPS
        and not any(c["extra_occurrences"].values())
        and c["p_target"] >= src.LANGID_MIN_P_TARGET
        and not TRUNCATED_END.search(c["text"])
    )


def select(corpus: str, args: argparse.Namespace) -> None:
    traced = read_jsonl(WORK_DIR / f"traced_{corpus}.jsonl")
    report = json.loads((WORK_DIR / f"candidates_report_{corpus}.json").read_text())
    scores = selection_scores(corpus)
    langid = {r["id"]: r for r in read_jsonl(WORK_DIR / f"langid_{corpus}.jsonl")}
    for c in traced:
        c.update(scores[c["id"]])
        c["p_target"] = langid[c["id"]]["p_target"]
    sources = sorted(report)
    alive = defaultdict(list)
    for c in traced:
        if survives(c):
            alive[c["source"]].append(c)
    alloc = water_fill({s: len(alive[s]) for s in sources}, src.FINAL_SIZE, random.Random(f"{args.seed}-select-{corpus}"))
    chosen = []
    for s in sources:
        ranked = sorted(alive[s], key=lambda c: (-c["selection_score"], c["id"]))
        chosen += ranked[: alloc[s]]
    chosen.sort(key=lambda c: (c["source"], c["id"]))

    rows = [{
        "text": c["text"], "domain": c["domain"], "id": c["id"], "source": c["source"], "lang": c["lang"],
        "group": c["group"], "coverage_tokens": c["coverage_tokens"], "coverage_chars": c["coverage_chars"],
        "n_tokens": c["n_tokens"], "lang_p": c["p_target"], "selection_max_sim": c["selection_max_sim"],
        "selection_score": c["selection_score"], **{k: c.get(k) for k in PROVENANCE},
    } for c in chosen]
    bps.write_jsonl(rt.prompt_set_path(f"specific_{corpus}"), rows)

    per_source = {}
    for s in sources:
        mine = [c for c in traced if c["source"] == s]
        picked = [c for c in chosen if c["source"] == s]
        per_source[s] = {
            **report[s],
            "passing": len(alive[s]),
            "selected": alloc[s],
            "selection_max_sim_survivors": round(statistics.mean(c["selection_max_sim"] for c in alive[s]), 4)
            if alive[s] else None,
            "selection_max_sim_selected": round(statistics.mean(c["selection_max_sim"] for c in picked), 4)
            if picked else None,
            "full_matches": sum(c["group"] not in bps.KEPT_GROUPS or any(c["extra_occurrences"].values())
                                for c in mine),
            "other_language": sum(c["p_target"] < src.LANGID_MIN_P_TARGET for c in mine),
            "truncated": sum(bool(TRUNCATED_END.search(c["text"])) for c in mine),
        }
    selection = {
        "corpus": corpus,
        "index": src.CORPUS_INDEX[corpus],
        "extra_indexes": list(src.CORPUS_EXTRA_INDEXES[corpus]),
        "size": len(chosen),
        "candidates": len(traced),
        "passing": sum(map(len, alive.values())),
        "per_source": per_source,
        "stats": bps.set_stats(f"unseen_{corpus}", f"candidates_{corpus}", src.CORPUS_INDEX[corpus], rows,
                               excluded=None),
    }
    (WORK_DIR / f"selection_{corpus}.json").write_text(json.dumps(selection, indent=2))
    print(f"{corpus}: {len(chosen)} prompts; per source {dict((s, alloc[s]) for s in sources)}")


def verify(corpus: str, args: argparse.Namespace, enc) -> None:
    rows = read_jsonl(rt.prompt_set_path(f"specific_{corpus}"))
    selection = json.loads((WORK_DIR / f"selection_{corpus}.json").read_text())
    assert len(rows) == selection["size"] == src.FINAL_SIZE, (len(rows), selection["size"])
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    assert len({r["text"] for r in rows}) == len(rows), "duplicate texts"
    counts = Counter(r["source"] for r in rows)
    assert all(counts[s] == v["selected"] for s, v in selection["per_source"].items()), counts
    assert all(r["group"] in bps.KEPT_GROUPS for r in rows)
    texts = [r["text"] for r in rows]
    for index_name in (src.CORPUS_INDEX[corpus], *src.CORPUS_EXTRA_INDEXES[corpus]):
        engine = gbo.open_engine(index_name, enc, args.max_page_table_gb)
        with ThreadPoolExecutor(args.threads) as pool:
            occ = list(tqdm(pool.map(lambda t: gbo.count_occurrences(t, engine, enc), texts), total=len(texts),
                            desc=f"Verify {corpus} vs {index_name}", unit="text"))
        found = [t for t, n in zip(texts, occ) if n]
        assert not found, f"{len(found)} prompts occur verbatim in {index_name}: {found[:3]}"
        print(f"{corpus}: 0 of {len(texts)} prompts occur verbatim in {index_name}")
    print(f"{corpus}: verified {len(rows)} prompts ({dict(counts)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=("select", "verify"))
    parser.add_argument("--corpora", nargs="+", choices=list(src.CORPUS_SOURCES), default=list(src.CORPUS_SOURCES))
    parser.add_argument("--seed", type=int, default=src.SEED)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-page-table-gb", type=float, default=2.0)
    args = parser.parse_args()
    enc = None
    for step in args.steps:
        for corpus in args.corpora:
            if step == "select":
                select(corpus, args)
            else:
                if enc is None:
                    from transformers import AutoTokenizer

                    enc = AutoTokenizer.from_pretrained(rt.TOKENIZER_MODEL, add_bos_token=False, add_eos_token=False)
                verify(corpus, args, enc)


if __name__ == "__main__":
    main()

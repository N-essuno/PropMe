"""SimpleTrace `mixed` mode in detail, from the warm1 runs of the comparison.

For every input group it reports:
  - full-text detection: generations whose whole text is found in the index
    (match_tier exact_full_raw / exact_full_normalized), checked against a
    direct FIND of the full token sequence (--check-index);
  - how many distinct training documents contain the full text (<= docs-per-span);
  - NV-recall per generation;
  - the longest span per generation and per-generation agreement with OLMoTrace
    (Spearman rank correlation of longest span and coverage);
  - where the tokens highlighted by only one of mixed / OLMoTrace come from;
  - for the Dynaword generations, all of the above per prompt setting
    (generic / specific / prefix).

python olmotrace/comparison/mixed_analysis.py --check-index \
    > olmotrace/comparison/outputs/mixed_analysis.json
"""

import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMP = os.path.join(REPO, "olmotrace", "comparison")
E2E = os.path.join(CMP, "outputs", "e2e")
IDX = os.path.join(os.environ.get("PROPME_DATA_ROOT", os.path.join(REPO, "propme_data")), "indexes")
GROUPS = {
    "dynaword__dynaword_generations": f"{IDX}/dynaword_index",
    "dynaword__dynaword_verbatim": f"{IDX}/dynaword_index",
    "commonpile__commonpile_verbatim": f"{IDX}/commonpile_index/common_pile_train_index",
    "commonpile__english_novel": f"{IDX}/commonpile_index/common_pile_train_index",
    "dolma3__dolma3_verbatim": f"{IDX}/dolma3_index_link",
    "dolma3__english_novel": f"{IDX}/dolma3_index_link",
}
FULL_TIERS = ("exact_full_raw", "exact_full_normalized")
MIXED_MIN_SPAN_TOKENS = 4  # simple_trace.MIXED_MIN_SPAN_TOKENS


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else 0.0


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else None


def covered(spans):
    return {i for s in spans for i in range(s["start"], s["end"])}


def load_group(group):
    inputs = load_jsonl(os.path.join(CMP, "inputs", f"{group.split('__')[1]}.jsonl"))
    out = {}
    for tool in ("ot", "st_text", "st_mixed"):
        rows = load_jsonl(os.path.join(E2E, f"{group}__warm1__{tool}.results.jsonl"))
        out[tool] = sorted(rows, key=lambda r: r["index"])
    return inputs, out


def is_bow_flags(token_info, ids):
    return token_info.begin_of_word_flags(ids)


def describe(idx, inputs, res, enc, token_info, full_in_index):
    """Statistics for the generations at positions idx."""
    ot = [res["ot"][i] for i in idx]
    tx = [res["st_text"][i] for i in idx]
    mx = [res["st_mixed"][i] for i in idx]
    n = len(idx)

    full_gens, norm_gens, full_doc_counts, max_nv = 0, 0, [], []
    tiers = {"exact_full_raw": 0, "exact_full_normalized": 0, "partial": 0}
    for r in mx:
        docs = [d for s in r["spans"] for d in s["docs"]]
        for d in docs:
            tiers[d["match_tier"]] += 1
        full_docs = {d["id"] for d in docs if d["match_tier"] in FULL_TIERS}
        if any(d["match_tier"] == "exact_full_raw" for d in docs):
            full_gens += 1
        if any(d["match_tier"] == "exact_full_normalized" for d in docs):
            norm_gens += 1
        if full_docs:
            full_doc_counts.append(len(full_docs))
        max_nv.append(max((d["nv_recall"] for d in docs), default=0.0))

    longest = {t: [max((s["span_length"] for s in r["spans"]), default=0) for r in rows]
               for t, rows in (("ot", ot), ("st_text", tx), ("st_mixed", mx))}
    cov = {t: [len(covered(r["spans"])) / max(1, len(enc.encode(r["generation"]))) for r in rows]
           for t, rows in (("ot", ot), ("st_text", tx), ("st_mixed", mx))}

    # Where single-tool tokens come from (mixed vs OLMoTrace).
    only_mixed = {"mixed span crosses a sentence end": 0, "mixed span starts inside a word": 0,
                  "other (different spans kept in step 2 / merging)": 0}
    only_ot = {"OLMoTrace span shorter than 4 tokens": 0, "step 2 kept different spans / merging": 0}
    jac_mt, jac_mo = [], []
    for o, t, m in zip(ot, tx, mx):
        ids = enc.encode(m["generation"])
        bow = is_bow_flags(token_info, ids)
        co, ct, cm = covered(o["spans"]), covered(t["spans"]), covered(m["spans"])
        for u, a, lst in ((cm | ct, cm & ct, jac_mt), (cm | co, cm & co, jac_mo)):
            if u:
                lst.append(len(a) / len(u))
        # Each mixed-only token counted once, by its most specific cause over the spans containing it.
        cause = {}
        for s in m["spans"]:
            if any(c in "!.?\n" for c in s["text"].rstrip()[:-1]):
                c = 0
            elif s["start"] < len(bow) and not bow[s["start"]]:
                c = 1
            else:
                c = 2
            for i in range(s["start"], s["end"]):
                if i not in co:
                    cause[i] = min(cause.get(i, 2), c)
        keys = list(only_mixed)
        for c in cause.values():
            only_mixed[keys[c]] += 1
        for s in o["spans"]:
            extra = [i for i in range(s["start"], s["end"]) if i not in cm]
            if not extra:
                continue
            key = ("OLMoTrace span shorter than 4 tokens" if s["span_length"] < MIXED_MIN_SPAN_TOKENS
                   else "step 2 kept different spans / merging")
            only_ot[key] += len(extra)

    def share(d):
        tot = sum(d.values())
        return {"total_tokens": tot, **{k: round(v / tot, 3) if tot else 0.0 for k, v in d.items()}}

    entry = {
        "generations": n,
        "full_text_found_exact": full_gens,
        "full_text_found_normalized_only": norm_gens,
        "docs_by_match_tier": tiers,
        "distinct_docs_with_full_text": {
            "1": sum(c == 1 for c in full_doc_counts),
            "2-9": sum(2 <= c <= 9 for c in full_doc_counts),
            ">=10 (sample cap)": sum(c >= 10 for c in full_doc_counts),
        },
        "gens_max_nv_recall_gt_0.5": sum(v > 0.5 for v in max_nv),
        "gens_max_nv_recall_ge_0.9": sum(v >= 0.9 for v in max_nv),
        "mean_max_nv_recall": mean(max_nv),
        "coverage": {t: mean(v) for t, v in cov.items()},
        "longest_span_mean": {t: mean(v) for t, v in longest.items()},
        "gens_longest_span_ge_20": {t: sum(x >= 20 for x in v) for t, v in longest.items()},
        "gens_longest_span_ge_60": {t: sum(x >= 60 for x in v) for t, v in longest.items()},
        "spearman_vs_ot": {
            "longest_span_mixed": spearman(longest["st_mixed"], longest["ot"]),
            "longest_span_text": spearman(longest["st_text"], longest["ot"]),
            "coverage_mixed": spearman(cov["st_mixed"], cov["ot"]),
            "coverage_text": spearman(cov["st_text"], cov["ot"]),
        },
        "jaccard_mixed_vs_ot": mean(jac_mo),
        "jaccard_mixed_vs_text": mean(jac_mt),
        "tokens_only_mixed": share(only_mixed),
        "tokens_only_ot": share(only_ot),
    }
    if full_in_index is not None:
        truth = [full_in_index[i] for i in idx]
        pred = [any(d["match_tier"] == "exact_full_raw" for s in m["spans"] for d in s["docs"]) for m in mx]
        entry["full_text_check_vs_index"] = {
            "in_index": sum(truth),
            "found_by_mixed_and_in_index": sum(a and b for a, b in zip(pred, truth)),
            "found_by_mixed_not_in_index": sum(a and not b for a, b in zip(pred, truth)),
            "in_index_missed_by_mixed": sum(b and not a for a, b in zip(pred, truth)),
        }
    return entry


def check_index(group, inputs, enc):
    """FIND of each input's full token sequence (single process, small page-table footprint)."""
    sys.path.insert(0, os.path.join(REPO, "03_tracing"))
    import simple_trace as st
    engine = st.PageTableBoundedEngine(GROUPS[group], enc.eos_token_id, 1e9)
    return [engine.find(input_ids=enc.encode(x["text"]))["cnt"] > 0 for x in inputs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-index", action="store_true")
    a = p.parse_args()
    sys.path.insert(0, os.path.join(REPO, "olmotrace"))
    import olmo_trace as ot
    enc = ot.load_tokenizer()
    token_info = ot.TokenInfo(enc)

    report = {}
    for group in GROUPS:
        inputs, res = load_group(group)
        full_in_index = check_index(group, inputs, enc) if a.check_index else None
        report[group] = {"all": describe(list(range(len(inputs))), inputs, res, enc, token_info, full_in_index)}
        if "setting" in inputs[0]:
            for setting in ("generic", "specific", "prefix"):
                idx = [i for i, x in enumerate(inputs) if x["setting"] == setting]
                report[group][setting] = describe(idx, inputs, res, enc, token_info, full_in_index)
        summary = json.load(open(os.path.join(E2E, f"{group}__warm1__st_mixed.summary.json")))
        report[group]["simpletrace_summary"] = {k: summary[k] for k in (
            "generations_full_matches_ratio", "unique_full_matches", "avg_nv_recall", "avg_nv_recall_on_hits",
            "generations_above_nv_recall_threshold_ratio", "generations_with_n_token_span_ratio",
            "average_longest_span_length", "unique_total_docs", "total_docs")}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

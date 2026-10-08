"""Compare the outputs and resource usage of the end-to-end runs.

Output agreement uses the warm1 runs (full input sets). Both tools use the
same Llama-2 tokenizer without BOS/EOS, so span token offsets are directly
comparable.

python olmotrace/comparison/analyze.py > olmotrace/comparison/outputs/analysis.json
"""

import json
import os
import statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMP = os.path.join(REPO, "olmotrace", "comparison")
E2E = os.path.join(CMP, "outputs", "e2e")
TOOLS = ["ot", "st_text", "st_mixed"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else 0.0


def token_set(spans):
    out = set()
    for sp in spans:
        out.update(range(sp["start"], sp["end"]))
    return out


def per_gen(tool, row, n_tokens):
    """Normalize one result line to spans / docs for either tool."""
    if tool == "ot":
        spans = [(s["start"], s["end"]) for s in row["spans"]]
        docs = {(d["shard"], d["doc_ix"]) for d in row["documents"]}
        doc_keys = {str(d["doc_ix"]) for d in row["documents"]} | {str(d["id"]) for d in row["documents"]}
        contains = []
        for d in row["documents"]:
            contexts = [s["context"] for s in d["snippets"]]
            for fi in d["filtered_span_indices"]:
                text = row["filtered_spans"][fi]["text"].strip()
                contains.append(any(text in c for c in contexts))
    else:
        spans = [(s["start"], s["end"]) for s in row["spans"]]
        docs = {str(d["id"]) for s in row["spans"] for d in s["docs"]}
        doc_keys = docs
        contains = [s["text"].strip() in d["text"] for s in row["spans"] for d in s["docs"]]
    spans_d = [{"start": b, "end": e} for b, e in spans]
    covered = token_set(spans_d)
    # Sentence-boundary characters inside a span (not as its last character).
    span_texts = [s["text"].rstrip() for s in row["spans"]]
    return {
        "inner_boundary": [any(c in "!.?\n" for c in t[:-1]) for t in span_texts],
        "spans": sorted(spans),
        "covered": covered,
        "coverage": len(covered) / n_tokens if n_tokens else 0.0,
        "num_docs": len(docs),
        "doc_keys": doc_keys,
        "span_lengths": [e - b for b, e in spans],
        "doc_contains_span": contains,
    }


def compare_outputs(group):
    inputs = load_jsonl(os.path.join(CMP, "inputs", f"{group.split('__')[1]}.jsonl"))
    res = {}
    for tool in TOOLS:
        path = os.path.join(E2E, f"{group}__warm1__{tool}.results.jsonl")
        if not os.path.exists(path):
            return None
        rows = load_jsonl(path)
        rows.sort(key=lambda r: r["index"])
        assert len(rows) == len(inputs), (path, len(rows), len(inputs))
        res[tool] = rows
    n_tokens = [r["num_tokens"] for r in res["ot"]]
    norm = {t: [per_gen(t, row, n) for row, n in zip(res[t], n_tokens)] for t in TOOLS}

    out = {"generations": len(inputs), "avg_tokens": mean(n_tokens), "tools": {}}
    for t in TOOLS:
        g = norm[t]
        lens = [x for r in g for x in r["span_lengths"]]
        contains = [c for r in g for c in r["doc_contains_span"]]
        entry = {
            "coverage": mean(r["coverage"] for r in g),
            "gens_with_spans": sum(bool(r["spans"]) for r in g),
            "spans_per_gen": mean(len(r["spans"]) for r in g),
            "span_len_mean": mean(lens),
            "span_len_median": statistics.median(lens) if lens else 0,
            "span_len_max": max(lens, default=0),
            "docs_per_gen": mean(r["num_docs"] for r in g),
            "doc_contains_span_rate": mean(contains) if contains else None,
            "spans_with_inner_boundary_char": mean(x for r in g for x in r["inner_boundary"]),
        }
        if t != "ot":
            jac, same = [], 0
            for a, b in zip(norm["ot"], g):
                union = a["covered"] | b["covered"]
                if union:
                    jac.append(len(a["covered"] & b["covered"]) / len(union))
                same += a["spans"] == b["spans"]
            entry["token_jaccard_vs_ot"] = mean(jac)
            entry["identical_spans_vs_ot"] = same / len(g)
            entry["tokens_only_ot"] = sum(len(a["covered"] - b["covered"]) for a, b in zip(norm["ot"], g))
            entry["tokens_only_this"] = sum(len(b["covered"] - a["covered"]) for a, b in zip(norm["ot"], g))
        if "source_id" in inputs[0]:
            hits = [str(i["source_id"]) in r["doc_keys"] or str(i["source_doc_ix"]) in r["doc_keys"]
                    for i, r in zip(inputs, g)]
            entry["source_doc_recall"] = mean(hits)
        out["tools"][t] = entry

    # OLMoTrace-only outputs (relevance, step latency) for context.
    ot_rows = res["ot"]
    docs = [d for r in ot_rows for d in r["documents"]]
    lat = defaultdict(list)
    for r in ot_rows:
        for k, v in r["latency_seconds"].items():
            lat[k].append(v)
    out["ot_extra"] = {
        "doc_relevance": {lvl: mean(d["relevance"] == lvl for d in docs) for lvl in ("high", "medium", "low")},
        "latency_share_step1": sum(lat["step1_maximal_spans"]) / sum(lat["total"]) if lat["total"] else None,
        "latency_share_step3": sum(lat["step3_retrieve"]) / sum(lat["total"]) if lat["total"] else None,
        "latency_share_step4_5": sum(lat["step4_5_merge_rerank"]) / sum(lat["total"]) if lat["total"] else None,
    }
    st_docs = [d for r in res["st_mixed"] for s in r["spans"] for d in s["docs"]]
    out["st_mixed_extra"] = {
        "full_exact_match_gens": sum(any(d["match_tier"] == "exact_full_raw" for s in r["spans"] for d in s["docs"])
                                     for r in res["st_mixed"]),
        "avg_nv_recall": mean(d["nv_recall"] for d in st_docs),
    }
    return out


def timing():
    runs = load_jsonl(os.path.join(E2E, "runs.jsonl"))
    runs = [r for r in runs if not r["tag"].startswith("pilot") and r["returncode"] == 0]
    by_group = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        parts = r["tag"].split("__")
        if len(parts) != 3 or not parts[2].startswith(("cold", "warm")):
            continue  # warm-up and sensitivity runs (reported separately)
        corpus, name, phase = parts
        by_group[f"{corpus}__{name}"][phase][r["tool"]] = r
    out = {}
    for group, phases in by_group.items():
        g = {}
        for t in TOOLS:
            cold = [(ph, p[t]) for ph, p in phases.items() if ph.startswith("cold") and t in p]
            # Position of the tool within its slice (Latin square: s0 ot first, s1 st_text, s2 st_mixed).
            first = {"ot": "cold_s0", "st_text": "cold_s1", "st_mixed": "cold_s2"}[t]
            warm = [p[t] for ph, p in phases.items() if ph.startswith("warm") and t in p]
            g[t] = {
                "cold_first_trace_s": phases.get(first, {}).get(t, {}).get("trace_elapsed_s"),
                "cold_first_slice": first,
                "cold_slices_trace_s": {ph: r["trace_elapsed_s"] for ph, r in sorted(cold)},
                "warm_trace_s": [r["trace_elapsed_s"] for r in warm],
                "warm_wall_s": [r["wall_s"] for r in warm],
                "warm_cpu_s": [r["cpu_s"] for r in warm],
                "peak_rss_anon_gb": max((r["peak_rss_anon_gb"] for _, r in cold + [(0, w) for w in warm]), default=None),
                "peak_page_tables_gb": max((r["peak_page_tables_gb"] for _, r in cold + [(0, w) for w in warm]), default=None),
                "read_gb_cold_first": phases.get(first, {}).get(t, {}).get("read_gb"),
            }
        out[group] = g
    return out


def main():
    groups = sorted({"__".join(f.split("__")[:2]) for f in os.listdir(E2E)
                     if f.endswith(".results.jsonl") and len(f.split("__")) == 4})
    report = {"outputs": {}, "timing": timing()}
    for g in groups:
        r = compare_outputs(g)
        if r:
            report["outputs"][g] = r
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

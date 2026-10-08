"""Test 1: the shared step-1 primitive of both tools on a real index.

For every start position of every input text, compute the length of the
longest prefix of the suffix that occurs in the index with
  - OLMoTrace's get_longest_prefix_len (one FIND + both SA neighbours, LCP at the
    exact neighbour position), and
  - SimpleTrace's longest_indexed_prefix_len (one FIND + both SA neighbours, prefix
    searched anywhere in a 5*L-token window around each neighbour),
on the same engine object, and check each answer against the definition:
  valid(n):   n == 0 or count(suffix[:n]) > 0
  maximal(n): n == len(suffix) or count(suffix[:n+1]) == 0

It also times each tool's full step-1 lookup loop over one text (single thread):
OLMoTrace queries every begin-of-word start independently; SimpleTrace's
longest_prefix_lens reuses the previous start's match as a lower bound.
The call order of the two tools alternates between texts so neither one
systematically runs on pages warmed by the other.

python olmotrace/comparison/step1_compare.py --index-dir <dirs...> \
    --inputs olmotrace/comparison/inputs/english_novel.jsonl --limit 50 \
    --processes 8 --output olmotrace/comparison/outputs/step1_<name>.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "olmotrace"))
sys.path.insert(0, os.path.join(REPO, "03_tracing"))
import olmo_trace as ot  # noqa: E402
import simple_trace as st  # noqa: E402

_state = {}


def _init(index_dir):
    enc = ot.load_tokenizer()
    engine = ot.load_engine(index_dir, enc.eos_token_id)  # same flags as st.load_engine
    n = engine.engine.get_num_shards()
    _state.update(enc=enc, engine=engine, token_info=ot.TokenInfo(enc),
                  tok_cnt=[engine.engine.get_tok_cnt(s=s) for s in range(n)])


def _count(engine, ids) -> int:
    return engine.find(input_ids=ids)["cnt"] if ids else 1


def _check(engine, suffix, n) -> dict:
    valid = n == 0 or _count(engine, suffix[:n]) > 0
    maximal = n == len(suffix) or _count(engine, suffix[: n + 1]) == 0
    return {"valid": valid, "maximal": maximal}


def _process_text(job):
    idx, text = job
    enc, engine, ti, tok_cnt = _state["enc"], _state["engine"], _state["token_info"], _state["tok_cnt"]
    ids = enc.encode(text)
    L = len(ids)
    max_doc_toks = 5 * L  # as in SimpleTrace.trace_generation
    # SimpleTrace text-mode starts ("▁" tokens, excluding the last position).
    st_starts = [b for b in range(L - 1) if enc.convert_ids_to_tokens(ids[b])[0] == "▁"]
    ot_first = idx % 2 == 0

    per_suffix = []
    for b in st_starts:
        suffix = ids[b:]
        calls = [("ot", lambda: ot.get_longest_prefix_len(engine, suffix, tok_cnt)),
                 ("st", lambda: st.longest_indexed_prefix_len(engine, suffix, max_doc_toks))]
        if not ot_first:
            calls.reverse()
        rec = {"start": b, "suffix_len": len(suffix)}
        for name, fn in calls:
            t = time.perf_counter()
            rec[name] = fn()
            rec[f"{name}_s"] = time.perf_counter() - t
        for name in ("ot", "st"):
            rec.update({f"{name}_{k}": v for k, v in _check(engine, suffix, rec[name]).items()})
        per_suffix.append(rec)

    # Full step-1 lookup loops, single thread, two alternating repetitions.
    def ot_loop():
        return ot.get_maximal_matching_spans(engine, ids, ti, tok_cnt, executor=None)

    def st_loop():
        return st.longest_prefix_lens(engine, ids, st_starts, max_doc_toks, executor=None)

    def ot_loop_st_starts():
        return [ot.get_longest_prefix_len(engine, ids[b:], tok_cnt) for b in st_starts]

    loops = {}
    order = [("ot_step1", ot_loop), ("st_step1", st_loop), ("ot_lookups_on_st_starts", ot_loop_st_starts)]
    if not ot_first:
        order = [order[1], order[0], order[2]]
    for rep in (1, 2):
        for name, fn in order:
            t = time.perf_counter()
            out = fn()
            loops[f"{name}_rep{rep}_s"] = time.perf_counter() - t
            if name == "st_step1" and rep == 1:
                st_lens = out
    ot_bow_starts = sum(ti.begin_of_word_flags(ids))
    return {
        "index": idx,
        "num_tokens": L,
        "num_st_starts": len(st_starts),
        "num_ot_starts": ot_bow_starts,
        "st_loop_equals_single_calls": st_lens == [r["st"] for r in per_suffix],
        "per_suffix": per_suffix,
        **loops,
    }


def summarize(texts: list[dict]) -> dict:
    sfx = [r for t in texts for r in t["per_suffix"]]
    n = len(sfx)

    def rate(pred):
        return sum(1 for r in sfx if pred(r)) / n if n else 0.0

    no_match = [r for r in sfx if r["ot"] < r["suffix_len"]]
    out = {
        "texts": len(texts),
        "suffixes": n,
        "suffixes_without_full_match": len(no_match),
        "agree_rate": rate(lambda r: r["ot"] == r["st"]),
        "ot_correct_rate": rate(lambda r: r["ot_valid"] and r["ot_maximal"]),
        "st_correct_rate": rate(lambda r: r["st_valid"] and r["st_maximal"]),
        "ot_shorter": sum(r["ot"] < r["st"] for r in sfx),
        "st_shorter": sum(r["st"] < r["ot"] for r in sfx),
        "st_loop_equals_single_calls_all": all(t["st_loop_equals_single_calls"] for t in texts),
        "mean_match_len": statistics.mean(r["ot"] for r in sfx) if sfx else 0,
        "per_call_ms": {
            "ot_mean": 1e3 * statistics.mean(r["ot_s"] for r in sfx),
            "st_mean": 1e3 * statistics.mean(r["st_s"] for r in sfx),
            "ot_median": 1e3 * statistics.median(r["ot_s"] for r in sfx),
            "st_median": 1e3 * statistics.median(r["st_s"] for r in sfx),
            "ot_mean_no_full_match": 1e3 * statistics.mean(r["ot_s"] for r in no_match) if no_match else None,
            "st_mean_no_full_match": 1e3 * statistics.mean(r["st_s"] for r in no_match) if no_match else None,
        },
        "step1_loop_seconds_total": {
            k: sum(t[k] for t in texts)
            for k in texts[0] if k.endswith("_s") and "rep" in k
        },
        "starts_total": {"st": sum(t["num_st_starts"] for t in texts),
                         "ot": sum(t["num_ot_starts"] for t in texts)},
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", nargs="+", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with open(a.inputs) as f:
        texts = [json.loads(l)["text"] for l in f if l.strip()][: a.limit]
    index_dir = a.index_dir[0] if len(a.index_dir) == 1 else a.index_dir
    t0 = time.time()
    with Pool(a.processes, initializer=_init, initargs=(index_dir,)) as pool:
        results = sorted(pool.imap_unordered(_process_text, list(enumerate(texts))), key=lambda r: r["index"])
    summary = {"inputs": a.inputs, "index_dir": a.index_dir, "processes": a.processes,
               "wall_s": time.time() - t0, **summarize(results)}
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump({"summary": summary, "texts": results}, f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

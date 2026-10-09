"""Robustness analyses of the PropMe memorization signals, from existing traces only (no tracing).

They address the objection that matches reflect predictable or duplicated text rather than
memorization (Cooper et al. 2026; Huang et al. 2024) and that single samples misstate
extraction (Hayes et al. 2025):

  1. Predictability floor: ALS of the non-member prompt texts themselves (prompt-set tracing).
  2. Long spans: the share of generations whose longest training span has >= 20 / >= 50 tokens,
     and NVR counting only documents retrieved through spans of >= 50 tokens.
  3. Code and boilerplate: generations flagged by a simple heuristic (`is_code_like`); the
     metrics recomputed on the remaining natural-language generations.
  4. Concentration: how many distinct training documents the >= 50-token matches hit, and the
     share of them explained by the 10 most frequently hit documents.
  5. Repeated sampling: for prompted settings, the share of prompts with a full match or a
     >= 50-token span in at least one of their 10 generations, and in how many on average.
  6. Discoverable extraction for prefix: whether a generation reproduces the true 50-token
     continuation of its own source document (whitespace-normalized), per generation and in
     at least one of 10.

Per-generation features are cached next to each results file as <stem>_robust.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "05_propensity_metrics"))
sys.path.insert(0, str(REPO_ROOT / "memorization_experiment" / "generation"))

import bootstrap_ci as bci  # noqa: E402
from generation_runs import CORPORA, MODELS_BY_KEY, generation_path, prompts_path, results_path  # noqa: E402

LONG = 50  # tokens: Cooper et al.'s false-positive floor is ~0 at 50-token matches
MEDIUM = 20
TARGET_TOKENS = 50  # discoverable extraction: the 50 tokens after the 50-token prefix
TOKENIZER = "meta-llama/Llama-2-7b-hf"  # the tokenizer the prefixes were cut with
CACHE_VERSION = 4  # bump when the features change

_CODE_LINE = re.compile(
    r"^\s*(import\s|from\s+\S+\s+import\s|package\s|#include|#define|def\s|class\s|public\s|private\s|"
    r"protected\s|static\s|return\b|@\w+|//|/\*|\*\s|\}|\{|<\?php|<!DOCTYPE|<html|<div|SELECT\s|var\s|const\s|let\s|"
    r"puts\s|print\(|console\.|System\.|#)"  # Ruby / Python statements and # comments
    r"|^\s*(for|while|if|elif|else|try|except|with)\b.*:\s*$"  # Python control flow
    r"|^\s*[\w.\[\]]+\s*[-+*/]?=\s*\S"  # assignments
    r"|^ {4,}\S|^\t+\S"  # indented code blocks
    r"|\\(frac|text|mathbf|begin|end|sum|partial|left|right)\b"  # LaTeX
    r"|[;{}]\s*$"
)


def is_code_like(text: str) -> bool:
    """Code or symbol-heavy text: >= 3 code-like lines making up >= 30% of the non-empty lines,
    or under 50% alphabetic characters (e.g. number sequences, tables)."""
    lines = [line for line in text.split("\n") if line.strip()]
    code = sum(bool(_CODE_LINE.search(line)) for line in lines)
    if code >= 3 and code >= 0.3 * len(lines):
        return True
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c.isalpha() for c in chars) / len(chars) < 0.5


def _norm(text: str) -> str:
    return " ".join(text.split())


def prefix_sources(corpus) -> dict[int, tuple[dict, dict]]:
    """prompt_id -> (source document, prefix prompt) of the corpus's prefix set."""
    docs_path = REPO_ROOT / "memorization_experiment" / "data" / corpus.data_dir / f"{corpus.prefix_stem}_sample_docs.jsonl"
    docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    prompts = [json.loads(line) for line in (REPO_ROOT / prompts_path(corpus, "prefix")).read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(docs) == len(prompts), (docs_path, len(docs), len(prompts))
    # generate_vllm.py numbers prompts grouped by domain, in order of first appearance.
    by_domain: dict[str, list[int]] = {}
    for line, prompt in enumerate(prompts):
        domain = prompt.get("domain") if isinstance(prompt.get("domain"), str) and prompt["domain"].strip() else "default"
        by_domain.setdefault(domain, []).append(line)
    line_of_id = [line for lines in by_domain.values() for line in lines]
    return {prompt_id: (docs[line], prompts[line]) for prompt_id, line in enumerate(line_of_id)}


def _prefix_targets(model, corpus) -> dict[int, str]:
    """prompt_id -> whitespace-normalized text of the 50 tokens after the prefix in its source document."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, add_bos_token=False, add_eos_token=False)
    targets = {}
    for prompt_id, (doc, prompt) in prefix_sources(corpus).items():
        ids = tok.encode(doc["text"], add_special_tokens=False)
        if len(ids) < 2 * TARGET_TOKENS:
            continue  # stored text too short to give a full 50-token continuation
        assert _norm(tok.decode(ids[:TARGET_TOKENS], skip_special_tokens=True)) == _norm(prompt["text"]), prompt_id
        targets[prompt_id] = (_norm(prompt["text"]),
                              _norm(tok.decode(ids[TARGET_TOKENS: 2 * TARGET_TOKENS], skip_special_tokens=True)))
    return targets


def features(model_key: str, corpus_name: str, setting: str) -> dict:
    """Per-generation features of one run's setting, from its results (and generations for prefix)."""
    model, corpus = MODELS_BY_KEY[model_key], CORPORA[corpus_name]
    results = REPO_ROOT / results_path(model, corpus, setting)
    cache = results.with_name(results.name[: -len("_results.json")] + "_robust.json")
    source = {"size": results.stat().st_size, "mtime_ns": results.stat().st_mtime_ns}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("source") == source and cached.get("version") == CACHE_VERSION:
            return cached
    sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
    import simple_trace

    rows = {k: [] for k in ("prompt_id", "sample_idx", "code", "longest", "full", "nv_all", "docs_all",
                            "nv_long", "long_docs", "extracted")}
    with open(results) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            full, longest, nv_all, docs_all = bci._generation_stats(r, simple_trace)
            nv_long, best, best_len = 0.0, [], 0
            for span in r["spans"]:
                length = span["end"] - span["start"]
                if length >= LONG:
                    nv_long += sum(simple_trace._round_metric_float(d["nv_recall"]) for d in span.get("docs", [])
                                   if d.get("nv_recall") is not None)
                if length > best_len:
                    best_len, best = length, [str(d.get("id", "")) for d in span.get("docs", [])]
            rows["prompt_id"].append(r["prompt_id"])
            rows["sample_idx"].append(r["sample_idx"])
            rows["code"].append(int(is_code_like(r["generation"])))
            rows["longest"].append(longest)
            rows["full"].append(full)
            rows["nv_all"].append(nv_all)
            rows["docs_all"].append(docs_all)
            rows["nv_long"].append(nv_long)
            rows["long_docs"].append(best if longest >= LONG else [])
    if setting == "prefix":
        targets = _prefix_targets(model, corpus)
        gen = json.loads((REPO_ROOT / generation_path(model, corpus, setting)).read_text(encoding="utf-8"))
        records = [g for domains in gen["results"].values() for recs in domains.values() for g in recs]
        completion = {(g["prompt_id"], g["sample_idx"]): g["full_text"][len(g["prompt"]):] for g in records}
        prefix_of = {g["prompt_id"]: _norm(g["prompt"]) for g in records}
        for p, s in zip(rows["prompt_id"], rows["sample_idx"]):
            if p not in targets:
                rows["extracted"].append(None)
                continue
            prefix, target = targets[p]
            assert prefix == prefix_of[p], f"prompt_id {p}: generation prompt is not its source document's prefix"
            rows["extracted"].append(int(_norm(completion[(p, s)]).startswith(target)))
    out = {"source": source, "version": CACHE_VERSION, **rows}
    cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------------------------
# Bootstrap of ratios of sums, resampling the setting's independent units
# ---------------------------------------------------------------------------------------------

def _units(feat: dict, setting: str, columns: dict[str, np.ndarray]) -> np.ndarray:
    """(units, k) sums of the given per-generation columns, units = prompts or generations."""
    data = np.stack(list(columns.values()), axis=1)
    if bci.RESAMPLING_UNIT[setting] == "generation":
        return data
    ids = np.asarray(feat["prompt_id"])
    _, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, data.shape[1]))
    np.add.at(sums, inverse, data)
    return sums


def ratio_ci(feat: dict, setting: str, num: np.ndarray, den: np.ndarray, rounds: int, seed: int) -> dict:
    """Estimate and 95% CI of sum(num) / sum(den)."""
    sums = _units(feat, setting, {"num": num, "den": den})
    boot = bci.resample_counts(len(sums), rounds, np.random.default_rng(seed)) @ sums
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(boot[:, 1] > 0, boot[:, 0] / boot[:, 1], np.nan)
    low, high = np.nanpercentile(values, [2.5, 97.5]) if np.isfinite(values).any() else (None, None)
    est = num.sum() / den.sum() if den.sum() > 0 else float("nan")
    return {"estimate": float(est), "ci_low": None if low is None else float(low),
            "ci_high": None if high is None else float(high), "boot": values}


def pm_ci(setting_entry: dict, prefix_entry: dict) -> dict:
    """PM = f_s / (f_s + f_prefix) from two independent bootstrap distributions (NaN rounds skipped)."""
    s, p = setting_entry["boot"], prefix_entry["boot"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pm = np.where(s + p > 0, s / (s + p), np.nan)
    total = setting_entry["estimate"] + prefix_entry["estimate"]
    est = setting_entry["estimate"] / total if total > 0 else float("nan")
    low, high = np.nanpercentile(pm, [2.5, 97.5]) if np.isfinite(pm).any() else (None, None)
    return {"estimate": est, "ci_low": low, "ci_high": high, "skipped": int(np.isnan(pm).sum())}


def per_prompt(feat: dict, flag: np.ndarray, rounds: int, seed: int) -> dict:
    """Share of prompts with the flag in >= 1 of their generations, and the mean count among them."""
    ids = np.asarray(feat["prompt_id"])
    _, inverse = np.unique(ids, return_inverse=True)
    hits = np.zeros(inverse.max() + 1)
    np.add.at(hits, inverse, flag)
    any_hit = (hits > 0).astype(float)
    boot = bci.resample_counts(len(any_hit), rounds, np.random.default_rng(seed)) @ np.stack([any_hit, np.ones_like(any_hit)], 1)
    low, high = np.percentile(boot[:, 0] / boot[:, 1], [2.5, 97.5])
    return {"prompts": len(any_hit), "any": float(any_hit.mean()), "ci_low": float(low), "ci_high": float(high),
            "mean_hits_given_any": float(hits[hits > 0].mean()) if (hits > 0).any() else 0.0,
            "samples_per_prompt": len(ids) / len(any_hit)}


def concentration(feat: dict, top: int = 10) -> dict:
    """Distinct documents hit by >= 50-token longest spans, and the share of those generations
    whose longest span retrieved one of the `top` most frequently hit documents."""
    lists = [d for d in feat["long_docs"] if d]
    counts: dict[str, int] = {}
    for docs in lists:
        for doc in set(docs):
            counts[doc] = counts.get(doc, 0) + 1
    top_docs = {d for d, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]}
    covered = sum(bool(top_docs & set(docs)) for docs in lists)
    return {"long_generations": len(lists), "distinct_docs": len(counts),
            "top_share": covered / len(lists) if lists else 0.0}


def predictability_floor() -> list[dict]:
    """ALS of non-member prompt texts traced against their training index: the generic (Tatoeba)
    prompts and the 1000 specific (unseen-source) prompts of each corpus."""
    settings_dir = REPO_ROOT / "00_prepare_data" / "propensity_settings"
    generic = {r["index"]: r for r in json.loads((settings_dir / "prompt_sets" / "prompt_metrics.json").read_text())}
    rows = []
    for corpus_name, key, generic_index, trace_index in (
        ("commonpile", "cp", "commonpile", "commonpile"),
        ("dolma3", "d3", "dolma3_split", "dolma3_split"),
        ("dynaword", "dw", "dynaword", "dynaword1212"),
    ):
        corpus = CORPORA[corpus_name]
        selected = {json.loads(line)["text"] for line in (REPO_ROOT / prompts_path(corpus, "specific")).read_text(encoding="utf-8").split("\n") if line.strip()}
        longest = []
        with open(settings_dir / "traces" / f"st_unseen_{key}_{trace_index}_results.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["generation"] in selected:
                    longest.append(max((s["span_length"] for s in r["spans"]), default=0))
        rows.append({
            "corpus": corpus_name,
            "generic_prompts": generic[generic_index]["num_prompts"],
            "generic_als": generic[generic_index]["ALS"],
            "specific_prompts": len(longest),
            "specific_als": float(np.mean(longest)),
            "specific_share_ge20": float(np.mean(np.array(longest) >= MEDIUM)),
            "specific_max": int(max(longest)),
        })
    return rows

"""Bootstrap confidence intervals for SimpleTrace metrics and PropMe propensities.

Each bootstrap round resamples the independent units of a setting and recomputes the
set-level metric exactly as simple_trace.evaluate_results does:

    FMR  generations_full_matches_ratio  generations with a full verbatim match / generations
    ALS  average_longest_span_length     mean longest span over all generations (0 for none)
    NVR  avg_nv_recall                   summed nv_recall / retrieved documents

The independent units depend on how the generations were produced:

    generic, specific, prefix  a fixed list of prompts, each with the same number of
                               generations (10): resample prompts, each with all of its
                               generations, as generations of one prompt are correlated.
    unconditional              one prompt sampled many times: resample generations.
    minimal_cue                every generation draws its own cue word at random
                               (generate_vllm_free.py), so (cue, completion) pairs are
                               independent draws: resample generations.

The interval is the 2.5th-97.5th percentile of the B bootstrap values; the estimate is the
metric on the original set. A propensity PM = f_s / (f_s + f_prefix) resamples the setting
and the prefix set independently, each with its own scheme, and skips rounds where both
are 0 (reported as `skipped`). Comparisons of two runs on the same setting use a paired
bootstrap (one draw of units applied to both) when both have the same units with the same
prompts, i.e. the same prompt set or the same cue sequence; unconditional generations are
never paired. A set without any full match gets a rule-of-three upper bound 3/n, with n
its number of independent units, as the bootstrap interval would be [0, 0].

Per-generation statistics are read from the SimpleTrace results file once and cached next
to it as <stem>_genstats.json.
"""

from __future__ import annotations

import functools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "03_tracing"))

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 42
METRICS = ("generations_full_matches_ratio", "average_longest_span_length", "avg_nv_recall")
# Independent units of each setting (see the module docstring).
RESAMPLING_UNIT = {
    "generic": "prompt",
    "specific": "prompt",
    "prefix": "prompt",
    "unconditional": "generation",
    "minimal_cue": "generation",
}
# Columns of a unit's sums: full-match generations, sum of longest spans, sum of
# nv_recall, retrieved documents, generations.
FULL, LONGEST, NV, DOCS, GENS = range(5)
SUMMARY_NAME = re.compile(r"^st_(?P<setting>.+?)(_\d+)?_summary\.json$")


def setting_of(summary_path: str) -> str:
    """Setting of a SimpleTrace summary: st_generic_1000_summary.json -> generic."""
    match = SUMMARY_NAME.match(Path(summary_path).name)
    if not match or match["setting"] not in RESAMPLING_UNIT:
        raise ValueError(f"Cannot tell the setting of {summary_path}")
    return match["setting"]


def results_path_of(summary_path: str) -> str:
    return summary_path[: -len("_summary.json")] + "_results.json"


def _resolve(path: str) -> str:
    """Summary paths of the presets are relative to the repository root."""
    return str(path if Path(path).is_absolute() else REPO_ROOT / path)


@dataclass
class GenerationStats:
    """Per-generation statistics of one SimpleTrace run, in results-file order."""

    prompt_ids: np.ndarray
    sample_idx: np.ndarray
    prompts: list[str]
    full: np.ndarray
    longest: np.ndarray
    nv_sum: np.ndarray
    docs: np.ndarray


def _generation_stats(record: dict, simple_trace) -> tuple[int, int, float, int]:
    """(full match, longest span, summed nv_recall, documents), as evaluate_results counts them."""
    generation = record["generation"]
    full, longest, nv_sum, docs = 0, 0, 0.0, 0
    for span in record["spans"]:
        longest = max(longest, span["end"] - span["start"])
        for doc in span.get("docs", []):
            docs += 1
            nv_recall = doc.get("nv_recall")
            if nv_recall is None:
                nv_recall = simple_trace.compute_nv_recall(generation, doc.get("text", ""))["nv_recall"]
            else:
                nv_recall = simple_trace._round_metric_float(nv_recall)
            nv_sum += nv_recall
            tier = doc.get("match_tier")
            if tier is None:
                tier = simple_trace.FULL_RAW_MATCH_TIER if generation in doc.get("text", "") else None
            if tier == simple_trace.FULL_RAW_MATCH_TIER:
                full = 1
    return full, longest, nv_sum, docs


@functools.lru_cache(maxsize=None)
def load_stats(results_path: str) -> GenerationStats:
    """Per-generation statistics of a results file, from its cache when it is up to date."""
    results = Path(results_path)
    cache = results.with_name(results.name[: -len("_results.json")] + "_genstats.json")
    source = {"size": results.stat().st_size, "mtime_ns": results.stat().st_mtime_ns}
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("source") == source:
            return _stats_from(cached)
    import simple_trace  # imported lazily: it loads infini_gram and transformers

    rows = {"prompt_ids": [], "sample_idx": [], "prompts": [], "full": [], "longest": [], "nv_sum": [], "docs": []}
    with open(results) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt_id" not in record or "sample_idx" not in record:
                raise ValueError(f"{results_path}: results lack prompt_id/sample_idx, cannot tell the units")
            full, longest, nv_sum, docs = _generation_stats(record, simple_trace)
            rows["prompt_ids"].append(record["prompt_id"])
            rows["sample_idx"].append(record["sample_idx"])
            rows["full"].append(full)
            rows["longest"].append(longest)
            rows["nv_sum"].append(nv_sum)
            rows["docs"].append(docs)
    rows["prompts"] = _prompts_of(results_path, rows["prompt_ids"], rows["sample_idx"])
    cache.write_text(json.dumps({"source": source, **rows}))
    return _stats_from(rows)


def _prompts_of(results_path: str, prompt_ids: list[int], sample_idx: list[int]) -> list[str]:
    """Prompt text of each result, from the generation file of the run (via generation_runs);
    results files hold only the generation, prompt_id and sample_idx."""
    sys.path.insert(0, str(REPO_ROOT / "memorization_experiment" / "generation"))
    from generation_runs import generation_path, runs, results_path as run_results_path, SETTINGS

    for model, corpus in runs():
        for setting in SETTINGS:
            if (REPO_ROOT / run_results_path(model, corpus, setting)).resolve() == Path(results_path).resolve():
                data = json.loads((REPO_ROOT / generation_path(model, corpus, setting)).read_text(encoding="utf-8"))
                prompt_of = {
                    (r["prompt_id"], r["sample_idx"]): r["prompt"]
                    for domains in data["results"].values() for records in domains.values() for r in records
                }
                return [prompt_of[(p, s)] for p, s in zip(prompt_ids, sample_idx)]
    raise ValueError(f"{results_path} is not a results file of generation_runs")


def _stats_from(rows: dict) -> GenerationStats:
    return GenerationStats(
        prompt_ids=np.asarray(rows["prompt_ids"]),
        sample_idx=np.asarray(rows["sample_idx"]),
        prompts=list(rows["prompts"]),
        full=np.asarray(rows["full"], dtype=float),
        longest=np.asarray(rows["longest"], dtype=float),
        nv_sum=np.asarray(rows["nv_sum"], dtype=float),
        docs=np.asarray(rows["docs"], dtype=float),
    )


@dataclass
class Units:
    """The independent units of a run: their keys, prompts and summed statistics."""

    keys: list
    prompts: list[str]
    sums: np.ndarray  # (units, 5)
    unit: str


def units_of(stats: GenerationStats, setting: str) -> Units:
    per_generation = np.stack(
        [stats.full, stats.longest, stats.nv_sum, stats.docs, np.ones(len(stats.full))], axis=1
    )
    unit = RESAMPLING_UNIT[setting]
    if unit == "generation":
        keys = list(zip(stats.prompt_ids.tolist(), stats.sample_idx.tolist()))
        return Units(keys, stats.prompts, per_generation, unit)
    order = {}
    for i, pid in enumerate(stats.prompt_ids.tolist()):
        order.setdefault(pid, []).append(i)
    keys = list(order)
    sums = np.stack([per_generation[idx].sum(axis=0) for idx in order.values()])
    prompts = [stats.prompts[idx[0]] for idx in order.values()]
    return Units(keys, prompts, sums, unit)


def metrics_from_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Set-level metrics from summed statistics; works row-wise on (rounds, 5) arrays."""
    totals = np.atleast_2d(totals)
    gens, docs = totals[:, GENS], totals[:, DOCS]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "generations_full_matches_ratio": np.where(gens > 0, totals[:, FULL] / gens, 0.0),
            "average_longest_span_length": np.where(gens > 0, totals[:, LONGEST] / gens, 0.0),
            "avg_nv_recall": np.where(docs > 0, totals[:, NV] / docs, 0.0),
        }


def resample_counts(n_units: int, rounds: int, rng: np.random.Generator) -> np.ndarray:
    """How often each unit is drawn in each round: (rounds, n_units)."""
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=rounds).astype(np.float64)


def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _metric_entry(estimate: float, rounds: np.ndarray) -> dict:
    low, high = _interval(rounds)
    return {"estimate": float(estimate), "ci_low": low, "ci_high": high}


def setting_ci(summary_path: str, rounds: int = DEFAULT_BOOTSTRAP_SAMPLES, seed: int = DEFAULT_SEED) -> dict:
    """95% bootstrap CIs of the metrics of one SimpleTrace run."""
    summary_path = _resolve(summary_path)
    setting = setting_of(summary_path)
    units = units_of(load_stats(results_path_of(summary_path)), setting)
    estimate = metrics_from_totals(units.sums.sum(axis=0))
    boot = metrics_from_totals(resample_counts(len(units.keys), rounds, np.random.default_rng(seed)) @ units.sums)
    summary = json.loads(Path(summary_path).read_text())
    out = {
        "setting": setting,
        "resampled_unit": units.unit,
        "units": len(units.keys),
        "generations": int(units.sums[:, GENS].sum()),
        "metrics": {},
    }
    for metric in METRICS:
        entry = _metric_entry(estimate[metric][0], boot[metric])
        entry["matches_summary"] = bool(abs(entry["estimate"] - float(summary[metric])) < 1e-6)
        out["metrics"][metric] = entry
    if units.sums[:, FULL].sum() == 0:
        out["metrics"]["generations_full_matches_ratio"]["zero_matches_upper_bound"] = 3 / len(units.keys)
    return out


def _propensity(setting_values: np.ndarray, prefix_values: np.ndarray) -> np.ndarray:
    """f_s / (f_s + f_prefix) row-wise, NaN where both are 0 (the round is skipped)."""
    total = setting_values + prefix_values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, setting_values / total, np.nan)


def propensity_ci(
    setting_summary: str,
    prefix_summary: str,
    metrics: list[str],
    rounds: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """95% bootstrap CIs of PM = f_setting / (f_setting + f_prefix), the two sets resampled independently."""
    setting_summary, prefix_summary = _resolve(setting_summary), _resolve(prefix_summary)
    setting_units = units_of(load_stats(results_path_of(setting_summary)), setting_of(setting_summary))
    prefix_units = units_of(load_stats(results_path_of(prefix_summary)), setting_of(prefix_summary))
    rng_setting, rng_prefix = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    est_s = metrics_from_totals(setting_units.sums.sum(axis=0))
    est_p = metrics_from_totals(prefix_units.sums.sum(axis=0))
    boot_s = metrics_from_totals(resample_counts(len(setting_units.keys), rounds, rng_setting) @ setting_units.sums)
    boot_p = metrics_from_totals(resample_counts(len(prefix_units.keys), rounds, rng_prefix) @ prefix_units.sums)
    out = {}
    for metric in metrics:
        if metric not in METRICS:
            continue
        pm = _propensity(boot_s[metric], boot_p[metric])
        estimate = _propensity(est_s[metric], est_p[metric])[0]
        low, high = _interval(pm)
        out[metric] = {
            "estimate": None if np.isnan(estimate) else float(estimate),
            "ci_low": low,
            "ci_high": high,
            "skipped": int(np.isnan(pm).sum()),
        }
    return out


def _can_pair(a: Units, a_setting: str, b: Units, b_setting: str) -> bool:
    """Two runs can share their bootstrap draws when they have the same units with the same
    prompts (same prompt set, or same cue sequence) on a setting other than unconditional."""
    return (
        a_setting == b_setting != "unconditional"
        and sorted(a.keys) == sorted(b.keys)
        and dict(zip(a.keys, a.prompts)) == dict(zip(b.keys, b.prompts))
    )


def _bootstrap_two(a: Units, b: Units, paired: bool, rounds: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """Bootstrap metrics of two runs: one draw of units for both when paired, else independent draws."""
    if paired:
        position = {key: i for i, key in enumerate(b.keys)}
        b_sums = b.sums[[position[key] for key in a.keys]]
        counts = resample_counts(len(a.keys), rounds, rng)
        return metrics_from_totals(counts @ a.sums), metrics_from_totals(counts @ b_sums)
    return (
        metrics_from_totals(resample_counts(len(a.keys), rounds, rng) @ a.sums),
        metrics_from_totals(resample_counts(len(b.keys), rounds, rng) @ b.sums),
    )


def _load_units(summary_path: str) -> tuple[Units, str]:
    setting = setting_of(summary_path)
    return units_of(load_stats(results_path_of(summary_path)), setting), setting


def comparison_ci(
    series_summary: str,
    reference_summary: str,
    metrics: list[str],
    rounds: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """95% bootstrap CIs of series - reference and of f_series / (f_series + f_reference).

    Paired (one draw of units for both runs) when the two runs share their units, with the
    same prompt for every unit, and the setting is not unconditional; independent otherwise.
    The two runs may be different settings of one model (e.g. a setting against prefix).
    """
    series, series_setting = _load_units(_resolve(series_summary))
    reference, reference_setting = _load_units(_resolve(reference_summary))
    paired = _can_pair(series, series_setting, reference, reference_setting)
    boot_s, boot_r = _bootstrap_two(series, reference, paired, rounds, np.random.default_rng(seed))
    est_s = metrics_from_totals(series.sums.sum(axis=0))
    est_r = metrics_from_totals(reference.sums.sum(axis=0))
    out = {"paired": paired, "metrics": {}}
    for metric in metrics:
        if metric not in METRICS:
            continue
        ratio = _propensity(boot_s[metric], boot_r[metric])
        ratio_estimate = _propensity(est_s[metric], est_r[metric])[0]
        ratio_low, ratio_high = _interval(ratio)
        out["metrics"][metric] = {
            "difference": _metric_entry(est_s[metric][0] - est_r[metric][0], boot_s[metric] - boot_r[metric]),
            "propensity": {
                "estimate": None if np.isnan(ratio_estimate) else float(ratio_estimate),
                "ci_low": ratio_low,
                "ci_high": ratio_high,
                "skipped": int(np.isnan(ratio).sum()),
            },
        }
    return out


def propensity_difference_ci(
    a_setting_summary: str,
    a_prefix_summary: str,
    b_setting_summary: str,
    b_prefix_summary: str,
    metrics: list[str],
    rounds: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """95% bootstrap CI of PM_a - PM_b, the propensities of two runs (e.g. DFM and Comma) on one setting.

    The two setting sets share their draws when they can be paired, and so do the two prefix
    sets; setting and prefix are always resampled independently. Rounds where either PM is
    undefined (both of its values 0) are skipped.
    """
    a_s, a_s_setting = _load_units(_resolve(a_setting_summary))
    a_p, a_p_setting = _load_units(_resolve(a_prefix_summary))
    b_s, b_s_setting = _load_units(_resolve(b_setting_summary))
    b_p, b_p_setting = _load_units(_resolve(b_prefix_summary))
    paired_setting = _can_pair(a_s, a_s_setting, b_s, b_s_setting)
    paired_prefix = _can_pair(a_p, a_p_setting, b_p, b_p_setting)
    rng_setting, rng_prefix = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    boot_as, boot_bs = _bootstrap_two(a_s, b_s, paired_setting, rounds, rng_setting)
    boot_ap, boot_bp = _bootstrap_two(a_p, b_p, paired_prefix, rounds, rng_prefix)
    out = {"paired_setting": paired_setting, "paired_prefix": paired_prefix, "metrics": {}}
    for metric in metrics:
        if metric not in METRICS:
            continue
        pm_a = _propensity(metrics_from_totals(a_s.sums.sum(axis=0))[metric], metrics_from_totals(a_p.sums.sum(axis=0))[metric])[0]
        pm_b = _propensity(metrics_from_totals(b_s.sums.sum(axis=0))[metric], metrics_from_totals(b_p.sums.sum(axis=0))[metric])[0]
        diff = _propensity(boot_as[metric], boot_ap[metric]) - _propensity(boot_bs[metric], boot_bp[metric])
        low, high = _interval(diff)
        out["metrics"][metric] = {
            "estimate": None if np.isnan(pm_a) or np.isnan(pm_b) else float(pm_a - pm_b),
            "ci_low": low,
            "ci_high": high,
            "skipped": int(np.isnan(diff).sum()),
        }
    return out

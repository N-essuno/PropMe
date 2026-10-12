"""Load token-length distributions for fully matched generations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_memorization_experiments import DEFAULT_LENGTH_BUCKETS  # noqa: E402


FULL_RAW_MATCH_TIER = "exact_full_raw"


@dataclass(frozen=True)
class LengthBin:
    label: str
    legend_label: str
    lower: int
    upper: int | None
    color: str
    text_color: str

    def contains(self, length: int) -> bool:
        return length >= self.lower and (self.upper is None or length <= self.upper)


LENGTH_BIN_COLORS = (
    ("#F7FBFF", "#111111"),
    ("#DEEBF7", "#111111"),
    ("#C6DBEF", "#111111"),
    ("#9ECAE1", "#111111"),
    ("#6BAED6", "#111111"),
    ("#4292C6", "#FFFFFF"),
    ("#2171B5", "#FFFFFF"),
    ("#08519C", "#FFFFFF"),
    ("#08306B", "#FFFFFF"),
)


def _build_length_bins(raw_buckets: str) -> tuple[LengthBin, ...]:
    bins: list[LengthBin] = []
    for index, raw_bucket in enumerate(raw_buckets.split(",")):
        lower_raw, upper_raw = raw_bucket.strip().split("-", 1)
        lower = int(lower_raw)
        upper = None if upper_raw.lower() == "inf" else int(upper_raw)
        color, text_color = LENGTH_BIN_COLORS[index % len(LENGTH_BIN_COLORS)]
        bins.append(
            LengthBin(
                label=f"({lower}, {'inf' if upper is None else upper})",
                legend_label=(
                    f"{lower}+ tokens"
                    if upper is None
                    else f"{lower}–{upper} tokens"
                ),
                lower=lower,
                upper=upper,
                color=color,
                text_color=text_color,
            )
        )
    return tuple(bins)


FULL_MATCH_LENGTH_BINS = _build_length_bins(DEFAULT_LENGTH_BUCKETS)


@dataclass(frozen=True)
class FullMatchLengthDistribution:
    total_generations: int
    counts: dict[str, int]

    @property
    def full_match_count(self) -> int:
        return sum(self.counts.values())

    def generation_ratio(self, bin_label: str) -> float:
        if self.total_generations == 0:
            return 0.0
        return self.counts.get(bin_label, 0) / self.total_generations

    def within_full_matches_percentage(self, bin_label: str) -> float:
        if self.full_match_count == 0:
            return 0.0
        return 100.0 * self.counts.get(bin_label, 0) / self.full_match_count


def results_path_for_summary(summary_path: Path) -> Path:
    suffix = "_summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"Expected a summary filename ending in {suffix}: {summary_path}")
    return summary_path.with_name(summary_path.name.removesuffix(suffix) + "_results.json")


def _length_bin_for(span_length: int) -> LengthBin:
    for length_bin in FULL_MATCH_LENGTH_BINS:
        if length_bin.contains(span_length):
            return length_bin
    raise ValueError(f"Full-match span length must be positive, got {span_length}")


def load_full_match_length_distribution(
    summary_path: Path,
) -> FullMatchLengthDistribution:
    """Reconstruct one full-match token length per generation from trace JSONL."""
    results_path = results_path_for_summary(summary_path)
    counts = {length_bin.label: 0 for length_bin in FULL_MATCH_LENGTH_BINS}
    total_generations = 0

    with results_path.open(encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            total_generations += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{results_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            full_match_lengths: list[int] = []
            for span_index, span in enumerate(row.get("spans", [])):
                docs = span.get("docs", [])
                if not any(
                    doc.get("match_tier") == FULL_RAW_MATCH_TIER for doc in docs
                ):
                    continue
                span_length = span.get("span_length")
                if not isinstance(span_length, int) or isinstance(span_length, bool):
                    raise ValueError(
                        f"{results_path}:{line_number}: full-match span {span_index} "
                        "has no integer 'span_length'"
                    )
                full_match_lengths.append(span_length)

            if full_match_lengths:
                longest_full_match = max(full_match_lengths)
                counts[_length_bin_for(longest_full_match).label] += 1

    return FullMatchLengthDistribution(
        total_generations=total_generations,
        counts=counts,
    )


def active_length_bins(
    distributions: dict[str, FullMatchLengthDistribution],
) -> list[LengthBin]:
    """Return standard bins observed in at least one plotted series."""
    return [
        length_bin
        for length_bin in FULL_MATCH_LENGTH_BINS
        if any(
            distribution.counts.get(length_bin.label, 0) > 0
            for distribution in distributions.values()
        )
    ]

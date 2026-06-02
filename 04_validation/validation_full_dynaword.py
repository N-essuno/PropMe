from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_OUTPUT_DIR = REPO_ROOT / "04_validation" / "output"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memorization_experiment.sample_docs import parse_metadata

simple_trace = importlib.import_module("03_tracing.simple_trace")
trace_generation = simple_trace.trace_generation


@dataclass
class QuerySample:
    sample_ix: int
    source_sample_ix: int
    query_kind: str
    doc_ix: int
    expected_doc_id: str
    query_text: str
    query_token_count: int
    source_doc_len: int | None
    source_disp_len: int | None
    query_start_token: int
    query_end_token: int
    metadata: dict[str, Any]


def _parse_optional_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def _doc_id_from_metadata(metadata: dict[str, Any], doc_ix: int) -> str:
    for key in ("id", "doc_id", "source", "url"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return str(doc_ix)


def _doc_id_from_trace_doc(doc: dict[str, Any]) -> str:
    for key in ("id", "doc_id", "source", "url"):
        value = doc.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _load_unigram_probs(path: str) -> dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["prob"] for k, v in raw.items()}


def _decode_query_window(
    token_ids: list[int],
    tokenizer: Any,
    *,
    start: int,
    end: int,
    min_query_tokens: int,
) -> tuple[int, int, str]:
    if not token_ids:
        raise ValueError("Cannot build a query from an empty document.")
    start = max(0, start)
    end = min(len(token_ids), end)

    if end - start < min_query_tokens:
        raise ValueError(
            f"Document/window has {end - start} tokens, below --min-query-tokens={min_query_tokens}."
        )

    query_text = tokenizer.decode(token_ids[start:end]).strip()
    if not query_text:
        raise ValueError("Decoded query text is empty.")

    # Re-encode to make metric denominators match SimpleTrace's tokenization of
    # the actual query string, not just the sampled source token window.
    query_token_count = len(tokenizer.encode(query_text))
    if query_token_count < min_query_tokens:
        raise ValueError(
            f"Decoded query has {query_token_count} tokens, below --min-query-tokens={min_query_tokens}."
        )

    return start, end, query_text


def _is_word_initial_token(tokenizer: Any, token_id: int) -> bool:
    token = tokenizer.convert_ids_to_tokens(token_id)
    return isinstance(token, str) and token.startswith("▁")


def _is_clean_trace_window(
    token_ids: list[int],
    tokenizer: Any,
    *,
    start: int,
    end: int,
    min_query_tokens: int,
) -> bool:
    """Mirror SimpleTrace's final-span cleanliness constraints."""
    if start >= end or end - start < min_query_tokens:
        return False
    if not _is_word_initial_token(tokenizer, token_ids[start]):
        return False
    if end < len(token_ids) and not _is_word_initial_token(tokenizer, token_ids[end]):
        return False

    span_text = tokenizer.decode(token_ids[start:end]).strip()
    if not span_text:
        return False
    # return True
    return not any(ch in "!.?\n" for ch in span_text[:-1])


def _find_clean_trace_window(
    token_ids: list[int],
    tokenizer: Any,
    *,
    target: str,
    max_query_tokens: int,
    min_query_tokens: int,
) -> tuple[int, int, str]:
    """Find a long clean window near the requested document region.

    SimpleTrace drops spans containing interior sentence punctuation, so exact
    validation queries should be sampled from spans that SimpleTrace can retain.
    """
    if not token_ids:
        raise ValueError("Cannot build a query from an empty document.")
    if max_query_tokens < min_query_tokens:
        raise ValueError("--query-token-len/--partial-query-tokens is below --min-query-tokens")

    word_starts = [
        i for i, token_id in enumerate(token_ids) if _is_word_initial_token(tokenizer, token_id)
    ]
    if not word_starts:
        raise ValueError("Document contains no word-initial token starts.")

    doc_len = len(token_ids)
    doc_mid = doc_len / 2

    def start_score(start: int) -> tuple[float, int]:
        if target == "start":
            return (start, 0)
        if target == "middle":
            return (abs(start - doc_mid), start)
        if target == "end":
            return (abs((start + max_query_tokens) - doc_len), -start)
        # For the generic exact query, prefer the longest clean region; start is
        # only a tie-breaker after a candidate is found.
        return (0, start)

    ordered_starts = sorted(word_starts, key=start_score)
    best: tuple[int, int, str] | None = None
    best_score: tuple[float, int, int] | None = None

    for start in ordered_starts:
        max_end = min(doc_len, start + max_query_tokens)
        min_end = start + min_query_tokens
        if min_end > max_end:
            continue

        for end in range(max_end, min_end - 1, -1):
            if not _is_clean_trace_window(
                token_ids,
                tokenizer,
                start=start,
                end=end,
                min_query_tokens=min_query_tokens,
            ):
                continue

            text = tokenizer.decode(token_ids[start:end]).strip()
            length = end - start
            center = (start + end) / 2
            if target == "start":
                score = (start, -length, end)
            elif target == "middle":
                score = (abs(center - doc_mid), -length, start)
            elif target == "end":
                score = (abs(end - doc_len), -length, -start)
            else:
                score = (-length, abs(center - doc_mid), start)

            if best_score is None or score < best_score:
                best = (start, end, text)
                best_score = score
            break

        # For positional targets, the first good nearby start is enough unless
        # a previous candidate is still clearly better by length.
        if best is not None and target in {"start", "middle", "end"}:
            break

    if best is None:
        raise ValueError(
            f"No clean traceable {target} window with at least {min_query_tokens} tokens."
        )
    return best


def _find_clean_trace_window_from_start(
    token_ids: list[int],
    tokenizer: Any,
    *,
    max_query_tokens: int,
    min_query_tokens: int,
) -> tuple[int, int, str]:
    """Find a clean window anchored at token 0."""
    if not token_ids:
        raise ValueError("Cannot build a query from an empty document.")
    if max_query_tokens < min_query_tokens:
        raise ValueError("--partial-query-tokens is below --min-query-tokens")

    start = 0
    max_end = min(len(token_ids), start + max_query_tokens)
    min_end = start + min_query_tokens
    if min_end > max_end:
        raise ValueError(
            f"Document start window has only {max_end - start} tokens, below --min-query-tokens={min_query_tokens}."
        )

    for end in range(max_end, min_end - 1, -1):
        if _is_clean_trace_window(
            token_ids,
            tokenizer,
            start=start,
            end=end,
            min_query_tokens=min_query_tokens,
        ):
            return start, end, tokenizer.decode(token_ids[start:end]).strip()

    raise ValueError(
        "No clean traceable start-anchored window with at least "
        f"{min_query_tokens} tokens."
    )


def _find_clean_trace_window_from_middle(
    token_ids: list[int],
    tokenizer: Any,
    *,
    max_query_tokens: int,
    min_query_tokens: int,
) -> tuple[int, int, str]:
    """Find a clean window that must cover the document midpoint."""
    if not token_ids:
        raise ValueError("Cannot build a query from an empty document.")
    if max_query_tokens < min_query_tokens:
        raise ValueError("--partial-query-tokens is below --min-query-tokens")

    doc_len = len(token_ids)
    midpoint = doc_len // 2

    # Candidate starts must leave room for a valid window and allow the
    # midpoint to be inside [start, end).
    min_start = max(0, midpoint - max_query_tokens + 1)
    max_start = min(midpoint, doc_len - min_query_tokens)
    if min_start > max_start:
        raise ValueError(
            "No midpoint-covering window can satisfy the token length constraints."
        )

    candidate_starts = list(range(min_start, max_start + 1))
    candidate_starts.sort(key=lambda start: (abs((start + max_query_tokens / 2) - midpoint), start))

    for start in candidate_starts:
        max_end = min(doc_len, start + max_query_tokens)
        min_end = start + min_query_tokens
        if min_end > max_end:
            continue

        for end in range(max_end, min_end - 1, -1):
            if not (start <= midpoint < end):
                continue
            if _is_clean_trace_window(
                token_ids,
                tokenizer,
                start=start,
                end=end,
                min_query_tokens=min_query_tokens,
            ):
                return start, end, tokenizer.decode(token_ids[start:end]).strip()

    raise ValueError(
        "No clean traceable midpoint-covering window with at least "
        f"{min_query_tokens} tokens."
    )


def _find_clean_trace_window_to_end(
    token_ids: list[int],
    tokenizer: Any,
    *,
    max_query_tokens: int,
    min_query_tokens: int,
) -> tuple[int, int, str]:
    """Find a clean window anchored at the document end."""
    if not token_ids:
        raise ValueError("Cannot build a query from an empty document.")
    if max_query_tokens < min_query_tokens:
        raise ValueError("--partial-query-tokens is below --min-query-tokens")

    end = len(token_ids)
    min_start = max(0, end - max_query_tokens)
    max_start = end - min_query_tokens
    if max_start < min_start:
        raise ValueError(
            f"Document end window has only {end} tokens, below --min-query-tokens={min_query_tokens}."
        )

    for start in range(min_start, max_start + 1):
        if _is_clean_trace_window(
            token_ids,
            tokenizer,
            start=start,
            end=end,
            min_query_tokens=min_query_tokens,
        ):
            return start, end, tokenizer.decode(token_ids[start:end]).strip()

    raise ValueError(
        "No clean traceable end-anchored window with at least "
        f"{min_query_tokens} tokens."
    )


def _choose_query_window(
    token_ids: list[int],
    tokenizer: Any,
    *,
    query_token_len: int | None,
    min_query_tokens: int,
    rng: random.Random,
) -> tuple[int, int, str]:
    """Choose a random token window and decode it into a query string."""
    if query_token_len is None or len(token_ids) <= query_token_len:
        return _decode_query_window(
            token_ids,
            tokenizer,
            start=0,
            end=len(token_ids),
            min_query_tokens=min_query_tokens,
        )

    max_start = len(token_ids) - query_token_len
    candidate_starts = list(range(max_start + 1))
    rng.shuffle(candidate_starts)

    start = candidate_starts[0]
    for candidate in candidate_starts:
        token = tokenizer.convert_ids_to_tokens(token_ids[candidate])
        if isinstance(token, str) and token.startswith("▁"):
            start = candidate
            break
    return _decode_query_window(
        token_ids,
        tokenizer,
        start=start,
        end=start + query_token_len,
        min_query_tokens=min_query_tokens,
    )


def _clean_partial_windows(
    token_ids: list[int],
    tokenizer: Any,
    *,
    partial_query_tokens: int,
    min_query_tokens: int,
) -> dict[str, tuple[int, int, str]]:
    windows: dict[str, tuple[int, int, str]] = {}
    try:
        windows["partial_start"] = _find_clean_trace_window_from_start(
            token_ids,
            tokenizer,
            max_query_tokens=partial_query_tokens,
            min_query_tokens=min_query_tokens,
        )
    except ValueError:
        pass

    try:
        windows["partial_middle"] = _find_clean_trace_window_from_middle(
            token_ids,
            tokenizer,
            max_query_tokens=partial_query_tokens,
            min_query_tokens=min_query_tokens,
        )
    except ValueError:
        pass

    try:
        windows["partial_end"] = _find_clean_trace_window_to_end(
            token_ids,
            tokenizer,
            max_query_tokens=partial_query_tokens,
            min_query_tokens=min_query_tokens,
        )
    except ValueError:
        pass

    return windows


def _build_query_sample(
    *,
    sample_ix: int,
    source_sample_ix: int,
    query_kind: str,
    doc_ix: int,
    expected_doc_id: str,
    query_text: str,
    tokenizer: Any,
    source_doc_len: int | None,
    source_disp_len: int | None,
    query_start_token: int,
    query_end_token: int,
    metadata: dict[str, Any],
) -> QuerySample:
    return QuerySample(
        sample_ix=sample_ix,
        source_sample_ix=source_sample_ix,
        query_kind=query_kind,
        doc_ix=doc_ix,
        expected_doc_id=expected_doc_id,
        query_text=query_text,
        query_token_count=len(tokenizer.encode(query_text)),
        source_doc_len=source_doc_len,
        source_disp_len=source_disp_len,
        query_start_token=query_start_token,
        query_end_token=query_end_token,
        metadata=metadata,
    )


def sample_queries(
    *,
    engine: InfiniGramEngine,
    tokenizer: Any,
    num_samples: int,
    query_token_len: int | None,
    partial_query_tokens: int,
    min_query_tokens: int,
    seed: int,
    max_attempts_multiplier: int,
    include_full: bool,
    include_partials: bool,
) -> list[QuerySample]:
    """Sample random docs and expand each one into exact full/partial queries."""
    if num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    rng = random.Random(seed)
    total_doc_cnt = engine.engine.get_total_doc_cnt()
    if total_doc_cnt < 1:
        raise RuntimeError("Index contains no documents.")

    target = min(num_samples, total_doc_cnt)
    max_attempts = min(total_doc_cnt, max(target * max_attempts_multiplier, target))
    candidate_doc_ixs = rng.sample(range(total_doc_cnt), k=max_attempts)
    required_query_kinds: list[str] = []
    if include_full:
        required_query_kinds.append("full")
    if include_partials:
        required_query_kinds.extend(["partial_start", "partial_middle", "partial_end"])

    samples: list[QuerySample] = []
    source_samples = 0
    skipped = 0
    for doc_ix in candidate_doc_ixs:
        if source_samples >= target:
            break

        doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=None)
        token_ids = doc.get("token_ids", [])
        metadata = parse_metadata(doc.get("metadata"))
        expected_doc_id = _doc_id_from_metadata(metadata, doc_ix)
        source_doc_len = doc.get("doc_len")
        source_disp_len = doc.get("disp_len")

        exact_window: tuple[int, int, str] | None = None
        if include_full:
            try:
                full_end = len(token_ids)
                if query_token_len is not None:
                    full_end = min(full_end, query_token_len)
                exact_window = _decode_query_window(
                    token_ids,
                    tokenizer,
                    start=0,
                    end=full_end,
                    min_query_tokens=min_query_tokens,
                )
            except ValueError:
                skipped += 1
                continue

        partial_windows: dict[str, tuple[int, int, str]] = {}
        if include_partials:
            partial_windows = _clean_partial_windows(
                token_ids,
                tokenizer,
                partial_query_tokens=partial_query_tokens,
                min_query_tokens=min_query_tokens,
            )
            if any(kind not in partial_windows for kind in ("partial_start", "partial_middle", "partial_end")):
                skipped += 1
                continue

        source_sample_ix = source_samples
        sample_count_before_doc = len(samples)

        if include_full:
            assert exact_window is not None
            exact_start, exact_end, exact_text = exact_window
            samples.append(
                _build_query_sample(
                    sample_ix=len(samples),
                    source_sample_ix=source_sample_ix,
                    query_kind="full",
                    doc_ix=doc_ix,
                    expected_doc_id=expected_doc_id,
                    query_text=exact_text,
                    tokenizer=tokenizer,
                    source_doc_len=source_doc_len,
                    source_disp_len=source_disp_len,
                    query_start_token=exact_start,
                    query_end_token=exact_end,
                    metadata=metadata,
                )
            )

        if include_partials:
            for query_kind in ("partial_start", "partial_middle", "partial_end"):
                part_start, part_end, part_text = partial_windows[query_kind]
                samples.append(
                    _build_query_sample(
                        sample_ix=len(samples),
                        source_sample_ix=source_sample_ix,
                        query_kind=query_kind,
                        doc_ix=doc_ix,
                        expected_doc_id=expected_doc_id,
                        query_text=part_text,
                        tokenizer=tokenizer,
                        source_doc_len=source_doc_len,
                        source_disp_len=source_disp_len,
                        query_start_token=part_start,
                        query_end_token=part_end,
                        metadata=metadata,
                    )
                )

        if len(samples) - sample_count_before_doc != len(required_query_kinds):
            skipped += 1
            continue

        source_samples += 1

    if source_samples < target:
        raise RuntimeError(
            f"Only collected {source_samples} usable source documents with all required query kinds "
            f"({', '.join(required_query_kinds)}) out of requested {target}; skipped {skipped}. "
            "Lower --min-query-tokens, lower --partial-query-tokens, or raise --max-attempts-multiplier."
        )
    if not samples:
        raise RuntimeError(
            "No validation queries were created. Enable at least one query family "
            "or lower the query length thresholds."
        )

    return samples


def write_samples(samples: list[QuerySample], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def _ordered_unique_doc_ids(final_spans: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for span in final_spans:
        for doc in span.get("docs", []):
            doc_id = _doc_id_from_trace_doc(doc)
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                ordered.append(doc_id)
    return ordered


def _target_doc_entries(
    final_spans: list[dict[str, Any]],
    expected_doc_id: str,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for span in final_spans:
        for doc in span.get("docs", []):
            if _doc_id_from_trace_doc(doc) == expected_doc_id:
                docs.append(doc)
    return docs


def validate_one_result(
    *,
    sample: QuerySample,
    result: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    final_spans = result.get("final_spans", [])
    unique_doc_ids = _ordered_unique_doc_ids(final_spans)
    target_docs = _target_doc_entries(final_spans, sample.expected_doc_id)

    target_doc_rank = (
        unique_doc_ids.index(sample.expected_doc_id) + 1
        if sample.expected_doc_id in unique_doc_ids
        else None
    )
    target_doc_retrieved = int(target_doc_rank is not None)

    is_partial_query = sample.query_kind.startswith("partial_")
    is_exact_text_query = sample.query_kind == "full" or is_partial_query
    exact_text_match = int(
        is_exact_text_query
        and any(sample.query_text in doc.get("text", "") for doc in target_docs)
    )

    return {
        "sample_ix": sample.sample_ix,
        "source_sample_ix": sample.source_sample_ix,
        "query_kind": sample.query_kind,
        "doc_ix": sample.doc_ix,
        "expected_doc_id": sample.expected_doc_id,
        "query_token_count": sample.query_token_count,
        "query_start_token": sample.query_start_token,
        "query_end_token": sample.query_end_token,
        "runtime_seconds": runtime_seconds,
        "retrieved_doc_ids": unique_doc_ids,
        "target_doc_retrieved": target_doc_retrieved,
        "target_doc_rank": target_doc_rank,
        "exact_text_match": exact_text_match,
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_text_rows = [
        row
        for row in rows
        if row["query_kind"] == "full" or row["query_kind"].startswith("partial_")
    ]
    summary = {
        "source_doc_retrieval_rate": _mean(
            [row["target_doc_retrieved"] for row in rows]
        ),
        "exact_text_match_rate": _mean(
            [row["exact_text_match"] for row in exact_text_rows]
        ),
    }

    for query_kind in ("full", "partial_start", "partial_middle", "partial_end"):
        kind_rows = [row for row in rows if row["query_kind"] == query_kind]
        summary[f"{query_kind}_doc_retrieval_rate"] = _mean(
            [row["target_doc_retrieved"] for row in kind_rows]
        )
        summary[f"{query_kind}_exact_text_match_rate"] = _mean(
            [row["exact_text_match"] for row in kind_rows]
        )

    return summary


def write_jsonl(rows: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_trace_results(
    rows: list[tuple[QuerySample, dict[str, Any], float]],
    output_path: str,
) -> None:
    """Write raw SimpleTrace rows without using query text as a unique key."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample, result, runtime_seconds in rows:
            record = {
                "sample_ix": sample.sample_ix,
                "source_sample_ix": sample.source_sample_ix,
                "query_kind": sample.query_kind,
                "doc_ix": sample.doc_ix,
                "expected_doc_id": sample.expected_doc_id,
                "runtime_seconds": runtime_seconds,
                "generation": sample.query_text,
                "spans": [
                    {
                        "start": sp["start"],
                        "end": sp["end"],
                        "span_length": sp["end"] - sp["start"],
                        "text": sp["text"],
                        "docs": [
                            {
                                "text": doc.get("text", ""),
                                "id": doc.get("id", ""),
                            }
                            for doc in sp.get("docs", [])
                        ],
                    }
                    for sp in result.get("final_spans", [])
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clean_for_log(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _format_missing_doc_log(
    *,
    sample: QuerySample,
    result: dict[str, Any],
    validation_row: dict[str, Any],
    runtime_seconds: float,
) -> str:
    final_spans = result.get("final_spans", [])
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(
        "MISSING SOURCE DOC "
        f"sample_ix={sample.sample_ix} "
        f"source_sample_ix={sample.source_sample_ix} "
        f"query_kind={sample.query_kind}"
    )
    lines.append(f"expected_doc_id: {sample.expected_doc_id}")
    lines.append(f"doc_ix: {sample.doc_ix}")
    lines.append(f"runtime_seconds: {runtime_seconds:.6f}")
    lines.append(f"query_token_count: {sample.query_token_count}")
    lines.append(f"target_doc_retrieved_anywhere: {validation_row.get('target_doc_retrieved')}")
    lines.append(f"target_doc_rank: {validation_row.get('target_doc_rank')}")
    lines.append(
        f"query_source_window: [{sample.query_start_token}, {sample.query_end_token})"
    )
    lines.append(
        "retrieved_doc_ids: "
        + json.dumps(validation_row.get("retrieved_doc_ids", []), ensure_ascii=False)
    )
    lines.append(f"total_spans: {len(final_spans)}")
    lines.append("")
    lines.append("Query Text:")
    lines.append(_clean_for_log(sample.query_text))

    if not final_spans:
        lines.append("")
        lines.append("[NO FINAL SPANS]")
        lines.append("")
        return "\n".join(lines)

    for span_ix, span in enumerate(final_spans, start=1):
        span_docs = span.get("docs", [])
        lines.append("")
        lines.append("=" * 20 + f" SPAN {span_ix} / {len(final_spans)} " + "=" * 20)
        lines.append(
            f"Span token range: [{span.get('start', '')}, {span.get('end', '')})"
        )
        lines.append(f"Span length: {span.get('end', 0) - span.get('start', 0)}")
        lines.append("Span Text:")
        lines.append(_clean_for_log(span.get("text", "")))
        if not span_docs:
            lines.append("[NO DOCS]")
            continue

        for doc_ix, doc in enumerate(span_docs, start=1):
            lines.append("")
            lines.append("-" * 10 + f" Document {doc_ix} / {len(span_docs)} " + "-" * 10)
            lines.append(f"id: {_doc_id_from_trace_doc(doc)}")
            for key in (
                "source",
                "url",
                "doc_id",
            ):
                if key in doc:
                    lines.append(f"{key}: {_clean_for_log(doc.get(key))}")
            lines.append("text:")
            lines.append(_clean_for_log(doc.get("text", "")))

    lines.append("")
    return "\n".join(lines)


def write_missing_doc_logs(
    trace_rows: list[tuple[QuerySample, dict[str, Any], float]],
    validation_rows: list[dict[str, Any]],
    output_path: str,
) -> dict[str, int]:
    """Write SimpleTrace-style details when the expected doc id is absent."""
    missing = [
        (sample, result, runtime_seconds, validation_row)
        for (sample, result, runtime_seconds), validation_row in zip(
            trace_rows, validation_rows
        )
        if validation_row.get("target_doc_retrieved") == 0
    ]
    missing_anywhere_count = sum(
        1 for row in validation_rows if row.get("target_doc_retrieved") == 0
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if not missing:
            f.write("No examples where the expected source-document id is absent.\n")
            return {
                "logged_missing_anywhere": 0,
                "missing_anywhere": missing_anywhere_count,
            }

        for sample, result, runtime_seconds, validation_row in missing:
            f.write(
                _format_missing_doc_log(
                    sample=sample,
                    result=result,
                    validation_row=validation_row,
                    runtime_seconds=runtime_seconds,
                )
            )
            f.write("\n")
    return {
        "logged_missing_anywhere": len(missing),
        "missing_anywhere": missing_anywhere_count,
    }


def write_json(payload: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample dynaword index documents and validate SimpleTrace retrieval."
    )
    parser.add_argument(
        "--index-dir",
        default=str(REPO_ROOT / "00_data" / "dynaword_index"),
        help="Path to the dynaword InfiniGram index directory.",
    )
    parser.add_argument(
        "--unigram-probs-path",
        default=str(REPO_ROOT / "02_unigram_probs" / "unigram_probs_dynaword.json"),
        help="Path to dynaword unigram probabilities JSON.",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=20,
        help="Number of random source documents/windows to validate.",
    )
    parser.add_argument(
        "--query-token-len",
        type=_parse_optional_int,
        default=None,
        help=(
            "Maximum token length for full-document queries. Default None uses the "
            "entire sampled document."
        ),
    )
    parser.add_argument(
        "--partial-query-tokens",
        type=int,
        default=128,
        help=(
            "Maximum token length for clean partial start/middle/end substring checks. "
            "Actual windows may be shorter to avoid punctuation SimpleTrace would drop."
        ),
    )
    parser.add_argument(
        "--min-query-tokens",
        type=int,
        default=20,
        help="Skip sampled documents/windows shorter than this many query tokens.",
    )
    parser.add_argument(
        "--no-full",
        dest="include_full",
        action="store_false",
        help="Disable exact clean source-window queries.",
    )
    parser.add_argument(
        "--no-partials",
        dest="include_partials",
        action="store_false",
        help="Disable partial start/middle/end substring queries.",
    )
    parser.set_defaults(
        include_full=True,
        include_partials=True,
    )
    parser.add_argument(
        "--docs-per-span",
        type=int,
        default=10,
        help="Maximum retrieved documents per SimpleTrace span.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face tokenizer model name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible document/window sampling and trace subsampling.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=20,
        help="Try up to num_samples * this many random documents to find usable samples.",
    )
    parser.add_argument(
        "--sample-output",
        default=str(VALIDATION_OUTPUT_DIR / "validation_full_samples_dynaword.jsonl"),
        help="JSONL output path for sampled source queries.",
    )
    parser.add_argument(
        "--trace-output",
        default=str(VALIDATION_OUTPUT_DIR / "validation_full_traces_dynaword.jsonl"),
        help="JSONL output path for raw SimpleTrace results.",
    )
    parser.add_argument(
        "--per-query-output",
        default=str(VALIDATION_OUTPUT_DIR / "validation_full_per_query_dynaword.jsonl"),
        help="JSONL output path for per-query validation metrics.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(VALIDATION_OUTPUT_DIR / "validation_full_summary_dynaword.json"),
        help="JSON output path for aggregate validation metrics.",
    )
    parser.add_argument(
        "--missing-log-output",
        default=str(VALIDATION_OUTPUT_DIR / "validation_full_missing_doc_logs.txt"),
        help="Text log path for SimpleTrace-style details when the expected doc is not retrieved.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.docs_per_span < 1:
        raise ValueError("--docs-per-span must be >= 1")
    if args.query_token_len is not None and args.query_token_len < 1:
        raise ValueError("--query-token-len must be >= 1 or None")
    if args.partial_query_tokens < 1:
        raise ValueError("--partial-query-tokens must be >= 1")
    if args.min_query_tokens < 1:
        raise ValueError("--min-query-tokens must be >= 1")
    if args.max_attempts_multiplier < 1:
        raise ValueError("--max-attempts-multiplier must be >= 1")
    if not (args.include_full or args.include_partials):
        raise ValueError("At least one query family must be enabled.")

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        add_bos_token=False,
        add_eos_token=False,
    )
    engine = InfiniGramEngine(
        index_dir=args.index_dir,
        eos_token_id=tokenizer.eos_token_id,
        precompute_unigram_logprobs=False,
    )
    unigram_probs = _load_unigram_probs(args.unigram_probs_path)

    samples = sample_queries(
        engine=engine,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        query_token_len=args.query_token_len,
        partial_query_tokens=args.partial_query_tokens,
        min_query_tokens=args.min_query_tokens,
        seed=args.seed,
        max_attempts_multiplier=args.max_attempts_multiplier,
        include_full=args.include_full,
        include_partials=args.include_partials,
    )
    write_samples(samples, args.sample_output)

    trace_rows: list[tuple[QuerySample, dict[str, Any], float]] = []
    validation_rows: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Validating", unit="query"):
        start_time = time.perf_counter()
        result = trace_generation(
            generation=sample.query_text,
            engine=engine,
            enc=tokenizer,
            unigram_probs=unigram_probs,
            docs_per_span=args.docs_per_span,
        )
        runtime_seconds = time.perf_counter() - start_time
        trace_rows.append((sample, result, runtime_seconds))
        validation_rows.append(
            validate_one_result(
                sample=sample,
                result=result,
                runtime_seconds=runtime_seconds,
            )
        )

    write_trace_results(trace_rows, args.trace_output)
    write_missing_doc_logs(
        trace_rows,
        validation_rows,
        args.missing_log_output,
    )

    summary = summarize_validation(validation_rows)
    summary["config"] = {
        "index_dir": args.index_dir,
        "unigram_probs_path": args.unigram_probs_path,
        "num_samples": args.num_samples,
        "query_token_len": args.query_token_len,
        "partial_query_tokens": args.partial_query_tokens,
        "min_query_tokens": args.min_query_tokens,
        "include_full": args.include_full,
        "include_partials": args.include_partials,
        "docs_per_span": args.docs_per_span,
        "tokenizer_model": args.tokenizer_model,
        "seed": args.seed,
    }
    summary["outputs"] = {
        "sample_output": args.sample_output,
        "trace_output": args.trace_output,
        "per_query_output": args.per_query_output,
        "summary_output": args.summary_output,
        "missing_log_output": args.missing_log_output,
    }

    write_jsonl(validation_rows, args.per_query_output)
    write_json(summary, args.summary_output)

    print("\n" + "=" * 30 + " VALIDATION SUMMARY " + "=" * 30)
    for key, value in summary.items():
        if key in {"config", "outputs"}:
            continue
        print(f"{key}: {value}")
    print(f"\nWrote samples: {args.sample_output}")
    print(f"Wrote traces: {args.trace_output}")
    print(f"Wrote per-query metrics: {args.per_query_output}")
    print(f"Wrote summary: {args.summary_output}")
    print(f"Wrote missing-source-doc logs: {args.missing_log_output}")


if __name__ == "__main__":
    main()

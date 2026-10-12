"""
python 03_tracing/simple_trace.py \
    --dataset dummy \
    --index-dir 00_prepare_data/dummy_index \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dummy.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --results-output simpletrace_results_dummy.jsonl \
    --summary-output simpletrace_evaluation_summary_dummy.json \
    --length-buckets 1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf \
"""

import ast
import argparse
import bisect
from collections.abc import Sequence
from decimal import Decimal, ROUND_HALF_UP
import math
import random
import json
import os
import re
import time
import signal
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError
from difflib import SequenceMatcher

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

if __package__:
    from .data_loading import *
else:
    from data_loading import *
from tqdm import tqdm

start_time = time.time()

TEXT_MATCH_MODE = "text"
MIXED_MATCH_MODE = "mixed"
FULL_RAW_MATCH_TIER = "exact_full_raw"
FULL_NORMALIZED_MATCH_TIER = "exact_full_normalized"
PARTIAL_MATCH_TIER = "partial"
MIXED_MIN_SPAN_TOKENS = 4
TEXT_MODE_BOUNDARY_CHARS = "!.?\n"  # sentence boundaries in text mode
DEFAULT_METRIC_DECIMALS = 9
DEFAULT_SEED = 42  # seeds the per-generation document sampling in trace_generation
_metric_decimals = DEFAULT_METRIC_DECIMALS

# Lock that serializes print() calls across threads so output lines
# from different generations never interleave in the terminal.
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe drop-in replacement for print()."""
    with _print_lock:
        print(*args, **kwargs)


def _parse_index_metadata(raw_metadata) -> dict:
    """Parse InfiniGram metadata stored as a Python literal or JSON object."""
    if isinstance(raw_metadata, dict):
        parsed = raw_metadata
    elif isinstance(raw_metadata, str):
        try:
            parsed = ast.literal_eval(raw_metadata)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(raw_metadata)
            except json.JSONDecodeError as json_error:
                raise ValueError(
                    "InfiniGram metadata is neither a Python literal nor valid JSON"
                ) from json_error
    else:
        raise ValueError(
            f"Unsupported InfiniGram metadata type: {type(raw_metadata).__name__}"
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            f"InfiniGram metadata must decode to an object, got {type(parsed).__name__}"
        )

    metadata = parsed.get("metadata", parsed)
    if not isinstance(metadata, dict):
        raise ValueError(
            "InfiniGram metadata wrapper field 'metadata' must contain an object"
        )
    return metadata


_NV_TRANSLATION_TABLE = str.maketrans({
    # Curly/directional single quotes/apostrophes -> ASCII apostrophe
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    # Curly/directional double quotes -> ASCII double quote
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "«": '"',
    "»": '"',
    # Dash variants -> em dash
    "–": "—",
    "―": "—",
    "−": "—",
    # Unicode ellipsis -> ASCII 3 dots
    "…": "...",
})


def _nv_normalize_text(text: str) -> str:
    """Light normalization used before nv-recall block matching."""
    norm = unicodedata.normalize("NFKC", text).translate(_NV_TRANSLATION_TABLE)
    # Collapse spaced-dot ellipses (". . ." variants) to canonical "...".
    norm = re.sub(r"\.\s+\.\s+\.", "...", norm)
    # If ellipsis is glued to an alnum token, insert one separating space.
    norm = re.sub(r"\.\.\.(?=[0-9A-Za-z])", "... ", norm)
    # Strip Books3-like emphasis markers: _like this_ -> like this
    norm = re.sub(r"_([^_]+)_", r"\1", norm)
    return norm.lower()


def _nv_tokenize(text: str) -> list[str]:
    return _nv_normalize_text(text).split()


def _nv_identify_blocks(ref_words: list[str], cand_words: list[str]) -> list[dict]:
    """Return base exact-match blocks from SequenceMatcher (size > 0 only)."""
    matcher = SequenceMatcher(a=ref_words, b=cand_words)
    blocks = []
    for match in matcher.get_matching_blocks():
        if match.size == 0:
            continue
        blocks.append({
            "i_start": match.a,
            "j_start": match.b,
            "i_end": match.a + match.size,
            "j_end": match.b + match.size,
            "matched_len": match.size,
        })
    return blocks


def _nv_merge_blocks(blocks: list[dict], tau_gap: int, tau_align: int) -> list[dict]:
    """Merge adjacent, aligned blocks while preserving monotone order."""
    if not blocks:
        return []

    merged = [dict(blocks[0])]
    for nxt in blocks[1:]:
        curr = merged[-1]
        gap_ref = nxt["i_start"] - curr["i_end"]
        gap_cand = nxt["j_start"] - curr["j_end"]
        if max(gap_ref, gap_cand) <= tau_gap and abs(gap_ref - gap_cand) <= tau_align:
            # Merge spans, but keep matched length conservative (sum only exact blocks).
            curr["i_end"] = nxt["i_end"]
            curr["j_end"] = nxt["j_end"]
            curr["matched_len"] += nxt["matched_len"]
        else:
            merged.append(dict(nxt))
    return merged


def _nv_filter_blocks(blocks: list[dict], min_len: int) -> list[dict]:
    """Keep only blocks long enough to count as near-verbatim extraction."""
    return [b for b in blocks if b["matched_len"] >= min_len]


def _nv_clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _move_key_to_end(mapping: dict, key: str) -> dict:
    """Ensure `key` is last in insertion order when present."""
    if key in mapping:
        value = mapping.pop(key)
        mapping[key] = value
    return mapping


def _move_summary_doc_id_fields_to_end(mapping: dict) -> dict:
    """Keep long document-ID list fields grouped at the end of summary outputs."""
    for key in (
        "full_exact_match_doc_ids",
        "full_normalized_match_doc_ids",
        "doc_ids_above_nv_recall_threshold",
    ):
        _move_key_to_end(mapping, key)
    return mapping


def _set_metric_decimals(places: int) -> None:
    """Configure the decimal precision used for fractional metrics."""
    global _metric_decimals
    _metric_decimals = places


def _round_metric_float(value: float, places: int | None = None) -> float:
    """Round decimal-valued metrics to a fixed precision."""
    if places is None:
        places = _metric_decimals
    quant = Decimal("1").scaleb(-places)
    rounded = float(Decimal(str(float(value))).quantize(quant, rounding=ROUND_HALF_UP))
    if rounded == 0.0:
        return 0.0
    return rounded


def _compute_nv_recall_with_params(
    reference_text: str,
    candidate_text: str,
    *,
    tau1_gap: int = 2,
    tau1_align: int = 1,
    l1: int = 20,
    tau2_gap: int = 10,
    tau2_align: int = 3,
    l2: int = 100,
) -> dict:
    """
    Compute nv-recall using explicit merge/filter parameters.

    Returns:
      {
        "nv_recall": float,
        "matched_words": int,
        "reference_words": int,
        "candidate_words": int,
        "missing_words": int,
        "additional_words": int,
      }
    """
    ref_words = _nv_tokenize(reference_text)
    cand_words = _nv_tokenize(candidate_text)

    if not ref_words:
        return {
            "nv_recall": 0.0,
            "matched_words": 0,
            "reference_words": 0,
            "candidate_words": len(cand_words),
            "missing_words": 0,
            "additional_words": len(cand_words),
        }

    blocks = _nv_identify_blocks(ref_words, cand_words)
    blocks = _nv_merge_blocks(blocks, tau1_gap, tau1_align)
    blocks = _nv_filter_blocks(blocks, l1)
    blocks = _nv_merge_blocks(blocks, tau2_gap, tau2_align)
    blocks = _nv_filter_blocks(blocks, l2)

    matched_words = sum(b["matched_len"] for b in blocks)
    reference_words = len(ref_words)
    candidate_words = len(cand_words)

    return {
        "nv_recall": _round_metric_float(matched_words / reference_words),
        "matched_words": matched_words,
        "reference_words": reference_words,
        "candidate_words": candidate_words,
        "missing_words": reference_words - matched_words,
        "additional_words": candidate_words - matched_words,
    }


def compute_nv_recall(
    reference_text: str,
    candidate_text: str,
) -> dict:
    """Adaptive nv-recall variant for arbitrary reference lengths.

    Thresholds scale with reference length (in words), then are clamped so
    behavior remains stable for both short and long sequences.
    """
    ref_words = _nv_tokenize(reference_text)
    n_ref = len(ref_words)
    if n_ref == 0:
        return _compute_nv_recall_with_params(reference_text, candidate_text)

    # Length-adaptive thresholds:
    # - for short refs, lower min_len enables non-zero recall;
    # - for long refs, caps recover conservative long-text behavior.
    l1 = _nv_clamp(round(0.15 * n_ref), 2, 20)
    l2 = _nv_clamp(round(0.35 * n_ref), 4, 100)

    tau1_gap = _nv_clamp(round(0.02 * n_ref), 1, 6)
    tau1_align = _nv_clamp(round(0.01 * n_ref), 1, 3)
    tau2_gap = _nv_clamp(round(0.08 * n_ref), 2, 12)
    tau2_align = _nv_clamp(round(0.03 * n_ref), 1, 5)

    stats = _compute_nv_recall_with_params(
        reference_text,
        candidate_text,
        tau1_gap=tau1_gap,
        tau1_align=tau1_align,
        l1=l1,
        tau2_gap=tau2_gap,
        tau2_align=tau2_align,
        l2=l2,
    )
    stats["adaptive_params"] = {
        "tau1_gap": tau1_gap,
        "tau1_align": tau1_align,
        "l1": l1,
        "tau2_gap": tau2_gap,
        "tau2_align": tau2_align,
        "l2": l2,
        "reference_words": n_ref,
    }
    return stats


def compute_longest_prefix(query, doc):
    """Helper: length of the longest prefix of `query` appearing as a
    contiguous sub-sequence anywhere in `doc`."""

    def shared_prefix_length(list1, list2):
        length = 0
        for a, b in zip(list1, list2):
            if a == b:
                length += 1
            else:
                break
        return length

    first_id = query[0]
    start_idx = [i for i, v in enumerate(doc) if v == first_id]
    longest = 0
    for si in start_idx:
        longest = max(longest, shared_prefix_length(query, doc[si:]))
    return longest


def longest_indexed_prefix_len(engine: InfiniGramEngine, suffix: list[int], max_doc_toks: int) -> int:
    """Length of the longest prefix of `suffix` that occurs in the index.

    - cnt > 0: the full suffix exists, so the entire suffix is a match.
    - cnt == 0: find() returns the empty rank range [l, l) where the suffix
      would sit in the sorted suffix array. The indexed suffix sharing the
      longest prefix with it is one of its two neighbours, rank l-1 (just
      before) or rank l (just after), so both are checked in every shard.
      Each neighbour's document window (up to max_doc_toks tokens) is
      searched for the prefix anywhere with compute_longest_prefix(), and
      the longest match across neighbours and shards is kept. All windows
      are fetched in one batch.
    """
    res = engine.find(input_ids=suffix)
    if res['cnt'] > 0:
        return len(suffix)

    ranks = []
    for s, (rank_start, _) in enumerate(res['segment_by_shard']):
        if rank_start > 0:
            ranks.append((s, rank_start - 1))
        ranks.append((s, rank_start))
    docs = _get_docs_by_ranks(engine, ranks, max_doc_toks)
    return max(
        (compute_longest_prefix(suffix, doc['token_ids']) for doc in docs),
        default=0,
    )


STEP1_BLOCK_SIZE = 4


def longest_prefix_lens(
    engine: InfiniGramEngine,
    gen_ids: list[int],
    starts,
    max_doc_toks: int,
    executor: ThreadPoolExecutor | None = None,
    stop_event: threading.Event | None = None,
) -> list[int] | None:
    """longest_indexed_prefix_len(engine, gen_ids[b:], max_doc_toks) for every b in `starts`.

    Same results with fewer index reads. If the suffix at an earlier start b'
    matched k tokens, gen_ids[b':b'+k] occurs in the index, so gen_ids[b:b'+k]
    does too and the suffix at b matches at least lb = k - (b - b') tokens.
    If lb covers the whole suffix, that is the answer. Otherwise one find() of
    the (lb + 1)-token prefix tells whether the answer is exactly lb; only if
    that prefix also occurs is the full lookup (with its neighbour document
    fetches in every shard) needed.

    Starts are processed left to right in blocks of STEP1_BLOCK_SIZE so each
    start can use the previous one's result; blocks run concurrently on
    `executor`. Returns None if `stop_event` is set.
    """
    def block_lens(block) -> list[int] | None:
        lens = []
        prev = None
        for b in block:
            if stop_event is not None and stop_event.is_set():
                return None
            suffix = gen_ids[b:]
            lower_bound = 0 if prev is None else max(0, prev[1] - (b - prev[0]))
            if lower_bound >= len(suffix):
                n = len(suffix)
            elif lower_bound > 0 and engine.find(input_ids=suffix[: lower_bound + 1])['cnt'] == 0:
                n = lower_bound
            else:
                n = longest_indexed_prefix_len(engine, suffix, max_doc_toks)
            lens.append(n)
            prev = (b, n)
        return lens

    blocks = [starts[i : i + STEP1_BLOCK_SIZE] for i in range(0, len(starts), STEP1_BLOCK_SIZE)]
    mapper = executor.map if executor is not None else map
    lens = []
    for block_result in mapper(block_lens, blocks):
        if block_result is None:
            return None
        lens.extend(block_result)
    return lens


def _normalize_mixed_text(text: str) -> str:
    """Light normalization for structure-aware full-document matching."""
    norm = unicodedata.normalize("NFKC", text)
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")
    norm = re.sub(r"[ \t]+", " ", norm)
    norm = re.sub(r" *\n *", "\n", norm)
    norm = re.sub(r"\n{3,}", "\n\n", norm)
    return norm.strip()


def _classify_match_tier(
    generation: str,
    doc_text: str,
    *,
    match_mode: str,
    normalized_generation: str | None = None,
) -> str:
    """Classify whether the retrieved doc matches the full generation or only a span."""
    if generation and generation in doc_text:
        return FULL_RAW_MATCH_TIER

    if match_mode == MIXED_MATCH_MODE:
        norm_generation = normalized_generation or _normalize_mixed_text(generation)
        if norm_generation:
            norm_doc = _normalize_mixed_text(doc_text)
            if norm_generation in norm_doc:
                return FULL_NORMALIZED_MATCH_TIER

    return PARTIAL_MATCH_TIER


class _RankSequence(Sequence):
    """(shard, rank) pairs of a find() result, in shard-then-rank order.

    Indexed lazily so frequent spans (millions of hits) are never
    materialized. random.sample() picks the same positions from any sequence
    of the same length, so sampling results match those of a flat list.
    """

    def __init__(self, segment_by_shard):
        self._segments = []
        self._offsets = []
        total = 0
        for s, (rank_start, rank_end) in enumerate(segment_by_shard):
            if rank_end > rank_start:
                self._segments.append((s, rank_start))
                self._offsets.append(total)
                total += rank_end - rank_start
        self._len = total

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._len))]
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError(i)
        seg = bisect.bisect_right(self._offsets, i) - 1
        s, rank_start = self._segments[seg]
        return (s, rank_start + i - self._offsets[seg])


def _flatten_ranks(find_res: dict) -> Sequence:
    """Flatten a find() result into (shard, rank) pairs."""
    return _RankSequence(find_res["segment_by_shard"])


def _get_docs_by_ranks(engine: InfiniGramEngine, ranks: list[tuple[int, int]], max_disp_len: int) -> list[dict]:
    """Batched get_doc_by_rank(); the engine fetches the documents in parallel.

    Calls the C++ binding directly: InfiniGramEngine.get_docs_by_ranks() in
    infini-gram 2.6.0 calls a misspelled method and raises AttributeError.
    """
    ranks = list(ranks)
    cpp = engine.engine
    num_shards = cpp.get_num_shards()
    valid = max_disp_len > 0 and all(
        0 <= s < num_shards and 0 <= r < cpp.get_tok_cnt(s=s) for s, r in ranks
    )
    if not valid:
        # Fall back to per-rank calls, which report errors as before.
        return [engine.get_doc_by_rank(s=s, rank=r, max_disp_len=max_disp_len) for s, r in ranks]
    results = cpp.get_docs_by_ranks(list_of_s_and_rank=ranks, max_disp_len=max_disp_len)
    return [
        {
            'doc_ix': res.doc_ix,
            'doc_len': res.doc_len,
            'disp_len': res.disp_len,
            'needle_offset': res.needle_offset,
            'metadata': res.metadata,
            'token_ids': res.token_ids,
            'blocked': res.blocked,
        }
        for res in results
    ]


def _sample_evenly(items: list, k: int) -> list:
    """Pick k items spread across the input order, deterministically."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]

    n = len(items)
    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    seen = set()
    sampled = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        sampled.append(items[idx])

    if len(sampled) == k:
        return sampled

    for idx, item in enumerate(items):
        if idx in seen:
            continue
        sampled.append(item)
        if len(sampled) == k:
            break

    return sampled


def _retrieve_docs_for_ranks(
    generation: str,
    ranks: list[tuple[int, int]],
    engine: InfiniGramEngine,
    enc,
    *,
    max_doc_toks: int,
    match_mode: str,
    limit: int,
    deterministic: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """Fetch document records for rank hits and annotate their match tier.

    With more hits than `limit`, picks evenly spread hits if `deterministic`,
    otherwise a random sample drawn from `rng` (the global RNG if None).
    """
    if limit < 1 or not ranks:
        return []

    if len(ranks) > limit:
        if deterministic:
            selected_ranks = _sample_evenly(ranks, limit)
        else:
            selected_ranks = (rng or random).sample(ranks, limit)
    else:
        selected_ranks = list(ranks)

    normalized_generation = (
        _normalize_mixed_text(generation)
        if match_mode == MIXED_MATCH_MODE else None
    )
    docs = []
    seen_examples: set[tuple[int, int]] = set()

    raw_docs = _get_docs_by_ranks(engine, selected_ranks, max_doc_toks)
    for (s, r), raw_doc in zip(selected_ranks, raw_docs):
        doc_ix = raw_doc.get("doc_ix")
        example_key = (s, int(doc_ix)) if doc_ix is not None else (s, int(r))
        if example_key in seen_examples:
            continue
        seen_examples.add(example_key)

        doc_meta = _parse_index_metadata(raw_doc["metadata"])
        doc_id = (
            doc_meta.get("id")
            or doc_meta.get("doc_id")
            or doc_meta.get("source")
            or doc_meta.get("url")
            or doc_meta.get("doc_ix")
            or raw_doc.get("doc_ix")
            or ""
        )
        doc_text = enc.decode(raw_doc["token_ids"])
        match_tier = _classify_match_tier(
            generation,
            doc_text,
            match_mode=match_mode,
            normalized_generation=normalized_generation,
        )
        nv_stats = compute_nv_recall(generation, doc_text)
        docs.append({
            "text": doc_text,
            "id": doc_id,
            "match_tier": match_tier,
            "nv_recall": nv_stats["nv_recall"],
            "nv_matched_words": nv_stats["matched_words"],
            "nv_reference_words": nv_stats["reference_words"],
            "nv_candidate_words": nv_stats["candidate_words"],
            "nv_missing_words": nv_stats["missing_words"],
            "nv_additional_words": nv_stats["additional_words"],
            **doc_meta,
        })
    return docs


def _retrieve_docs_for_find_result(
    generation: str,
    find_res: dict,
    engine: InfiniGramEngine,
    enc,
    *,
    max_doc_toks: int,
    match_mode: str,
    limit: int,
    deterministic: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """Fetch document records for an InfiniGram find() result."""
    return _retrieve_docs_for_ranks(
        generation,
        _flatten_ranks(find_res),
        engine,
        enc,
        max_doc_toks=max_doc_toks,
        match_mode=match_mode,
        limit=limit,
        deterministic=deterministic,
        rng=rng,
    )


def _keep_span(
    span_ids: list[int],
    span_text: str,
    *,
    start: int,
    end: int,
    total_tokens: int,
    enc,
    gen_ids: list[int],
    match_mode: str,
) -> bool:
    """Mode-specific span filter."""
    if match_mode == MIXED_MATCH_MODE:
        span_len = end - start
        if start == 0 and end == total_tokens:
            return True
        if span_len < MIXED_MIN_SPAN_TOKENS:
            return False
        return any(not ch.isspace() for ch in span_text)

    if any(ch in TEXT_MODE_BOUNDARY_CHARS for ch in span_text[:-1]):
        return False
    first_tok = enc.convert_ids_to_tokens(span_ids[0])
    if first_tok[0] != '▁':
        return False
    if end < total_tokens and enc.convert_ids_to_tokens(gen_ids[end])[0] != '▁':
        return False
    return True


def _trim_to_sentence_boundary(
    start: int,
    end: int,
    gen_ids: list[int],
    enc,
    has_boundary_char: list[bool],
) -> int:
    """Shrink a text-mode span so it stops at the first sentence boundary.

    Returns the largest `new_end <= end` such that gen_ids[start:new_end]
    passes text-mode filters (a) and (c) of _keep_span: no sentence-boundary
    character except as the last character of the span text, and a
    word-initial token (or the end of the generation) right after the span.
    Every prefix of a matched span also occurs in the index, so the trimmed
    span is still a verbatim match. `has_boundary_char[i]` says whether
    token i decodes to text containing a boundary character.
    """
    def _no_inner_boundary(e: int) -> bool:
        text = enc.decode(gen_ids[start:e])
        return not any(ch in TEXT_MODE_BOUNDARY_CHARS for ch in text[:-1])

    first = next((i for i in range(start, end) if has_boundary_char[i]), None)
    if first is not None:
        # Spans ending before the first boundary token contain no boundary
        # character, and filter (a) only gets stricter as the span grows, so
        # binary-search the longest valid end in [first, end].
        lo, hi = first, end
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _no_inner_boundary(mid):
                lo = mid
            else:
                hi = mid - 1
        end = lo

    total_tokens = len(gen_ids)
    while start < end < total_tokens and enc.convert_ids_to_tokens(gen_ids[end])[0] != '▁':
        end -= 1
    return end


def _build_mixed_anchor_spans(
    spans: list[tuple[int, int]],
    gen_ids: list[int],
    unigram_probs: dict,
    *,
    max_anchors: int = 8,
) -> list[tuple[int, int, list[int]]]:
    """Select long, distinctive exact spans to search for normalized full matches."""
    candidates = []
    seen_bounds = set()
    for start, end in spans:
        span_len = end - start
        if span_len < MIXED_MIN_SPAN_TOKENS:
            continue
        key = (start, end)
        if key in seen_bounds:
            continue
        seen_bounds.add(key)
        span_ids = gen_ids[start:end]
        prob = math.prod(unigram_probs.get(_id, 1.0) for _id in span_ids)
        candidates.append((start, end, span_ids, span_len, prob))

    candidates.sort(key=lambda item: (-item[3], item[4], item[0]))
    return [(start, end, span_ids) for start, end, span_ids, _, _ in candidates[:max_anchors]]


def _collect_mixed_full_match_docs(
    generation: str,
    anchor_spans: list[tuple[int, int, list[int]]],
    engine: InfiniGramEngine,
    enc,
    *,
    max_doc_toks: int,
    docs_per_span: int,
) -> list[dict]:
    """Look for normalized full-document matches using long exact anchor spans."""
    if not anchor_spans or docs_per_span < 1:
        return []

    per_anchor_limit = max(1, math.ceil(docs_per_span / len(anchor_spans)))
    matched_docs = []
    seen_doc_keys = set()
    for _, _, span_ids in anchor_spans:
        span_res = engine.find(input_ids=span_ids)
        if span_res.get("cnt", 0) == 0:
            continue
        anchor_docs = _retrieve_docs_for_find_result(
            generation,
            span_res,
            engine,
            enc,
            max_doc_toks=max_doc_toks,
            match_mode=MIXED_MATCH_MODE,
            limit=per_anchor_limit,
            deterministic=True,
        )
        for doc in anchor_docs:
            if doc["match_tier"] not in (FULL_RAW_MATCH_TIER, FULL_NORMALIZED_MATCH_TIER):
                continue
            doc_key = (str(doc.get("id", "")).strip(), doc.get("text", ""))
            if doc_key in seen_doc_keys:
                continue
            seen_doc_keys.add(doc_key)
            matched_docs.append(doc)
            if len(matched_docs) >= docs_per_span:
                return matched_docs
    return matched_docs


def _doc_identity(doc: dict) -> tuple[str, str, str]:
    """Stable identity tuple for span-level doc deduplication."""
    return (
        str(doc.get("id", "")).strip(),
        doc.get("match_tier", ""),
        doc.get("text", ""),
    )


def _group_verified_overlapping_spans(
    filt_spans: list[tuple],
    gen_ids: list[int],
    engine: InfiniGramEngine,
) -> list[list[int]]:
    """Group overlaps only when their complete union exists in the index.

    Two independently verified spans can overlap in a generation without the
    text covering their union occurring contiguously anywhere in the corpus.
    This is especially common for repeated text, where separate occurrences
    form a transitive chain of overlaps. Verify every proposed extension so a
    final span never gains unverified tokens merely through interval merging.
    """
    if not filt_spans:
        return []

    groups = [[0]]
    curr_start = filt_spans[0][0]
    curr_end = filt_spans[0][1]

    for i, (start, end, *_) in enumerate(filt_spans[1:], start=1):
        if start < curr_end:
            proposed_start = min(curr_start, start)
            proposed_end = max(curr_end, end)
            proposed_res = engine.find(
                input_ids=gen_ids[proposed_start:proposed_end]
            )
            if proposed_res.get("cnt", 0) > 0:
                groups[-1].append(i)
                curr_start = proposed_start
                curr_end = proposed_end
                continue

        groups.append([i])
        curr_start = start
        curr_end = end

    return groups


def _attach_full_span(
    final_spans: list[dict],
    *,
    generation: str,
    total_tokens: int,
    full_docs: list[dict],
) -> list[dict]:
    """Attach or enrich the full-generation span while keeping partial spans."""
    if not full_docs:
        return final_spans

    deduped_full_docs = []
    seen_docs = set()
    for doc in full_docs:
        doc_key = _doc_identity(doc)
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        deduped_full_docs.append(doc)

    for idx, span in enumerate(final_spans):
        if span.get("start") != 0 or span.get("end") != total_tokens:
            continue

        merged_docs = []
        seen_merged_docs = set()
        for doc in deduped_full_docs + span.get("docs", []):
            doc_key = _doc_identity(doc)
            if doc_key in seen_merged_docs:
                continue
            seen_merged_docs.add(doc_key)
            merged_docs.append(doc)

        updated_span = {
            "start": 0,
            "end": total_tokens,
            "text": generation,
            "docs": merged_docs,
        }
        if idx == 0:
            return [updated_span] + final_spans[1:]
        return [updated_span] + final_spans[:idx] + final_spans[idx + 1:]

    return [{
        "start": 0,
        "end": total_tokens,
        "text": generation,
        "docs": deduped_full_docs,
    }] + final_spans


def trace_generation(
    generation: str,
    engine: InfiniGramEngine,
    enc,
    unigram_probs: dict,
    docs_per_span: int = 10,
    match_mode: str = TEXT_MATCH_MODE,
    stop_event: threading.Event = None,
    executor: ThreadPoolExecutor | None = None,
    seed: int | None = DEFAULT_SEED,
) -> dict:
    """
    Run the full SimpleTrace pipeline for a single generation string.

    If `executor` is given, the Step 1 index queries of different suffixes
    run concurrently on its threads; results are identical either way.

    Spans found in more than `docs_per_span` documents get a random sample of
    them. The sample is drawn from an RNG seeded with (`seed`, generation), so
    results do not depend on worker scheduling or on which other generations
    are traced; `seed=None` uses the unseeded global RNG.

    Returns a dict with keys:
        "generation"  - original text
        "gen_ids"     - token ids
        "final_spans" - list of traced segment dicts (start, end, text, docs)
    """
    gen_ids = enc.encode(generation)
    L = len(gen_ids)
    # String seeds are hashed with SHA-512, so they are stable across processes.
    rng = None if seed is None else random.Random(f"{seed}:{generation}")
    max_doc_toks = L * 5 # retrieved docs can be 10 times as long as the generation
    full_span_docs: list[dict] = []

    if match_mode == MIXED_MATCH_MODE and gen_ids:
        full_match_res = engine.find(input_ids=gen_ids)
        if full_match_res.get("cnt", 0) > 0:
            full_span_docs = _retrieve_docs_for_find_result(
                generation,
                full_match_res,
                engine,
                enc,
                max_doc_toks=max_doc_toks,
                match_mode=match_mode,
                limit=docs_per_span,
                deterministic=True,
            )

    # ------------------------------------------------------------------
    # Step 1: Find maximal matching spans
    #
    # For every suffix of the tokenized generation (gen_ids[start:]),
    # query the InfiniGram index to find how much of it appears verbatim
    # in the training corpus:
    #   - cnt > 0: the full suffix exists → the entire suffix is a match.
    #   - cnt == 0: no exact match → the engine returns the position where
    #     the suffix would sit in the sorted suffix array. We fetch the
    #     documents of both neighbouring entries (just before and just
    #     after) and walk both sequences token-by-token with
    #     compute_longest_prefix() to find the longest prefix of the
    #     suffix that does appear somewhere in those documents.
    # Each suffix produces a candidate span (start, start + matched_toks).
    #
    # Suffixes are independent, so their queries may run concurrently on
    # `executor` threads (the engine releases the GIL). In text mode a span
    # must start with a word-initial token (filter pass 1b below), so the
    # other start positions are not queried at all.
    # ------------------------------------------------------------------
    starts = range(L - 1)
    if match_mode == TEXT_MATCH_MODE:
        starts = [
            start for start in starts
            if enc.convert_ids_to_tokens(gen_ids[start])[0] == '▁'
        ]

    lens = longest_prefix_lens(engine, gen_ids, starts, max_doc_toks, executor, stop_event)
    if lens is None:
        print(f"[INFO] Stop requested - aborting '{generation[:40]}...'", flush=True)
        return {
            "generation": generation,
            "gen_ids": gen_ids,
            "match_mode": match_mode,
            "final_spans": [],
        }
    spans = [(start, start + matched_toks) for start, matched_toks in zip(starts, lens)]

    # Filter pass 1 - retain only "clean", self-contained spans:
    #   a) No sentence-ending punctuation (! . ? newline) in the interior
    #      of the span, which would indicate the span crosses a sentence
    #      boundary and is likely an accidental match.
    #   b) The first token must be word-initial (Llama-2 BPE tokens that
    #      start a word carry the '▁' prefix), so spans start at a word
    #      boundary rather than mid-word.
    #   c) The token immediately after the span (if any) must also be
    #      word-initial, so the span ends at a clean word boundary.
    # In text mode, a span violating (a) or (c) is first trimmed back to the
    # sentence boundary (and then to a word boundary) instead of being
    # discarded, so a match running across sentences keeps its first sentence.
    has_boundary_char = None
    if match_mode == TEXT_MATCH_MODE:
        has_boundary_char = [
            any(ch in TEXT_MODE_BOUNDARY_CHARS for ch in enc.decode([tok]))
            for tok in gen_ids
        ]
    full_spans = []
    for start, end in spans:
        if match_mode == TEXT_MATCH_MODE:
            end = _trim_to_sentence_boundary(start, end, gen_ids, enc, has_boundary_char)
        if start >= end:
            continue
        span_ids = gen_ids[start:end]
        span_text = enc.decode(span_ids)
        if _keep_span(
            span_ids,
            span_text,
            start=start,
            end=end,
            total_tokens=L,
            enc=enc,
            gen_ids=gen_ids,
            match_mode=match_mode,
        ):
            full_spans.append((start, end, span_ids, span_text))

    # Filter pass 2 - keep only maximal spans.
    # Iterate spans in start-position order and greedily retain only those
    # that extend the furthest right end seen so far, discarding any span
    # entirely subsumed by an already-accepted one.
    maximal_spans = []
    max_end_pos = -1
    for start, end, ids, text in sorted(full_spans):
        if end > max_end_pos:
            maximal_spans.append((start, end, ids, text))
            max_end_pos = end

    # ------------------------------------------------------------------
    # Step 2: Filter by unigram probability - keep rarest K spans
    #
    # Not every maximal span is equally worth tracing - very common
    # sequences (e.g. "the cat sat") may match many documents by chance.
    # We score each span by how *unlikely* it is under a unigram LM,
    # then retain only the K least-probable (most distinctive) spans.
    #
    # K = ceil(5% of generation length), minimum 1, so longer generations
    # get proportionally more traced spans.
    #
    # The joint unigram probability is the product of each token's
    # precomputed marginal probability - a simple proxy for how surprising
    # the exact token sequence is under a bag-of-words model. Lower →
    # rarer → more likely to reflect genuine memorisation.
    # ------------------------------------------------------------------
    K = max(math.ceil(0.05 * L), 1)
    filt_spans = []
    for start, end, ids, text in maximal_spans:
        # Multiply per-token unigram probabilities; default to 1.0 for
        # unknown tokens so they don't spuriously inflate rarity.
        prob = math.prod(unigram_probs.get(_id, 1.0) for _id in ids)
        filt_spans.append((start, end, ids, text, prob))
    # Sort ascending by joint probability (rarest first) and keep top K.
    filt_spans = sorted(filt_spans, key=lambda x: x[-1])[:K]
    filt_spans = sorted(filt_spans)  # restore left-to-right positional order

    # ------------------------------------------------------------------
    # Step 3: Retrieve enclosing training documents for each span
    #
    # engine.find() returns the contiguous range of sorted-index ranks
    # [rank_start, rank_end) covering all corpus positions where the span
    # appears verbatim. Each rank maps to one training document.
    #
    # If the hit count exceeds docs_per_span we draw a random subsample
    # of that many ranks (without replacement) to keep retrieval cost
    # bounded regardless of how frequently the span appears.
    #
    # For each selected rank, get_doc_by_rank() returns up to max_doc_toks
    # tokens of the enclosing document. Metadata may be stored as either a
    # Python literal or JSON string, depending on how the index was built.
    # Results are stored in span_to_docs keyed by span index i.
    # ------------------------------------------------------------------
    span_to_docs = defaultdict(list)
    for i, (start, end, ids, text, _) in enumerate(filt_spans):
        span_res = engine.find(input_ids=ids)
        assert span_res['cnt'] > 0  # guaranteed: span came from Step 1

        span_to_docs[i].extend(
            _retrieve_docs_for_find_result(
                generation,
                span_res,
                engine,
                enc,
                max_doc_toks=max_doc_toks,
                match_mode=match_mode,
                limit=docs_per_span,
                deterministic=False,
                rng=rng,
            )
        )

    # ------------------------------------------------------------------
    # Step 4: Merge corpus-verified overlapping spans into final segments
    #
    # Overlapping filtered spans are collapsed only when the complete union
    # also occurs verbatim in the index. Interval overlap alone is not enough:
    # independently verified matches may refer to different corpus positions
    # (or documents), and their union may never have occurred in training.
    #
    # Documents for a verified union are retrieved using the union itself.
    # Combining the constituent document lists would incorrectly associate a
    # larger span with documents that support only one of its smaller parts.
    # ------------------------------------------------------------------
    if not filt_spans:
        if match_mode == MIXED_MATCH_MODE:
            if not full_span_docs:
                anchor_spans = _build_mixed_anchor_spans(spans, gen_ids, unigram_probs)
                full_span_docs = _collect_mixed_full_match_docs(
                    generation,
                    anchor_spans,
                    engine,
                    enc,
                    max_doc_toks=max_doc_toks,
                    docs_per_span=docs_per_span,
                )
            if full_span_docs:
                return {
                    "generation": generation,
                    "gen_ids": gen_ids,
                    "match_mode": match_mode,
                    "final_spans": _attach_full_span(
                        [],
                        generation=generation,
                        total_tokens=L,
                        full_docs=full_span_docs,
                    ),
                }
        return {
            "generation": generation,
            "gen_ids": gen_ids,
            "match_mode": match_mode,
            "final_spans": [],
        }

    merged_groups = _group_verified_overlapping_spans(
        filt_spans,
        gen_ids,
        engine,
    )

    final_spans = []
    for group in merged_groups:
        group_spans = [filt_spans[i] for i in group]
        seg_start = min(s[0] for s in group_spans)
        seg_end = max(s[1] for s in group_spans)
        if len(group) == 1:
            all_docs = span_to_docs[group[0]][:docs_per_span]
        else:
            union_res = engine.find(input_ids=gen_ids[seg_start:seg_end])
            assert union_res.get("cnt", 0) > 0  # guaranteed by grouping
            all_docs = _retrieve_docs_for_find_result(
                generation,
                union_res,
                engine,
                enc,
                max_doc_toks=max_doc_toks,
                match_mode=match_mode,
                limit=docs_per_span,
                deterministic=False,
                rng=rng,
            )
        final_spans.append({
            "start": seg_start,
            "end": seg_end,
            "text": enc.decode(gen_ids[seg_start:seg_end]),
            "docs": all_docs,
        })

    if match_mode == MIXED_MATCH_MODE:
        if not full_span_docs:
            anchor_spans = _build_mixed_anchor_spans(spans, gen_ids, unigram_probs)
            full_span_docs = _collect_mixed_full_match_docs(
                generation,
                anchor_spans,
                engine,
                enc,
                max_doc_toks=max_doc_toks,
                docs_per_span=docs_per_span,
            )
        final_spans = _attach_full_span(
            final_spans,
            generation=generation,
            total_tokens=L,
            full_docs=full_span_docs,
        )

    return {
        "generation": generation,
        "gen_ids": gen_ids,
        "match_mode": match_mode,
        "final_spans": final_spans,
    }


def print_results(result: dict) -> None:
    """Pretty-print the tracing results for one generation."""
    final_spans = result["final_spans"]
    tprint(f'\nQuery Text: {result["generation"]}')
    for i, sp in enumerate(final_spans):
        tprint("\n" + "=" * 20 + f" SPAN {i + 1} / {len(final_spans)} " + "=" * 20)
        tprint(f"Span Text: {sp['text']}\n")
        for j, doc in enumerate(sp['docs']):
            tprint("-" * 10 + f" Document {j + 1} / {len(sp['docs'])} " + "-" * 10)
            for k in ('text', 'id', 'match_tier', 'nv_recall'):
                raw_v = doc.get(k, "")
                v = raw_v.replace('\n', ' ') if k == 'text' and isinstance(raw_v, str) else raw_v
                tprint(f"- {k} --> {v}")

def evaluate_results(
    results: dict[str, dict],
    length_buckets: list[tuple[int, int]] = None,
    summary_output_path: str = "simpletrace_evaluation_summary.json",
    span_length_exact_output_path: str | None = None,
    nv_recall_threshold: float = 0.0,
    n_token_span_ratio: int = 60,
) -> dict:
    """
    Compute aggregate statistics over a batch of trace_generation results.

    Parameters
    ----------
    results : dict
        Output of launch_simpletrace - maps a key (the generation text unless
        launch_simpletrace got explicit keys) to its trace dict (keys:
        "generation", "gen_ids", "final_spans"). The text is taken from
        "generation", falling back to the key.
    length_buckets : list of (lo, hi) int pairs, inclusive on both ends.
        Token-length ranges for the distribution table.
        Defaults to [(1,3), (4,6), (7,10), (11,20), (21, inf)].

    Returns
    -------
    dict with keys:
        total_generations       - number of generations processed
        generations_with_spans  - generations that produced at least one span
        total_spans             - total final spans across all generations
        average_span_length     - average of the largest span length per generation
                                  (0 for generations with no spans)
        min_span_length         - minimum span length across all generations
                                  (0 when no spans exist)
        max_span                - maximum span length across all generations
        n_token_span_ratio      - token-length threshold used by
                                  generations_with_n_token_span_ratio
        generations_with_n_token_span_ratio
                                - fraction of generations that have at least one
                                  span with length >= n_token_span_ratio tokens
        generations_full_matches_ratio
                                - fraction of generations that have at least one
                                  retrieved doc containing the full generation
        total_docs              - total documents retrieved
        full_exact_matches      - total exact-match document hits (non-unique):
                                  docs where the *full* generation text is a
                                  substring of the document text
        partial_matches         - docs where only a span (not the full
                                  generation) matched
        unique_total_docs       - unique retrieved docs across all generations
        unique_full_matches     - unique docs that contain at least one full
                                  generation-text match
        unique_full_matches_ratio
                                - unique_full_matches / unique_total_docs
        unique_partial_matches  - unique docs that have at least one partial
                                  match across all generations
        avg_nv_recall           - mean adaptive nv-recall across all retrieved docs
        avg_nv_recall_on_hits   - mean adaptive nv-recall across retrieved doc
                                  occurrences where nv_recall > 0
        max_nv_recall           - maximum adaptive nv-recall observed across all docs
        docs_with_nv_recall     - number of docs with adaptive nv-recall > 0
        total_nv_matched_words  - total matched words for adaptive nv-recall
        generations_with_nv_recall
                                - number of generations with at least one
                                  retrieved doc where adaptive nv-recall > 0
        generations_ratio_with_nv_recall
                                - fraction of generations with at least one
                                  retrieved doc where adaptive nv-recall > 0
        generations_above_nv_recall_threshold
                                - number of generations with at least one
                                  retrieved doc where nv_recall >
                                  nv_recall_threshold
        generations_above_nv_recall_threshold_ratio
                                - fraction of generations with at least one
                                  retrieved doc where nv_recall >
                                  nv_recall_threshold
        docs_above_nv_recall_threshold
                                - number of retrieved docs with
                                  nv_recall > nv_recall_threshold
        doc_ids_above_nv_recall_threshold
                                - unique retrieved document ids with
                                  nv_recall > nv_recall_threshold
        full_exact_match_doc_ids
                                - unique exact-match document ids (first-seen
                                  order), where the full generation appears
                                  verbatim in the document
        spans_length_counts_distribution
                                - dict mapping "(lo, hi)" string keys to the
                                  number of documents whose span length in
                                  tokens falls within that range
        spans_length_distribution
                                - dict mapping "(lo, hi)" string keys to the
                                  percentage (0-1) of documents in each span
                                  length bucket
        spans_length_exact_output_path
                                - path of the separate JSON file containing
                                  exact per-length span distributions
    """
    if length_buckets is None:
        length_buckets = [(1, 3), (4, 6), (7, 10), (11, 20), (21, float("inf"))]

    # Pre-build bucket labels in order so the output dict is sorted.
    bucket_labels = [
        f"({lo}, {'inf' if hi == float('inf') else hi})"
        for lo, hi in length_buckets
    ]

    total_generations = len(results)
    generations_with_spans = 0
    total_spans = 0
    min_span_length = None
    max_span = 0
    sum_largest_span_per_generation = 0
    generations_with_n_token_span = 0
    generations_with_full_matches = 0
    generations_with_full_normalized_matches = 0
    total_docs = 0
    full_exact_matches = 0
    # Unique exact-match doc ids, preserving first-seen order.
    full_exact_match_doc_ids: list[str] = []
    _full_exact_match_doc_ids_set: set[str] = set()
    full_normalized_matches = 0
    full_normalized_match_doc_ids: list[str] = []
    _full_normalized_match_doc_ids_set: set[str] = set()
    partial_matches = 0
    _unique_total_doc_ids_set: set[str] = set()
    _unique_full_match_doc_ids_set: set[str] = set()
    _unique_full_normalized_match_doc_ids_set: set[str] = set()
    _unique_partial_match_doc_ids_set: set[str] = set()
    total_nv_recall = 0.0
    max_nv_recall = 0.0
    docs_with_nv_recall = 0
    total_nv_matched_words = 0
    generations_with_nv_recall = 0
    generations_above_nv_recall_threshold = 0
    doc_ids_above_nv_recall_threshold: list[str] = []
    _doc_ids_above_nv_recall_threshold_set: set[str] = set()
    spans_length_counts_distribution: dict[str, int] = {label: 0 for label in bucket_labels}
    spans_length_counts_exact: dict[int, int] = {}

    for key, result in results.items():
        generation = result.get("generation", key)
        final_spans = result.get("final_spans", [])
        largest_span_len_for_generation = 0
        generation_has_n_token_span = False
        generation_has_full_match = False
        generation_has_full_normalized_match = False
        generation_has_nv_recall = False
        generation_above_nv_recall_threshold = False
        if final_spans:
            generations_with_spans += 1
        total_spans += len(final_spans)

        for span in final_spans:
            span_len = span["end"] - span["start"]   # length in tokens
            min_span_length = span_len if min_span_length is None else min(min_span_length, span_len)
            largest_span_len_for_generation = max(largest_span_len_for_generation, span_len)
            max_span = max(max_span, span_len)
            if span_len >= n_token_span_ratio:
                generation_has_n_token_span = True
            span_docs = span.get("docs", [])
            total_docs += len(span_docs)

            # Find which bucket this span length falls into.
            for (lo, hi), label in zip(length_buckets, bucket_labels):
                if lo <= span_len <= hi:
                    spans_length_counts_distribution[label] += len(span_docs)
                    break
            spans_length_counts_exact[span_len] = (
                spans_length_counts_exact.get(span_len, 0) + len(span_docs)
            )

            for doc in span_docs:
                doc_text = doc.get("text", "")
                doc_id = str(doc.get("id", "")).strip()
                if doc_id:
                    _unique_total_doc_ids_set.add(doc_id)
                # Reuse precomputed adaptive nv-recall if present, else compute on the fly.
                nv_recall = doc.get("nv_recall")
                if nv_recall is None:
                    nv_stats = compute_nv_recall(generation, doc_text)
                    nv_recall = nv_stats["nv_recall"]
                    total_nv_matched_words += nv_stats["matched_words"]
                else:
                    nv_recall = _round_metric_float(nv_recall)
                    total_nv_matched_words += int(doc.get("nv_matched_words", 0))

                total_nv_recall += nv_recall
                max_nv_recall = max(max_nv_recall, nv_recall)
                if nv_recall > 0:
                    docs_with_nv_recall += 1
                    generation_has_nv_recall = True
                if nv_recall > nv_recall_threshold:
                    generation_above_nv_recall_threshold = True
                    if doc_id and doc_id not in _doc_ids_above_nv_recall_threshold_set:
                        _doc_ids_above_nv_recall_threshold_set.add(doc_id)
                        doc_ids_above_nv_recall_threshold.append(doc_id)

                match_tier = doc.get("match_tier")
                if match_tier is None:
                    match_tier = (
                        FULL_RAW_MATCH_TIER
                        if generation in doc_text
                        else PARTIAL_MATCH_TIER
                    )

                if match_tier == FULL_RAW_MATCH_TIER:
                    # The entire generation text appears verbatim in this doc.
                    full_exact_matches += 1
                    if doc_id:
                        _unique_full_match_doc_ids_set.add(doc_id)
                        if doc_id not in _full_exact_match_doc_ids_set:
                            _full_exact_match_doc_ids_set.add(doc_id)
                            full_exact_match_doc_ids.append(doc_id)
                    generation_has_full_match = True
                elif match_tier == FULL_NORMALIZED_MATCH_TIER:
                    full_normalized_matches += 1
                    if doc_id:
                        _unique_full_normalized_match_doc_ids_set.add(doc_id)
                        if doc_id not in _full_normalized_match_doc_ids_set:
                            _full_normalized_match_doc_ids_set.add(doc_id)
                            full_normalized_match_doc_ids.append(doc_id)
                    generation_has_full_normalized_match = True
                else:
                    # Only the span (a sub-sequence of the generation) matched.
                    partial_matches += 1
                    partial_doc_id = str(doc.get("id", "")).strip()
                    if partial_doc_id:
                        _unique_partial_match_doc_ids_set.add(partial_doc_id)

        sum_largest_span_per_generation += largest_span_len_for_generation
        if generation_has_n_token_span:
            generations_with_n_token_span += 1
        if generation_has_full_match:
            generations_with_full_matches += 1
        if generation_has_full_normalized_match:
            generations_with_full_normalized_matches += 1
        if generation_has_nv_recall:
            generations_with_nv_recall += 1
        if generation_above_nv_recall_threshold:
            generations_above_nv_recall_threshold += 1

    avg_nv_recall = (
        _round_metric_float(total_nv_recall / total_docs) if total_docs > 0 else 0.0
    )
    avg_nv_recall_on_hits = (
        _round_metric_float(total_nv_recall / docs_with_nv_recall)
        if docs_with_nv_recall > 0
        else 0.0
    )
    generations_ratio_with_nv_recall = (
        _round_metric_float(generations_with_nv_recall / total_generations)
        if total_generations > 0 else 0.0
    )
    generations_above_nv_recall_threshold_ratio = (
        _round_metric_float(generations_above_nv_recall_threshold / total_generations)
        if total_generations > 0 else 0.0
    )
    docs_above_nv_recall_threshold = len(_doc_ids_above_nv_recall_threshold_set)
    unique_total_docs = len(_unique_total_doc_ids_set)
    unique_full_matches = len(_unique_full_match_doc_ids_set)
    unique_full_matches_ratio = (
        _round_metric_float(unique_full_matches / unique_total_docs)
        if unique_total_docs > 0 else 0.0
    )
    unique_full_normalized_matches = len(_unique_full_normalized_match_doc_ids_set)
    unique_full_normalized_matches_ratio = (
        _round_metric_float(unique_full_normalized_matches / unique_total_docs)
        if unique_total_docs > 0 else 0.0
    )
    unique_partial_matches = len(_unique_partial_match_doc_ids_set)
    min_span_length = 0 if min_span_length is None else min_span_length
    average_longest_span_length = (
        _round_metric_float(sum_largest_span_per_generation / total_generations)
        if total_generations > 0 else 0.0
    )
    generations_with_n_token_span_ratio = (
        _round_metric_float(generations_with_n_token_span / total_generations)
        if total_generations > 0 else 0.0
    )
    generations_full_matches_ratio = (
        _round_metric_float(generations_with_full_matches / total_generations)
        if total_generations > 0 else 0.0
    )
    generations_full_normalized_matches_ratio = (
        _round_metric_float(generations_with_full_normalized_matches / total_generations)
        if total_generations > 0 else 0.0
    )
    spans_length_distribution = {
        label: (_round_metric_float(count / total_docs)) if total_docs > 0 else 0.0
        for label, count in spans_length_counts_distribution.items()
    }
    spans_length_counts_exact_sorted = {
        str(span_len): spans_length_counts_exact[span_len]
        for span_len in sorted(spans_length_counts_exact)
    }
    spans_length_distribution_exact = {
        span_len: (_round_metric_float(count / total_docs)) if total_docs > 0 else 0.0
        for span_len, count in spans_length_counts_exact_sorted.items()
    }

    if span_length_exact_output_path:
        spans_length_exact_payload = {
            "total_docs": total_docs,
            "spans_length_counts_exact": spans_length_counts_exact_sorted,
            "spans_length_distribution_exact": spans_length_distribution_exact,
        }
        os.makedirs(os.path.dirname(span_length_exact_output_path) or ".", exist_ok=True)
        with open(span_length_exact_output_path, "w") as f:
            json.dump(spans_length_exact_payload, f, indent=4)

    # save eval results to a JSON file
    results = {
        "total_generations": total_generations,
        "generations_with_spans": generations_with_spans,
        "total_spans": total_spans,
        "average_span_length": average_longest_span_length,
        "average_longest_span_length": average_longest_span_length,
        "min_span_length": min_span_length,
        "max_span": max_span,
        "max_span_length": max_span,
        "n_token_span_ratio": n_token_span_ratio,
        "generations_with_n_token_span_ratio": generations_with_n_token_span_ratio,
        "generations_full_matches_ratio": generations_full_matches_ratio,
        "total_docs": total_docs,
        "unique_total_docs": unique_total_docs,
        "full_exact_matches": full_exact_matches,
        "unique_full_exact_matches": unique_full_matches,
        "unique_full_matches": unique_full_matches,
        "unique_full_matches_ratio": unique_full_matches_ratio,
        "full_normalized_matches": full_normalized_matches,
        "unique_full_normalized_matches": unique_full_normalized_matches,
        "unique_full_normalized_matches_ratio": unique_full_normalized_matches_ratio,
        "partial_matches": partial_matches,
        "unique_partial_matches": unique_partial_matches,
        "avg_nv_recall": avg_nv_recall,
        "avg_nv_recall_on_hits": avg_nv_recall_on_hits,
        "max_nv_recall": _round_metric_float(max_nv_recall),
        "docs_with_nv_recall": docs_with_nv_recall,
        "total_nv_matched_words": total_nv_matched_words,
        "generations_with_nv_recall": generations_with_nv_recall,
        "generations_with_nv_recall_ratio": generations_ratio_with_nv_recall,
        "generations_above_nv_recall_threshold": generations_above_nv_recall_threshold,
        "generations_above_nv_recall_threshold_ratio": generations_above_nv_recall_threshold_ratio,
        "nv_recall_threshold": _round_metric_float(nv_recall_threshold),
        "docs_above_nv_recall_threshold": docs_above_nv_recall_threshold,
        "spans_length_counts_distribution": spans_length_counts_distribution,
        "spans_length_distribution": spans_length_distribution,
        "spans_length_exact_output_path": span_length_exact_output_path,
        "generations_full_normalized_matches_ratio": generations_full_normalized_matches_ratio,
        "full_exact_match_doc_ids": full_exact_match_doc_ids,
        "full_normalized_match_doc_ids": full_normalized_match_doc_ids,
        "doc_ids_above_nv_recall_threshold": doc_ids_above_nv_recall_threshold,
    }

    # Backward-compatible alias used by existing reports/plots.
    if n_token_span_ratio == 60:
        results["generations_with_60_token_span_ratio"] = generations_with_n_token_span_ratio

    _move_summary_doc_id_fields_to_end(results)

    os.makedirs(os.path.dirname(summary_output_path) or ".", exist_ok=True)
    with open(summary_output_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def save_results(
    results: dict,
    output_path: str,
    record_fields: dict | None = None,
) -> None:
    """Save tracing results to a JSONL file, one line per generation.

    `record_fields` optionally maps a results key to extra fields written
    after "generation" (e.g. the prompt_id and sample_idx of the generation).

    Each line is a JSON object with:
        "generation" - the query text
        "match_mode" - tracing mode used for this generation
        "spans"      - list of traced segments, each containing:
            "start", "end", "span_length", "text" - token boundaries, token length, and decoded text
            "docs"                 - list of retrieved training documents,
                                     each with "text", "id", "match_tier", and adaptive nv-* metrics
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for key, result in results.items():
            sorted_spans = sorted(
                result["final_spans"],
                key=lambda sp: (sp["end"] - sp["start"]),
                reverse=True,
            )
            record = {
                "generation": result.get("generation", key),
                **(record_fields or {}).get(key, {}),
                "match_mode": result.get("match_mode", TEXT_MATCH_MODE),
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
                                "match_tier": doc.get("match_tier", PARTIAL_MATCH_TIER),
                                "nv_recall": _round_metric_float(doc.get("nv_recall", 0.0)),
                                "nv_matched_words": doc.get("nv_matched_words", 0),
                                "nv_reference_words": doc.get("nv_reference_words", 0),
                                "nv_candidate_words": doc.get("nv_candidate_words", 0),
                                "nv_missing_words": doc.get("nv_missing_words", 0),
                                "nv_additional_words": doc.get("nv_additional_words", 0),
                            }
                            for doc in sp["docs"]
                        ],
                    }
                    for sp in sorted_spans
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Per-process globals – initialised once per worker by _worker_init().
# ProcessPoolExecutor forks a fresh interpreter per worker, so these are
# independent copies; the mmap'd index is shared at OS page-cache level.
# ---------------------------------------------------------------------------
_worker_engine = None
_worker_enc = None
_worker_unigram_probs = None
_worker_executor = None

DEFAULT_MAX_PAGE_TABLE_GB = 1.0

def load_engine(index_dir, eos_token_id: int) -> InfiniGramEngine:
    """Open the InfiniGram index with disk prefetching turned off.

    Prefetching only issues speculative reads and does not change any query
    result. Turning it off made tracing ~8x faster on the Dolma3 indexes
    (see also OLMoTrace, arXiv:2504.07096, App. B).
    """
    return InfiniGramEngine(
        index_dir=index_dir,
        eos_token_id=eos_token_id,
        precompute_unigram_logprobs=False,
        ds_prefetch_depth=0,
        sa_prefetch_depth=0,
        od_prefetch_depth=0,
    )


def page_table_bytes() -> int:
    """Size of this process's page tables (VmPTE in /proc/self/status), or 0."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPTE:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class PageTableBoundedEngine:
    """InfiniGramEngine that reopens its index when page tables grow too large.

    The index files are memory-mapped, and every 2 MB region a lookup touches
    costs a 4 KB page-table page that the kernel cannot reclaim while the files
    stay mapped. A lookup binary-searches every shard, so on a multi-TB index
    such as Dolma3 (65 shards) one uncached find() can add ~20 MB of page
    tables, and a long run grows them until the container runs out of memory.

    After each call this wrapper checks the process's page tables and, above
    `max_page_table_bytes`, replaces the engine with a fresh one, which unmaps
    the files and frees the tables. The C++ engine crashes if an engine is
    created or destroyed while other threads are inside it, so the swap waits
    until no call is in flight and holds new calls back meanwhile. Calls made
    on the C++ engine through the `engine` attribute are tracked the same way.
    The engine holds no query state and cached index pages stay in the OS page
    cache, so results are unchanged and reopening is cheap.
    """

    _MIN_CALLS_BETWEEN_REOPENS = 32

    def __init__(self, index_dir, eos_token_id: int, max_page_table_bytes: float | None):
        self._index_dir = index_dir
        self._eos_token_id = eos_token_id
        self._max_page_table_bytes = max_page_table_bytes
        self._engine = load_engine(index_dir, eos_token_id)
        self._cond = threading.Condition()
        self._in_flight = 0
        self._reopening = False
        self._calls_since_reopen = 0
        self.reopen_count = 0

    @property
    def engine(self):
        """Tracked view of the underlying C++ engine (used by _get_docs_by_ranks)."""
        return _TrackedCppEngine(self)

    def __getattr__(self, name):
        if callable(getattr(InfiniGramEngine, name, None)):
            return lambda *args, **kwargs: self._call(lambda eng: eng, name, args, kwargs)
        with self._cond:
            while self._reopening:
                self._cond.wait()
            return getattr(self._engine, name)

    def _call(self, resolve, name: str, args, kwargs):
        with self._cond:
            while self._reopening:
                self._cond.wait()
            self._in_flight += 1
            engine = self._engine
        try:
            return getattr(resolve(engine), name)(*args, **kwargs)
        finally:
            with self._cond:
                self._in_flight -= 1
                self._calls_since_reopen += 1
                self._cond.notify_all()
            self._maybe_reopen()

    def _maybe_reopen(self) -> None:
        if self._max_page_table_bytes is None:
            return
        with self._cond:
            if self._reopening or self._calls_since_reopen < self._MIN_CALLS_BETWEEN_REOPENS:
                return
            if page_table_bytes() <= self._max_page_table_bytes:
                return
            self._reopening = True
            while self._in_flight:
                self._cond.wait()
            old_engine, self._engine = self._engine, None
            del old_engine
            self._engine = load_engine(self._index_dir, self._eos_token_id)
            self._calls_since_reopen = 0
            self.reopen_count += 1
            self._reopening = False
            self._cond.notify_all()


class _TrackedCppEngine:
    """Forwards calls to the current C++ engine of a PageTableBoundedEngine."""

    def __init__(self, owner: PageTableBoundedEngine):
        self._owner = owner

    def __getattr__(self, name):
        return lambda *args, **kwargs: self._owner._call(lambda eng: eng.engine, name, args, kwargs)


def _worker_init(
    index_dir: str,
    unigram_probs_path: str,
    metric_decimals: int,
    find_threads: int = 1,
    max_page_table_gb: float | None = DEFAULT_MAX_PAGE_TABLE_GB,
):
    """Called once in each worker process before any tasks are dispatched."""
    global _worker_engine, _worker_enc, _worker_unigram_probs, _worker_executor
    _set_metric_decimals(metric_decimals)
    _worker_executor = ThreadPoolExecutor(find_threads) if find_threads > 1 else None
    _worker_enc = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False
    )
    _worker_engine = PageTableBoundedEngine(
        index_dir,
        _worker_enc.eos_token_id,
        None if max_page_table_gb is None else max_page_table_gb * 1e9,
    )
    with open(unigram_probs_path) as f:
        _worker_unigram_probs = {int(k): v['prob'] for k, v in json.load(f).items()}


def _worker_trace(generation: str, docs_per_span: int, match_mode: str, seed: int | None) -> dict:
    """Thin wrapper that calls trace_generation using worker-local globals.    stop_event is not passed - worker processes receive SIGINT directly from
    the OS when Ctrl+C is pressed, which raises KeyboardInterrupt naturally.
    """
    return trace_generation(
        generation, _worker_engine, _worker_enc, _worker_unigram_probs,
        docs_per_span=docs_per_span, match_mode=match_mode, stop_event=None,
        executor=_worker_executor, seed=seed,
    )


def launch_simpletrace(
    index_dir: str,
    generations: list[str],
    unigram_probs_path: str,
    num_workers: int = 8,
    docs_per_span: int = 10,
    match_mode: str = TEXT_MATCH_MODE,
    metric_decimals: int = DEFAULT_METRIC_DECIMALS,
    enable_print: bool = False,
    find_threads: int = 4,
    max_page_table_gb: float | None = DEFAULT_MAX_PAGE_TABLE_GB,
    seed: int | None = DEFAULT_SEED,
    keys: list | None = None,
) -> dict:
    """Trace `generations` in worker processes; returns {key: trace dict} in input order.

    `keys` gives each generation a unique results key. By default the key is
    the generation text, so identical generations share a single result.
    """
    if keys is None:
        keys = generations
    elif len(keys) != len(generations):
        raise ValueError(f"Got {len(keys)} keys for {len(generations)} generations")
    elif len(set(keys)) != len(keys):
        raise ValueError("keys must be unique")

    # ---------------------------------------------------------------------------
    # Setup – engine and tokenizer are initialised inside each worker process
    # via _worker_init(), so they don't need to be pickled or sent over IPC.
    # ---------------------------------------------------------------------------
    start_time_local = time.time()

    # ---------------------------------------------------------------------------
    # Signal handling: installing a SIGINT handler guarantees Ctrl+C is caught
    # at the OS level immediately, even when the main thread is blocked inside a
    # C-extension call (where KeyboardInterrupt would not be raised until the
    # call returns). The handler sets a threading.Event that the polling loop
    # checks after each short-timeout wake-up.
    # ---------------------------------------------------------------------------
    _stop_event = threading.Event()

    def _sigint_handler(sig, frame):
        _stop_event.set()

    previous_sigint_handler = signal.signal(signal.SIGINT, _sigint_handler)

    results: dict[str, dict] = {}

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(index_dir, unigram_probs_path, metric_decimals, find_threads, max_page_table_gb),
    )

    futures = {
        executor.submit(_worker_trace, gen, docs_per_span, match_mode, seed): key
        for key, gen in zip(keys, generations)
    }
    pending = set(futures)
    try:
        with tqdm(total=len(generations), desc="Tracing", unit="gen") as pbar:
            while pending and not _stop_event.is_set():
                # Short timeout ensures the loop wakes up frequently to check
                # _stop_event, making Ctrl+C feel near-instant.
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures[future]
                    try:
                        results[key] = future.result()
                    except CancelledError:
                        pass
                    pbar.update(1)
    except KeyboardInterrupt:
        # Fallback in case the signal handler is not called in this environment.
        _stop_event.set()
    finally:
        if _stop_event.is_set():
            tprint("\n[INFO] Ctrl+C received - requesting stop...")
            for f in pending:
                f.cancel()
            # Workers still inside a trace would otherwise keep running, and
            # the interpreter joins them at exit, so the script would hang.
            for process in list((executor._processes or {}).values()):
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        # Restore the caller's handler so a later Ctrl+C interrupts normally.
        signal.signal(signal.SIGINT, previous_sigint_handler)

    end_time = time.time()
    print(f"\nElapsed Time: {end_time - start_time_local:.2f} seconds  ({len(generations)} generation(s), {num_workers} workers, {find_threads} find threads each)")

    # Return (and print) in submission order rather than completion order, so
    # the results file and the doc-id lists in the summary are reproducible.
    results = {key: results[key] for key in dict.fromkeys(keys) if key in results}
    if enable_print:
        for result in results.values():
            print_results(result)

    return results


def parse_length_buckets(raw: str) -> list[tuple[int, int]]:
    """Parse a bucket string like '1-3,4-6,7-10,11-20,21-inf'."""
    buckets: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid bucket '{item}'. Expected format 'lo-hi'.")
        lo_s, hi_s = item.split("-", 1)
        lo = int(lo_s)
        hi = float("inf") if hi_s.lower() == "inf" else int(hi_s)
        if lo < 1:
            raise ValueError(f"Invalid bucket '{item}'. Lower bound must be >= 1.")
        if hi != float("inf") and hi < lo:
            raise ValueError(f"Invalid bucket '{item}'. Upper bound must be >= lower bound.")
        buckets.append((lo, hi))

    if not buckets:
        raise ValueError("At least one valid length bucket must be provided.")
    return buckets


def resolve_output_path(path: str) -> str:
    """Return the output path exactly as provided (must include a filename)."""
    filename = os.path.basename(path.strip())
    if not filename:
        raise ValueError("Output path must include a file name.")
    return path


def build_span_length_exact_output_path(summary_output_path: str) -> str:
    """Build a default output path for exact span-length distribution stats."""
    base, ext = os.path.splitext(summary_output_path)
    if not ext:
        return f"{summary_output_path}_spans_length_exact.json"
    return f"{base}_spans_length_exact{ext}"


def load_generations(
    dataset_name: str,
    limit: int | None,
    is_jsonl: bool = False,
    text_field: str = 'text',
    is_generation_json: bool = False,
    generation_text_field: str = 'completion',
) -> list[str]:
    """
    Load generation strings from a supported dataset name.
    If is_jsonl is True, dataset_name is treated as a file path to a JSONL dataset,
    and text_field specifies which field contains the text to consider.
    If is_generation_json is True, dataset_name is treated as a path to a JSON file
    containing model generation results, and generation_text_field specifies which
    field to extract from each generation item.
    """
    if dataset_name == "laerebogen":
        return load_laerebogen(limit=limit)
    elif dataset_name == "dummy":
        return load_dummy_dataset()
    elif dataset_name == "generic":
        return load_generic_dataset()
    elif is_generation_json:
        return load_generation_dataset(dataset_name, limit=limit, text_field=generation_text_field)
    elif is_jsonl:
        return load_jsonl_dataset(dataset_name, limit=limit, text_field=text_field)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_trace_inputs(args: argparse.Namespace) -> tuple[list[str], list, dict]:
    """Generations to trace, a unique results key for each, and per-key fields for the results file.

    Generation JSON records that carry prompt_id and sample_idx (as written by
    generate_vllm.py and generate_vllm_free.py) are keyed "<prompt_id>:<sample_idx>";
    everything else is keyed by its position. Keying by text instead would merge
    identical generations, e.g. two samples of one prompt that came out the same.
    """
    if args.is_generation_json and args.dataset not in ("laerebogen", "dummy", "generic"):
        records = load_generation_records(args.dataset, args.generation_text_field, args.limit)
        strings = [record[args.generation_text_field] for record in records]
        if all("prompt_id" in r and "sample_idx" in r for r in records):
            keys = [f"{r['prompt_id']}:{r['sample_idx']}" for r in records]
            if len(set(keys)) == len(keys):
                fields = {
                    key: {"prompt_id": r["prompt_id"], "sample_idx": r["sample_idx"]}
                    for key, r in zip(keys, records)
                }
                return strings, keys, fields
            print("[WARN] prompt_id:sample_idx pairs are not unique; keying generations by position", flush=True)
    else:
        strings = load_generations(
            args.dataset,
            args.limit,
            is_jsonl=args.is_jsonl,
            text_field=args.text_field,
            is_generation_json=args.is_generation_json,
            generation_text_field=args.generation_text_field,
        )
    keys = list(range(len(strings)))
    return strings, keys, {key: {"index": key} for key in keys}


def parse_k_values(raw: str | None) -> list[int]:
    """Parse comma-separated positive integer k values, e.g. '1,5,10'."""
    if raw is None or not raw.strip():
        return []

    values: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        k = int(token)
        if k < 1:
            raise ValueError(f"Invalid k value '{token}'. k must be >= 1.")
        values.append(k)

    if not values:
        return []
    return sorted(set(values))


def count_distinct_examples_with_string(
    text: str,
    engine: InfiniGramEngine,
    enc,
    *,
    max_examples: int | None = None,
) -> int:
    """Count distinct training examples that contain `text` exactly.

    Distinctness is computed at document/example granularity (doc_ix per shard),
    matching the k-eidetic definition that counts examples rather than raw
    occurrence frequency.
    """
    token_ids = enc.encode(text, add_special_tokens=False)
    if not token_ids:
        return 0

    find_res = engine.find(input_ids=token_ids)
    if find_res.get("cnt", 0) == 0:
        return 0

    seen_examples: set[tuple[int, int]] = set()
    for shard_idx, (rank_start, rank_end) in enumerate(find_res["segment_by_shard"]):
        for rank in range(rank_start, rank_end):
            doc = engine.get_doc_by_rank(s=shard_idx, rank=rank, max_disp_len=1)
            doc_ix = doc.get("doc_ix")
            if doc_ix is None:
                # Fallback key if doc_ix is unavailable for any reason.
                example_key = (shard_idx, int(rank))
            else:
                example_key = (shard_idx, int(doc_ix))

            if example_key in seen_examples:
                continue

            seen_examples.add(example_key)
            if max_examples is not None and len(seen_examples) > max_examples:
                return max_examples + 1

    return len(seen_examples)


def evaluate_k_eidetic_memorization(
    generations: list[str],
    engine: InfiniGramEngine,
    enc,
    *,
    k_values: list[int],
    min_generation_chars: int = 1,
    min_generation_tokens: int = 1,
) -> dict:
    """Evaluate post-hoc k-eidetic memorization over generated strings.

    This function assumes each provided generation is already extracted from the
    model (i.e., produced as a continuation). It then checks how many distinct
    training examples contain each generated string.
    """
    if not k_values:
        return {}

    unique_generations = list(dict.fromkeys(generations))
    max_k = max(k_values)

    eligible_generations: list[str] = []
    for generation in unique_generations:
        if len(generation) < min_generation_chars:
            continue
        gen_token_count = len(enc.encode(generation, add_special_tokens=False))
        if gen_token_count < min_generation_tokens:
            continue
        eligible_generations.append(generation)

    counts_by_k = {k: 0 for k in k_values}
    not_in_index = 0
    doc_freq_hist: dict[str, int] = {}

    for generation in tqdm(eligible_generations, desc="k-eidetic", unit="gen"):
        # Early-stop counting once we know df > max(k_values).
        doc_freq = count_distinct_examples_with_string(
            generation,
            engine,
            enc,
            max_examples=max_k,
        )
        if doc_freq == 0:
            not_in_index += 1

        bucket_key = str(doc_freq) if doc_freq <= max_k else f">{max_k}"
        doc_freq_hist[bucket_key] = doc_freq_hist.get(bucket_key, 0) + 1

        for k in k_values:
            # k-eidetic requires presence in training data:
            # count only strings with 1 <= doc_freq <= k.
            if 1 <= doc_freq <= k:
                counts_by_k[k] += 1

    denom = len(eligible_generations)
    summary = {
        "k_eidetic_k_values": k_values,
        "k_eidetic_total_input_generations": len(generations),
        "k_eidetic_unique_generations": len(unique_generations),
        "k_eidetic_eligible_generations": denom,
        "k_eidetic_not_found_in_index": not_in_index,
        "k_eidetic_doc_frequency_histogram_capped": doc_freq_hist,
    }
    for k in k_values:
        count = counts_by_k[k]
        summary[f"k_eidetic_count_k_le_{k}"] = count
        summary[f"k_eidetic_rate_k_le_{k}"] = (
            _round_metric_float(count / denom) if denom > 0 else 0.0
        )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimpleTrace over a generation dataset.")
    parser.add_argument(
        "--dataset",
        default="dummy",
        help="Name of the dataset loader to use. Could be one implemented in data_loading.py (e.g. 'dummy'), a path to a JSONL file if --is-jsonl is set, or a path to a model generation JSON file if --is-generation-json is set. The loaded strings will be used as the generations to trace against the indexed corpus.",
    )
    parser.add_argument(
        "--is-jsonl",
        action="store_true",
        help="If set, --dataset is treated as a path to a JSONL file rather than a named dataset. The JSONL file should have one JSON object per line, and the text to be traced should be in the field specified by --text-field.",
    )
    parser.add_argument(
        "--is-generation-json",
        action="store_true",
        help="If set, --dataset is treated as a JSON file containing generation results (e.g. test_generations.json). Text is extracted from the field specified by --generation-text-field.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name containing the text to be traced in the JSONL file.",
    )
    parser.add_argument(
        "--generation-text-field",
        default="completion",
        help="Field name containing generation text in the generation JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of generations to process.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help=(
            "Trace only shard K of N (1 <= K <= N): every N-th generation, starting with the K-th. "
            "Run all N shards, then combine their results files with --merge-results."
        ),
    )
    parser.add_argument(
        "--merge-results",
        nargs="+",
        default=None,
        metavar="RESULTS",
        help=(
            "Skip tracing: rebuild the results of --dataset from these results files (e.g. the "
            "--shard runs), then evaluate and save them as a single run would."
        ),
    )
    parser.add_argument(
        "--index-dir",
        nargs="+",
        required=True,
        help=(
            "Path to one or more InfiniGram index directories. Multiple paths "
            "are loaded as one logical multi-shard index."
        ),
    )
    parser.add_argument(
        "--unigram-probs-path",
        default="02_unigram_probs/unigram_probs_dummy.json",
        help="Path to token unigram probabilities JSON.",
    )
    parser.add_argument(
        "--metric-decimals",
        type=int,
        default=DEFAULT_METRIC_DECIMALS,
        help="Decimal precision used for fractional metrics in saved results and summaries.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--max-page-table-gb",
        type=float,
        default=DEFAULT_MAX_PAGE_TABLE_GB,
        help=(
            "Reopen a worker's index once its page tables exceed this many GB "
            "(checked after each index call). Bounds memory on huge indexes "
            "such as Dolma3; does not change results. Total is about "
            "--num-workers times this value."
        ),
    )
    parser.add_argument(
        "--find-threads",
        type=int,
        default=4,
        help=(
            "Threads per worker process running the index queries of one "
            "generation concurrently. Does not change results. Helps most when "
            "the index is not yet in the OS page cache; use 1 for repeated runs "
            "over the same generations."
        ),
    )
    parser.add_argument(
        "--docs-per-span",
        type=int,
        default=10,
        help="Maximum number of retrieved documents per span.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Seed for sampling --docs-per-span documents from spans with more "
            "matches. Each generation gets its own RNG seeded with (seed, text), "
            "so results are reproducible regardless of workers or threads."
        ),
    )
    parser.add_argument(
        "--match-mode",
        choices=[TEXT_MATCH_MODE, MIXED_MATCH_MODE],
        default=TEXT_MATCH_MODE,
        help=(
            "Tracing mode. 'text' keeps the original prose-oriented span filters; "
            "'mixed' also supports full text, code, math, and mixed-content documents."
        ),
    )
    parser.add_argument(
        "--enable-print",
        action="store_true",
        help="Print detailed span/document matches to stdout.",
    )
    parser.add_argument(
        "--results-output",
        default="simpletrace_results_test.jsonl",
        help="Output filename for JSONL tracing results.",
    )
    parser.add_argument(
        "--summary-output",
        default="simpletrace_evaluation_summary_test.json",
        help="Output filename for evaluation summary JSON.",
    )
    parser.add_argument(
        "--span-length-exact-output",
        default="",
        help=(
            "Optional output filename for exact span-length distribution JSON. "
            "If empty, it is auto-derived from --summary-output."
        ),
    )
    parser.add_argument(
        "--length-buckets",
        default="1-3,4-6,7-10,11-20,21-30,31-40,51-60,61-70,71-80,81-90,91-100,101-inf",
        help="Comma-separated token-length buckets as 'lo-hi' (use 'inf' for open upper bound).",
    )
    parser.add_argument(
        "--nv-recall-threshold",
        type=float,
        default=0.5,
        help="Threshold used to count/list docs with adaptive nv_recall > threshold in evaluation summary.",
    )
    parser.add_argument(
        "--n-token-span-ratio",
        dest="n_token_span_ratio",
        type=int,
        default=60,
        help=(
            "Token-length threshold N used in generations_with_n_token_span_ratio: "
            "fraction of generations with at least one span of length >= N."
        ),
    )
    parser.add_argument(
        "--k-eidetic-values",
        default="",
        help=(
            "Optional comma-separated k values for post-hoc k-eidetic memorization "
            "evaluation (e.g. '1,5,10'). Empty disables this evaluation."
        ),
    )
    parser.add_argument(
        "--k-eidetic-min-generation-chars",
        type=int,
        default=1,
        help="Minimum character length for generations included in k-eidetic evaluation.",
    )
    parser.add_argument(
        "--k-eidetic-min-generation-tokens",
        type=int,
        default=1,
        help="Minimum token length for generations included in k-eidetic evaluation.",
    )
    return parser


def parse_shard(raw: str) -> tuple[int, int]:
    """'K/N' -> (K, N), with 1 <= K <= N."""
    try:
        k, n = (int(part) for part in raw.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like K/N, got {raw!r}") from None
    if not 1 <= k <= n:
        raise ValueError(f"--shard needs 1 <= K <= N, got {raw!r}")
    return k, n


def load_results_files(paths: list[str], keys: list, record_fields: dict) -> dict:
    """Results of `keys`, in that order, rebuilt from save_results files (e.g. one per --shard).

    A record is matched to its key by the fields save_results wrote for it (prompt_id and
    sample_idx, or index). Each key must appear exactly once across the files.
    """
    key_of_fields = {json.dumps(fields, sort_keys=True): key for key, fields in record_fields.items()}
    loaded: dict = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                fields = {name: record[name] for name in next(iter(record_fields.values()))}
                key = key_of_fields.get(json.dumps(fields, sort_keys=True))
                if key is None:
                    raise ValueError(f"{path}: result {fields} is not a generation of --dataset")
                if key in loaded:
                    raise ValueError(f"{path}: result {fields} appears more than once")
                loaded[key] = {
                    "generation": record["generation"],
                    "match_mode": record["match_mode"],
                    # save_results orders spans by length; tracing yields them by position.
                    "final_spans": [
                        {"start": sp["start"], "end": sp["end"], "text": sp["text"], "docs": sp["docs"]}
                        for sp in sorted(record["spans"], key=lambda sp: sp["start"])
                    ],
                }
    missing = [key for key in keys if key not in loaded]
    if missing:
        raise ValueError(f"{len(missing)} generations of --dataset have no result in --merge-results, e.g. {missing[:3]}")
    return {key: loaded[key] for key in keys}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if args.find_threads < 1:
        raise ValueError("--find-threads must be >= 1")
    if args.max_page_table_gb <= 0:
        raise ValueError("--max-page-table-gb must be > 0")
    if args.metric_decimals < 0:
        raise ValueError("--metric-decimals must be >= 0")
    if args.docs_per_span < 1:
        raise ValueError("--docs-per-span must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    if not (0.0 <= args.nv_recall_threshold <= 1.0):
        raise ValueError("--nv-recall-threshold must be between 0 and 1")
    if args.n_token_span_ratio < 1:
        raise ValueError("--n-token-span-ratio must be >= 1")
    if args.k_eidetic_min_generation_chars < 1:
        raise ValueError("--k-eidetic-min-generation-chars must be >= 1")
    if args.k_eidetic_min_generation_tokens < 1:
        raise ValueError("--k-eidetic-min-generation-tokens must be >= 1")
    if args.is_jsonl and args.is_generation_json:
        raise ValueError("--is-jsonl and --is-generation-json are mutually exclusive")
    if args.shard and args.merge_results:
        raise ValueError("--shard and --merge-results are mutually exclusive")

    strings, keys, record_fields = load_trace_inputs(args)
    if args.shard:
        k, n = parse_shard(args.shard)
        picked = range(k - 1, len(strings), n)
        strings = [strings[i] for i in picked]
        keys = [keys[i] for i in picked]
        record_fields = {key: record_fields[key] for key in keys}
        print(f"[INFO] Shard {k}/{n}: {len(strings)} generations", flush=True)
    _set_metric_decimals(args.metric_decimals)
    length_buckets = parse_length_buckets(args.length_buckets)
    k_values = parse_k_values(args.k_eidetic_values)
    results_output_path = resolve_output_path(args.results_output)
    summary_output_path = resolve_output_path(args.summary_output)
    span_length_exact_output_path = (
        resolve_output_path(args.span_length_exact_output)
        if args.span_length_exact_output.strip()
        else build_span_length_exact_output_path(summary_output_path)
    )

    index_dir = args.index_dir[0] if len(args.index_dir) == 1 else args.index_dir

    if args.merge_results:
        results = load_results_files(args.merge_results, keys, record_fields)
        print(f"[INFO] Merged {len(results)} results from {len(args.merge_results)} files", flush=True)
    else:
        results = launch_simpletrace(
            index_dir=index_dir,
            generations=strings,
            unigram_probs_path=args.unigram_probs_path,
            num_workers=args.num_workers,
            docs_per_span=args.docs_per_span,
            match_mode=args.match_mode,
            metric_decimals=args.metric_decimals,
            enable_print=args.enable_print,
            find_threads=args.find_threads,
            max_page_table_gb=args.max_page_table_gb,
            seed=args.seed,
            keys=keys,
        )

    eval_stats = evaluate_results(
        results,
        length_buckets=length_buckets,
        summary_output_path=summary_output_path,
        span_length_exact_output_path=span_length_exact_output_path,
        nv_recall_threshold=args.nv_recall_threshold,
        n_token_span_ratio=args.n_token_span_ratio,
    )

    if k_values:
        # k-eidetic evaluation requires exact index-level counting of distinct
        # examples containing each generated string.
        eidetic_enc = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            add_bos_token=False,
            add_eos_token=False,
        )
        eidetic_engine = PageTableBoundedEngine(
            index_dir, eidetic_enc.eos_token_id, args.max_page_table_gb * 1e9
        )
        eidetic_stats = evaluate_k_eidetic_memorization(
            strings,
            eidetic_engine,
            eidetic_enc,
            k_values=k_values,
            min_generation_chars=args.k_eidetic_min_generation_chars,
            min_generation_tokens=args.k_eidetic_min_generation_tokens,
        )
        eval_stats.update(eidetic_stats)
        _move_summary_doc_id_fields_to_end(eval_stats)
        with open(summary_output_path, "w") as f:
            json.dump(eval_stats, f, indent=4)

    _move_summary_doc_id_fields_to_end(eval_stats)

    print("\n" + "=" * 40 + " EVALUATION SUMMARY " + "=" * 40)
    for k, v in eval_stats.items():
        print(f"{k}: {v}")

    save_results(results, results_output_path, record_fields)
        

if __name__ == "__main__":
    main()

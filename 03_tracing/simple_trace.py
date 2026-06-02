"""
python 03_tracing/simple_trace.py \
    --dataset dummy \
    --index-dir 00_data/dummy_index \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dummy.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --results-output simpletrace_results_dummy.jsonl \
    --summary-output simpletrace_evaluation_summary_dummy.json \
    --length-buckets 1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf \
"""

import ast
import argparse
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
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, CancelledError
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
DEFAULT_METRIC_DECIMALS = 9
_metric_decimals = DEFAULT_METRIC_DECIMALS

# Lock that serializes print() calls across threads so output lines
# from different generations never interleave in the terminal.
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe drop-in replacement for print()."""
    with _print_lock:
        print(*args, **kwargs)


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


def _flatten_ranks(find_res: dict) -> list[tuple[int, int]]:
    """Flatten a find() result into (shard, rank) pairs."""
    return [
        (s, r)
        for s, (rank_start, rank_end) in enumerate(find_res["segment_by_shard"])
        for r in range(rank_start, rank_end)
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
) -> list[dict]:
    """Fetch document records for rank hits and annotate their match tier."""
    if limit < 1 or not ranks:
        return []

    if len(ranks) > limit:
        if deterministic:
            selected_ranks = _sample_evenly(ranks, limit)
        else:
            selected_ranks = random.sample(ranks, limit)
    else:
        selected_ranks = list(ranks)

    normalized_generation = (
        _normalize_mixed_text(generation)
        if match_mode == MIXED_MATCH_MODE else None
    )
    docs = []
    seen_examples: set[tuple[int, int]] = set()

    for s, r in selected_ranks:
        raw_doc = engine.get_doc_by_rank(s=s, rank=r, max_disp_len=max_doc_toks)
        doc_ix = raw_doc.get("doc_ix")
        example_key = (s, int(doc_ix)) if doc_ix is not None else (s, int(r))
        if example_key in seen_examples:
            continue
        seen_examples.add(example_key)

        doc_meta = ast.literal_eval(raw_doc["metadata"])["metadata"]
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

    punc_chars = "!.?\n"
    if any(ch in punc_chars for ch in span_text[:-1]):
        return False
    first_tok = enc.convert_ids_to_tokens(span_ids[0])
    if first_tok[0] != '▁':
        return False
    if end < total_tokens and enc.convert_ids_to_tokens(gen_ids[end])[0] != '▁':
        return False
    return True


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
) -> dict:
    """
    Run the full SimpleTrace pipeline for a single generation string.

    Returns a dict with keys:
        "generation"  - original text
        "gen_ids"     - token ids
        "final_spans" - list of traced segment dicts (start, end, text, docs)
    """
    gen_ids = enc.encode(generation)
    L = len(gen_ids)
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
    #   - cnt == 0: no exact match → the engine returns the rank of the
    #     nearest neighbour in the sorted suffix array. We fetch that
    #     neighbour document and walk both sequences token-by-token with
    #     compute_longest_prefix() to find the longest prefix of the
    #     suffix that does appear somewhere in that document.
    # Each suffix produces a candidate span (start, start + matched_toks).
    # ------------------------------------------------------------------
    spans = []
    for start in range(L - 1):
        # Cooperatively check for stop request between index queries.
        if stop_event is not None and stop_event.is_set():
            print(f"[INFO] Stop requested - aborting '{generation[:40]}...'", flush=True)
            return {
                "generation": generation,
                "gen_ids": gen_ids,
                "match_mode": match_mode,
                "final_spans": [],
            }

        _suffix = gen_ids[start:]
        _res = engine.find(input_ids=_suffix)

        if _res['cnt'] == 0:
            # No verbatim hit: fall back to the nearest-neighbour document
            # returned by the index to find the longest matching prefix.
            # Each shard contributes one nearest-neighbour rank; we check
            # all of them and keep the longest match found across shards.
            _shards = _res['segment_by_shard']
            matched_toks = 0
            for s, (rank_start, _) in enumerate(_shards):
                _doc_ids = engine.get_doc_by_rank(
                    s=s,
                    rank=rank_start,
                    max_disp_len=max_doc_toks,
                )['token_ids']
                matched_toks = max(matched_toks, compute_longest_prefix(_suffix, _doc_ids))
        else:
            # Verbatim hit: the entire suffix is a match.
            matched_toks = len(_suffix)
        spans.append((start, start + matched_toks))

    # Filter pass 1 - retain only "clean", self-contained spans:
    #   a) No sentence-ending punctuation (! . ? newline) in the interior
    #      of the span, which would indicate the span crosses a sentence
    #      boundary and is likely an accidental match.
    #   b) The first token must be word-initial (Llama-2 BPE tokens that
    #      start a word carry the '▁' prefix), so spans start at a word
    #      boundary rather than mid-word.
    #   c) The token immediately after the span (if any) must also be
    #      word-initial, so the span ends at a clean word boundary.
    full_spans = []
    for start, end in spans:
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
    # tokens of the enclosing document. The metadata is stored as a Python
    # literal string in the index and parsed with ast.literal_eval().
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
            )
        )

    # ------------------------------------------------------------------
    # Step 4: Merge overlapping spans into final traced segments
    #
    # Adjacent or overlapping filtered spans are collapsed into single,
    # wider segments so the final output presents coherent, non-redundant
    # text regions rather than a pile of potentially overlapping snippets.
    #
    # Spans are already sorted by start position. We iterate sequentially
    # and start a new group whenever the next span begins at or after the
    # current group's end; otherwise we extend the current group.
    #
    # The document budget docs_per_span is divided evenly among the
    # constituent spans of each group (ceil division) so the total doc
    # count per merged segment stays close to docs_per_span regardless
    # of how many spans were merged.
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

    merged_groups = [[0]]
    curr_end = filt_spans[0][1]
    for i, (start, end, *_) in enumerate(filt_spans[1:], start=1):
        if start < curr_end:
            # Overlapping or adjacent: extend the current group.
            curr_end = max(curr_end, end)
            merged_groups[-1].append(i)
        else:
            # Non-overlapping: start a new group.
            curr_end = end
            merged_groups.append([i])

    final_spans = []
    for group in merged_groups:
        # Divide the doc budget evenly across the spans being merged.
        docs_budget = math.ceil(docs_per_span / len(group))
        all_docs = []
        for i in group:
            all_docs.extend(span_to_docs[i][:docs_budget])
        group_spans = [filt_spans[i] for i in group]
        seg_start = min(s[0] for s in group_spans)
        seg_end = max(s[1] for s in group_spans)
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
    results : dict[str, dict]
        Output of launch_simpletrace - maps generation text to its trace dict
        (keys: "generation", "gen_ids", "final_spans").
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

    for generation, result in results.items():
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

def save_results(results: dict[str, dict], output_path: str) -> None:
    """Save tracing results to a JSONL file, one line per generation.

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
        for generation, result in results.items():
            sorted_spans = sorted(
                result["final_spans"],
                key=lambda sp: (sp["end"] - sp["start"]),
                reverse=True,
            )
            record = {
                "generation": generation,
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

def _worker_init(index_dir: str, unigram_probs_path: str, metric_decimals: int):
    """Called once in each worker process before any tasks are dispatched."""
    global _worker_engine, _worker_enc, _worker_unigram_probs
    _set_metric_decimals(metric_decimals)
    _worker_enc = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False
    )
    _worker_engine = InfiniGramEngine(
        index_dir=index_dir,
        eos_token_id=_worker_enc.eos_token_id,
        precompute_unigram_logprobs=False,
    )
    with open(unigram_probs_path) as f:
        _worker_unigram_probs = {int(k): v['prob'] for k, v in json.load(f).items()}


def _worker_trace(generation: str, docs_per_span: int, match_mode: str) -> dict:
    """Thin wrapper that calls trace_generation using worker-local globals.    stop_event is not passed - worker processes receive SIGINT directly from
    the OS when Ctrl+C is pressed, which raises KeyboardInterrupt naturally.
    """
    return trace_generation(
        generation, _worker_engine, _worker_enc, _worker_unigram_probs,
        docs_per_span=docs_per_span, match_mode=match_mode, stop_event=None,
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
) -> dict[str, dict]:
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

    signal.signal(signal.SIGINT, _sigint_handler)

    results: dict[str, dict] = {}

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(index_dir, unigram_probs_path, metric_decimals),
    )

    futures = {
        executor.submit(_worker_trace, gen, docs_per_span, match_mode): gen
        for gen in generations
    }
    pending = set(futures)
    try:
        with tqdm(total=len(generations), desc="Tracing", unit="gen") as pbar:
            while pending and not _stop_event.is_set():
                # Short timeout ensures the loop wakes up frequently to check
                # _stop_event, making Ctrl+C feel near-instant.
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    gen = futures[future]
                    try:
                        results[gen] = future.result()
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
        executor.shutdown(wait=False, cancel_futures=True)

    end_time = time.time()
    print(f"\nElapsed Time: {end_time - start_time_local:.2f} seconds  ({len(generations)} generation(s), {num_workers} workers)")

    # Print in original submission order
    if enable_print:
        for gen in generations:
            if gen in results:
                print_results(results[gen])
    
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
        "--index-dir",
        help="Path to the InfiniGram index directory.",
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
        "--docs-per-span",
        type=int,
        default=10,
        help="Maximum number of retrieved documents per span.",
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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
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

    strings = load_generations(
        args.dataset,
        args.limit,
        is_jsonl=args.is_jsonl,
        text_field=args.text_field,
        is_generation_json=args.is_generation_json,
        generation_text_field=args.generation_text_field,
    )
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

    results = launch_simpletrace(
        index_dir=args.index_dir,
        generations=strings,
        unigram_probs_path=args.unigram_probs_path,
        num_workers=args.num_workers,
        docs_per_span=args.docs_per_span,
        match_mode=args.match_mode,
        metric_decimals=args.metric_decimals,
        enable_print=args.enable_print,
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
        eidetic_engine = InfiniGramEngine(
            index_dir=args.index_dir,
            eos_token_id=eidetic_enc.eos_token_id,
            precompute_unigram_logprobs=False,
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

    save_results(results, results_output_path)
        

if __name__ == "__main__":
    main()

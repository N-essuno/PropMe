"""
OLMoTrace reproduction, based only on:
  Liu et al. (2025), "OLMoTrace: Tracing Language Model Outputs Back to
  Trillions of Training Tokens", arXiv:2504.07096.

python olmotrace/olmo_trace.py \
    --dataset dummy \
    --index-dir 00_prepare_data/dummy_index \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dummy.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --results-output olmotrace_results_dummy.jsonl \
    --summary-output olmotrace_summary_dummy.json \
    --length-buckets 1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf

Pipeline (paper §3, Figure 2):
  1. Find maximal matching spans (§3.1, Algorithm 1).
  2. Keep the K = ceil(0.05 * L) spans with the smallest span unigram probability.
  3. Retrieve up to 10 enclosing document snippets per kept span.
  4. Merge overlapping spans; merge snippets from the same document.
  5. Rerank documents by BM25 (query = user prompt + LM response) and bucket
     relevance (normalized by 0.18 * #chars of the LM output; >=0.7 high,
     [0.5, 0.7) medium, <0.5 low). A span's relevance is the max over the
     documents enclosing it.
"""

import argparse
import ast
import json
import math
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from tqdm import tqdm

if __package__:
    from .bm25 import BM25_BACKEND, BM25Okapi
else:
    from bm25 import BM25_BACKEND, BM25Okapi

# Named dataset loaders are shared with SimpleTrace so both tools accept the
# same --dataset values.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "03_tracing"))
from data_loading import load_dummy_dataset, load_generic_dataset  # noqa: E402

TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"  # paper §3, step 1

# Hyperparameters of the paper's final setting.
SPAN_FRACTION = 0.05              # step 2: K = ceil(0.05 * L)
DOCS_PER_SPAN = 10                # step 3: up to 10 snippets per span
SNIPPET_CONTEXT_TOKENS = 80       # §2: snippet of 80 tokens around the span
DOCUMENT_CONTEXT_TOKENS = 500     # §2 / App. C: extended context, also used by BM25
BM25_NORMALIZATION_COEF = 0.18    # §4: max BM25 ~= 0.18 * #chars of LM output
HIGH_RELEVANCE_THRESHOLD = 0.7    # §4
MEDIUM_RELEVANCE_THRESHOLD = 0.5  # §4
RELEVANCE_LEVELS = ("high", "medium", "low")

# Byte-fallback pieces look like "<0x0A>"; bytes 0x80-0xBF continue a UTF-8
# character and therefore never begin a word.
_BYTE_PIECE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_DELIMITER_PIECES = {".", "▁.", "<0x0A>"}  # period and newline tokens (step 1)
_WHITESPACE_BYTE_PIECES = {"<0x09>", "<0x0A>", "<0x0D>"}
_BM25_WORD_RE = re.compile(r"\w+")


@dataclass
class TraceConfig:
    docs_per_span: int = DOCS_PER_SPAN
    span_fraction: float = SPAN_FRACTION
    snippet_context_tokens: int = SNIPPET_CONTEXT_TOKENS
    document_context_tokens: int = DOCUMENT_CONTEXT_TOKENS
    seed: int = 0


# ---------------------------------------------------------------------------
# Tokenizer-derived token classes
# ---------------------------------------------------------------------------

def is_begin_of_word_piece(piece: str) -> bool:
    """Whether a Llama-2 SentencePiece token begins a word on its own.

    The paper requires spans to start at a begin-of-word token and to be
    followed by one (or by the end of the output) so that spans "do not begin
    or end with incomplete words", but does not define the token set. A piece
    begins a word unless it continues the previous one: word pieces without the
    "▁" prefix (e.g. "ing", digits after "▁") and UTF-8 continuation bytes are
    continuations; "▁"-prefixed pieces, punctuation and other byte tokens
    (e.g. newline) are not.
    """
    if piece.startswith("▁"):
        return True
    byte_match = _BYTE_PIECE_RE.match(piece)
    if byte_match:
        return not (0x80 <= int(byte_match.group(1), 16) <= 0xBF)
    return not piece[:1].isalnum()


class TokenInfo:
    """Tokenizer plus begin-of-word and delimiter lookups."""

    def __init__(self, enc):
        self.enc = enc
        pieces = enc.convert_ids_to_tokens(list(range(len(enc))))
        self.is_bow = [is_begin_of_word_piece(p) for p in pieces]
        self.delimiter_ids = {i for i, p in enumerate(pieces) if p in _DELIMITER_PIECES}
        self.whitespace_byte_ids = {i for i, p in enumerate(pieces) if p in _WHITESPACE_BYTE_PIECES}

    def begin_of_word_flags(self, ids: list[int]) -> list[bool]:
        """Per-position begin-of-word flags for a token sequence.

        Llama-2 drops the "▁" prefix of a word that follows a newline or tab,
        so such a word is still treated as beginning a new word.
        """
        return [
            self.is_bow[t] or (i > 0 and ids[i - 1] in self.whitespace_byte_ids)
            for i, t in enumerate(ids)
        ]


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, add_bos_token=False, add_eos_token=False)


def load_engine(index_dir, eos_token_id: int) -> InfiniGramEngine:
    # App. B: prefetching is turned off (all prefetch depths set to 0).
    return InfiniGramEngine(
        index_dir=index_dir,
        eos_token_id=eos_token_id,
        ds_prefetch_depth=0,
        sa_prefetch_depth=0,
        od_prefetch_depth=0,
        precompute_unigram_logprobs=False,
    )


def load_unigram_logprobs(path: str) -> dict[int, float]:
    """Load the repository's precomputed unigram table (02_unigram_probs/*.json)."""
    with open(path) as f:
        raw = json.load(f)
    logprobs = {}
    for token_id, entry in raw.items():
        if "log_prob" in entry:
            logprobs[int(token_id)] = float(entry["log_prob"])
        else:
            prob = float(entry["prob"])
            logprobs[int(token_id)] = math.log(prob) if prob > 0 else float("-inf")
    return logprobs


# ---------------------------------------------------------------------------
# Step 1: maximal matching spans (§3.1, Algorithm 1)
# ---------------------------------------------------------------------------

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def get_longest_prefix_len(engine: InfiniGramEngine, s: list[int], tok_cnt_by_shard: list[int]) -> int:
    """Length of the longest prefix of `s` found in the corpus, with one FIND.

    If FIND returns an empty segment [l, l), the suffixes at SA ranks l-1 and l
    lexicographically precede/succeed `s`, so one of them shares the longest
    common prefix with it. With several shards, take the max over shards.
    """
    res = engine.find(input_ids=s)
    if "error" in res:
        raise RuntimeError(res["error"])
    if res["cnt"] > 0:
        return len(s)

    requests = []
    for shard, (l, _) in enumerate(res["segment_by_shard"]):
        for rank in (l - 1, l):
            if 0 <= rank < tok_cnt_by_shard[shard]:
                requests.append((shard, rank, len(s), 0))
    if not requests:
        return 0
    docs = engine.get_docs_by_ranks_2(requests)
    if isinstance(docs, dict):
        raise RuntimeError(docs["error"])

    best = 0
    for doc in docs:
        offset = doc["needle_offset"]
        # Ranks pointing at a document separator have no needle in the doc.
        if offset > len(doc["token_ids"]):
            continue
        best = max(best, _common_prefix_len(s, doc["token_ids"][offset:]))
    return best


def trim_to_self_contained(length: int, next_is_bow: list[bool], first_delimiter: int | None) -> int:
    """Shrink a matching prefix of a suffix until it satisfies the step-1 criteria.

    Equivalent to Algorithm 1's loop: while s[:len-1] contains a delimiter
    token or s[len] (the next token) is not begin-of-word, decrement len.
    `next_is_bow[i]` is the begin-of-word flag of suffix position i, and
    `first_delimiter` the index of the first delimiter token in the suffix.
    """
    if first_delimiter is not None and first_delimiter < length - 1:
        length = first_delimiter + 1
    while 0 < length < len(next_is_bow) and not next_is_bow[length]:
        length -= 1
    return length


def suppress_nonmaximal_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep spans whose end exceeds the end of every span starting before them."""
    kept = []
    max_end = 0
    for b, e in sorted(spans):
        if max_end < e:
            max_end = e
            kept.append((b, e))
    return kept


def get_maximal_matching_spans(
    engine: InfiniGramEngine,
    ids: list[int],
    token_info: TokenInfo,
    tok_cnt_by_shard: list[int],
    executor: ThreadPoolExecutor | None = None,
) -> list[tuple[int, int]]:
    L = len(ids)
    next_delimiter = [None] * (L + 1)
    for i in range(L - 1, -1, -1):
        next_delimiter[i] = i if ids[i] in token_info.delimiter_ids else next_delimiter[i + 1]

    is_bow = token_info.begin_of_word_flags(ids)
    starts = [b for b in range(L) if is_bow[b]]

    def span_from(b: int) -> tuple[int, int]:
        length = get_longest_prefix_len(engine, ids[b:], tok_cnt_by_shard)
        first_delim = None if next_delimiter[b] is None else next_delimiter[b] - b
        return b, b + trim_to_self_contained(length, is_bow[b:], first_delim)

    # The suffix queries are independent; the paper executes them in parallel.
    mapper = executor.map if executor is not None else map
    spans = [(b, e) for b, e in mapper(span_from, starts) if e > b]
    return suppress_nonmaximal_spans(spans)


# ---------------------------------------------------------------------------
# Step 2: keep long and unique spans
# ---------------------------------------------------------------------------

def num_spans_to_keep(L: int, span_fraction: float) -> int:
    # Exact rational arithmetic: 0.05 * 60 is 3.0000000000000004 in floats.
    return math.ceil(Fraction(str(span_fraction)) * L)


def span_unigram_logprob(span_ids: list[int], unigram_logprobs: dict[int, float]) -> float:
    """log of the product of token unigram probabilities (missing tokens -> prob 1)."""
    return sum(unigram_logprobs.get(t, 0.0) for t in span_ids)


def filter_spans(
    spans: list[tuple[int, int]],
    ids: list[int],
    unigram_logprobs: dict[int, float],
    span_fraction: float,
) -> list[tuple[int, int, float]]:
    K = num_spans_to_keep(len(ids), span_fraction)
    scored = [(b, e, span_unigram_logprob(ids[b:e], unigram_logprobs)) for b, e in spans]
    kept = sorted(scored, key=lambda x: (x[2], x[0]))[:K]
    return sorted(kept)


# ---------------------------------------------------------------------------
# Step 3: retrieve enclosing documents
# ---------------------------------------------------------------------------

def _parse_metadata(raw_metadata) -> dict:
    """Parse InfiniGram metadata stored as JSON or a Python literal."""
    if isinstance(raw_metadata, dict):
        parsed = raw_metadata
    else:
        try:
            parsed = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(raw_metadata)
            except (ValueError, SyntaxError):
                return {"raw": raw_metadata}
    if not isinstance(parsed, dict):
        return {"raw": parsed}
    inner = parsed.get("metadata", parsed)
    return inner if isinstance(inner, dict) else parsed


def _doc_id(metadata: dict, doc_ix: int):
    for key in ("id", "doc_id", "source", "url"):
        if metadata.get(key):
            return metadata[key]
    return doc_ix


def sample_occurrences(segment_by_shard, k: int, rng: random.Random) -> list[tuple[int, int]]:
    """Return (shard, rank) pairs: all occurrences, or k sampled at random."""
    sizes = [r - l for l, r in segment_by_shard]
    total = sum(sizes)
    picks = range(total) if total <= k else sorted(rng.sample(range(total), k))
    occurrences = []
    shard, base = 0, 0
    for i in picks:
        while i >= base + sizes[shard]:
            base += sizes[shard]
            shard += 1
        occurrences.append((shard, segment_by_shard[shard][0] + (i - base)))
    return occurrences


def retrieve_snippets(
    engine: InfiniGramEngine,
    span_ids: list[int],
    token_info: TokenInfo,
    config: TraceConfig,
    rng: random.Random,
) -> tuple[int, list[dict]]:
    # A second FIND locates all occurrences of the span (§3.1).
    res = engine.find(input_ids=span_ids)
    if "error" in res:
        raise RuntimeError(res["error"])
    occurrences = sample_occurrences(res["segment_by_shard"], config.docs_per_span, rng)
    if not occurrences:
        return res["cnt"], []

    docs = engine.get_docs_by_ranks_2(
        [(s, r, len(span_ids), config.document_context_tokens) for s, r in occurrences]
    )
    if isinstance(docs, dict):
        raise RuntimeError(docs["error"])

    snippets = []
    for (shard, rank), doc in zip(occurrences, docs):
        if doc.get("blocked"):
            continue
        toks = doc["token_ids"]
        offset = doc["needle_offset"]
        lo = max(0, offset - config.snippet_context_tokens)
        hi = offset + len(span_ids) + config.snippet_context_tokens
        metadata = _parse_metadata(doc["metadata"])
        snippets.append({
            "shard": shard,
            "rank": rank,
            "doc_ix": doc["doc_ix"],
            "doc_len": doc["doc_len"],
            "id": _doc_id(metadata, doc["doc_ix"]),
            "metadata": metadata,
            "snippet": token_info.enc.decode(toks[lo:hi]),
            "context": token_info.enc.decode(toks),
        })
    return res["cnt"], snippets


# ---------------------------------------------------------------------------
# Step 4: merge spans, merge documents
# ---------------------------------------------------------------------------

def merge_overlapping_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int, list[int]]]:
    """Union overlapping spans; returns (start, end, member indices) in order."""
    merged: list[tuple[int, int, list[int]]] = []
    for i, (b, e) in sorted(enumerate(spans), key=lambda x: x[1]):
        if merged and b < merged[-1][1]:
            start, end, members = merged[-1]
            merged[-1] = (start, max(end, e), members + [i])
        else:
            merged.append((b, e, [i]))
    return merged


def merge_documents(snippets_by_span: list[list[dict]]) -> list[dict]:
    """Group snippets retrieved from the same training document."""
    documents: dict[tuple[int, int], dict] = {}
    for span_idx, snippets in enumerate(snippets_by_span):
        for snip in snippets:
            key = (snip["shard"], snip["doc_ix"])
            doc = documents.get(key)
            if doc is None:
                doc = documents[key] = {
                    "shard": snip["shard"],
                    "doc_ix": snip["doc_ix"],
                    "doc_len": snip["doc_len"],
                    "id": snip["id"],
                    "metadata": snip["metadata"],
                    "filtered_span_indices": [],
                    "snippets": [],
                }
            if span_idx not in doc["filtered_span_indices"]:
                doc["filtered_span_indices"].append(span_idx)
            doc["snippets"].append({
                "filtered_span_index": span_idx,
                "snippet": snip["snippet"],
                "context": snip["context"],
            })
    return list(documents.values())


# ---------------------------------------------------------------------------
# Step 5: rerank and color documents by relevance
# ---------------------------------------------------------------------------

def bm25_tokenize(text: str) -> list[str]:
    return _BM25_WORD_RE.findall(text.lower())


def relevance_level(normalized_score: float) -> str:
    if normalized_score >= HIGH_RELEVANCE_THRESHOLD:
        return "high"
    if normalized_score >= MEDIUM_RELEVANCE_THRESHOLD:
        return "medium"
    return "low"


def score_documents(documents: list[dict], prompt: str | None, response: str) -> None:
    """Attach BM25 scores (retrieved docs as corpus, prompt + response as query)."""
    if not documents:
        return
    corpus = []
    for doc in documents:
        contexts = list(dict.fromkeys(s["context"] for s in doc["snippets"]))
        corpus.append(bm25_tokenize("\n".join(contexts)))
    query = bm25_tokenize(f"{prompt}\n{response}" if prompt else response)

    if any(corpus):
        scores = [float(x) for x in BM25Okapi(corpus).get_scores(query)]
    else:
        scores = [0.0] * len(documents)

    norm = BM25_NORMALIZATION_COEF * len(response)
    for doc, score in zip(documents, scores):
        doc["bm25_score"] = score
        doc["relevance_score"] = score / norm if norm > 0 else 0.0
        doc["relevance"] = relevance_level(doc["relevance_score"])


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def trace_generation(
    response: str,
    engine: InfiniGramEngine,
    token_info: TokenInfo,
    unigram_logprobs: dict[int, float],
    config: TraceConfig,
    *,
    prompt: str | None = None,
    index: int = 0,
    executor: ThreadPoolExecutor | None = None,
) -> dict:
    """Run the five OLMoTrace steps on one LM response."""
    enc = token_info.enc
    rng = random.Random(f"{config.seed}:{index}")
    num_shards = engine.engine.get_num_shards()
    tok_cnt_by_shard = [engine.engine.get_tok_cnt(s=s) for s in range(num_shards)]
    t0 = time.perf_counter()

    ids = enc.encode(response)
    maximal = get_maximal_matching_spans(engine, ids, token_info, tok_cnt_by_shard, executor)
    t1 = time.perf_counter()

    kept = filter_spans(maximal, ids, unigram_logprobs, config.span_fraction)
    t2 = time.perf_counter()

    filtered_spans = []
    snippets_by_span = []
    for b, e, logprob in kept:
        count, snippets = retrieve_snippets(engine, ids[b:e], token_info, config, rng)
        snippets_by_span.append(snippets)
        filtered_spans.append({
            "start": b,
            "end": e,
            "span_length": e - b,
            "text": enc.decode(ids[b:e]),
            "unigram_logprob": logprob,
            "count": count,
        })
    t3 = time.perf_counter()

    merged = merge_overlapping_spans([(b, e) for b, e, _ in kept])
    documents = merge_documents(snippets_by_span)
    score_documents(documents, prompt, response)
    documents.sort(key=lambda d: d["bm25_score"], reverse=True)

    filtered_to_merged = {}
    for m_idx, (_, _, members) in enumerate(merged):
        for f_idx in members:
            filtered_to_merged[f_idx] = m_idx
    for rank, doc in enumerate(documents):
        doc["rank"] = rank
        doc["span_indices"] = sorted({filtered_to_merged[i] for i in doc["filtered_span_indices"]})

    spans = []
    for m_idx, (b, e, members) in enumerate(merged):
        doc_ranks = [d["rank"] for d in documents if m_idx in d["span_indices"]]
        best = max((documents[r]["relevance_score"] for r in doc_ranks), default=None)
        spans.append({
            "start": b,
            "end": e,
            "span_length": e - b,
            "text": enc.decode(ids[b:e]),
            "relevance_score": best,
            "relevance": None if best is None else relevance_level(best),
            "filtered_span_indices": members,
            "doc_ranks": doc_ranks,
        })
    t4 = time.perf_counter()

    return {
        "index": index,
        "prompt": prompt,
        "generation": response,
        "num_tokens": len(ids),
        "num_chars": len(response),
        "num_maximal_spans": len(maximal),
        "filtered_spans": filtered_spans,
        "spans": spans,
        "documents": documents,
        "latency_seconds": {
            "step1_maximal_spans": t1 - t0,
            "step2_filter": t2 - t1,
            "step3_retrieve": t3 - t2,
            "step4_5_merge_rerank": t4 - t3,
            "steps_1_3": t3 - t0,
            "total": t4 - t0,
        },
    }


def print_results(result: dict) -> None:
    """Text rendering of the paper's UI: highlighted spans and document panel."""
    marks = {"high": "H", "medium": "M", "low": "L", None: "-"}
    text = result["generation"]
    print("\n" + "#" * 80)
    if result["prompt"]:
        print(f"Prompt: {result['prompt']}")
    print(f"Response: {text}")
    for i, sp in enumerate(result["spans"]):
        print(f"  [span {i}] ({marks[sp['relevance']]}) tokens {sp['start']}-{sp['end']}: "
              f"{sp['text']!r} -> docs {sp['doc_ranks']}")
    for doc in result["documents"]:
        print("-" * 10 + f" Document {doc['rank'] + 1} / {len(result['documents'])} "
              f"[{doc['relevance']}, bm25={doc['bm25_score']:.3f}] id={doc['id']} "
              f"spans={doc['span_indices']}")
        for snip in doc["snippets"]:
            print("    ... " + snip["snippet"].replace("\n", " ") + " ...")


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _bucket_label(lo: int, hi: float) -> str:
    return f"({lo}, {'inf' if hi == float('inf') else hi})"


def evaluate_results(
    results: list[dict],
    length_buckets: list[tuple[int, float]],
    config: TraceConfig,
    summary_output_path: str,
) -> dict:
    """Aggregate the statistics the paper reports in §3.2 and §4."""
    filtered_lengths = [sp["span_length"] for r in results for sp in r["filtered_spans"]]
    merged_lengths = [sp["span_length"] for r in results for sp in r["spans"]]
    span_counts = [sp["count"] for r in results for sp in r["filtered_spans"]]
    docs = [d for r in results for d in r["documents"]]
    spans = [sp for r in results for sp in r["spans"]]

    bucket_counts = {_bucket_label(lo, hi): 0 for lo, hi in length_buckets}
    for n in filtered_lengths:
        for lo, hi in length_buckets:
            if lo <= n <= hi:
                bucket_counts[_bucket_label(lo, hi)] += 1
                break
    exact_counts = {}
    for n in sorted(filtered_lengths):
        exact_counts[str(n)] = exact_counts.get(str(n), 0) + 1

    doc_relevance = {lvl: sum(d["relevance"] == lvl for d in docs) for lvl in RELEVANCE_LEVELS}
    span_relevance = {lvl: sum(sp["relevance"] == lvl for sp in spans) for lvl in RELEVANCE_LEVELS}
    max_bm25_per_char = [
        max(d["bm25_score"] for d in r["documents"]) / r["num_chars"]
        for r in results if r["documents"] and r["num_chars"]
    ]
    latencies = [r["latency_seconds"]["total"] for r in results]
    latencies_1_3 = [r["latency_seconds"]["steps_1_3"] for r in results]
    unique_docs = {(d["shard"], d["doc_ix"]) for d in docs}

    summary = {
        "total_generations": len(results),
        "generations_with_spans": sum(bool(r["spans"]) for r in results),
        "generations_with_prompt": sum(bool(r["prompt"]) for r in results),
        "avg_response_tokens": _ratio(sum(r["num_tokens"] for r in results), len(results)),
        "total_maximal_spans": sum(r["num_maximal_spans"] for r in results),
        "total_filtered_spans": len(filtered_lengths),
        "total_spans": len(merged_lengths),
        # §4 "Length of spans": after step 2, before merging.
        "filtered_span_length_mean": statistics.mean(filtered_lengths) if filtered_lengths else 0.0,
        "filtered_span_length_median": statistics.median(filtered_lengths) if filtered_lengths else 0.0,
        "min_span_length": min(filtered_lengths, default=0),
        "max_span_length": max(filtered_lengths, default=0),
        "merged_span_length_mean": statistics.mean(merged_lengths) if merged_lengths else 0.0,
        "max_merged_span_length": max(merged_lengths, default=0),
        "spans_length_counts_distribution": bucket_counts,
        "spans_length_distribution": {k: _ratio(v, len(filtered_lengths)) for k, v in bucket_counts.items()},
        "spans_length_counts_exact": exact_counts,
        # §3 step 3: "most spans appear no more than 10 times".
        "filtered_spans_count_le_docs_per_span_ratio": _ratio(
            sum(c <= config.docs_per_span for c in span_counts), len(span_counts)
        ),
        "total_docs": len(docs),
        "unique_total_docs": len(unique_docs),
        "avg_docs_per_generation": _ratio(len(docs), len(results)),
        # §4 "Relevance score of documents".
        "doc_relevance_counts": doc_relevance,
        "doc_relevance_distribution": {k: _ratio(v, len(docs)) for k, v in doc_relevance.items()},
        "span_relevance_counts": span_relevance,
        "span_relevance_distribution": {k: _ratio(v, len(spans)) for k, v in span_relevance.items()},
        "avg_doc_relevance_score": _ratio(sum(d["relevance_score"] for d in docs), len(docs)),
        "avg_max_bm25_per_response_char": _ratio(sum(max_bm25_per_char), len(max_bm25_per_char)),
        "max_max_bm25_per_response_char": max(max_bm25_per_char, default=0.0),
        # §3.2 latency (steps 1-3) and end-to-end latency per generation.
        "avg_latency_steps_1_3_seconds": _ratio(sum(latencies_1_3), len(results)),
        "avg_latency_seconds": _ratio(sum(latencies), len(results)),
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "config": {**asdict(config), "bm25_backend": BM25_BACKEND,
                   "bm25_normalization_coef": BM25_NORMALIZATION_COEF,
                   "relevance_thresholds": {"high": HIGH_RELEVANCE_THRESHOLD,
                                            "medium": MEDIUM_RELEVANCE_THRESHOLD}},
    }

    os.makedirs(os.path.dirname(summary_output_path) or ".", exist_ok=True)
    with open(summary_output_path, "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def save_results(results: list[dict], output_path: str) -> None:
    """One JSON line per generation, in input order."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

_worker_state: dict = {}


def _worker_init(index_dir, unigram_probs_path: str, config: TraceConfig, find_threads: int) -> None:
    enc = load_tokenizer()
    _worker_state.update(
        engine=load_engine(index_dir, enc.eos_token_id),
        token_info=TokenInfo(enc),
        unigram_logprobs=load_unigram_logprobs(unigram_probs_path),
        config=config,
        executor=ThreadPoolExecutor(find_threads) if find_threads > 1 else None,
    )


def _worker_trace(index: int, response: str, prompt: str | None) -> dict:
    st = _worker_state
    return trace_generation(
        response, st["engine"], st["token_info"], st["unigram_logprobs"], st["config"],
        prompt=prompt, index=index, executor=st["executor"],
    )


def launch_olmotrace(
    index_dir,
    items: list[tuple[str, str | None]],
    unigram_probs_path: str,
    config: TraceConfig,
    num_workers: int = 8,
    find_threads: int = 4,
    enable_print: bool = False,
) -> list[dict]:
    start = time.time()
    results: list[dict | None] = [None] * len(items)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(index_dir, unigram_probs_path, config, find_threads),
    ) as executor:
        futures = {
            executor.submit(_worker_trace, i, response, prompt): i
            for i, (response, prompt) in enumerate(items)
        }
        try:
            with tqdm(total=len(items), desc="Tracing", unit="gen") as pbar:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C received - cancelling pending generations...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print(f"\nElapsed Time: {time.time() - start:.2f} seconds  "
          f"({len(items)} generation(s), {num_workers} workers)")
    if enable_print:
        for result in results:
            print_results(result)
    return results


# ---------------------------------------------------------------------------
# Inputs and CLI
# ---------------------------------------------------------------------------

def load_generation_items(
    dataset_name: str,
    limit: int | None,
    is_jsonl: bool = False,
    text_field: str = "text",
    prompt_field: str = "prompt",
    is_generation_json: bool = False,
    generation_text_field: str = "completion",
    generation_prompt_field: str = "prompt",
) -> list[tuple[str, str | None]]:
    """Load (response, prompt) pairs. Prompts are None when not available."""
    if dataset_name == "dummy":
        items = [(t, None) for t in load_dummy_dataset()]
    elif dataset_name == "generic":
        items = [(t, None) for t in load_generic_dataset()]
    elif is_generation_json:
        with open(dataset_name) as f:
            data = json.load(f)
        items = []

        def _collect(node) -> None:
            if isinstance(node, dict):
                value = node.get(generation_text_field)
                if isinstance(value, str):
                    prompt = node.get(generation_prompt_field) if generation_prompt_field else None
                    items.append((value, prompt if isinstance(prompt, str) else None))
                for child in node.values():
                    _collect(child)
            elif isinstance(node, list):
                for child in node:
                    _collect(child)

        _collect(data)
        if not items:
            raise ValueError(f"No '{generation_text_field}' fields found in generation dataset: {dataset_name}")
    elif is_jsonl:
        items = []
        with open(dataset_name) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if text_field not in row:
                    continue
                prompt = row.get(prompt_field) if prompt_field else None
                items.append((row[text_field], prompt if isinstance(prompt, str) else None))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return items[:limit] if limit is not None else items


def parse_length_buckets(raw: str) -> list[tuple[int, float]]:
    """Parse a bucket string like '1-3,4-6,7-10,11-20,21-inf'."""
    buckets = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Invalid bucket '{item}'. Expected format 'lo-hi'.")
        lo_s, hi_s = item.split("-", 1)
        lo = int(lo_s)
        hi = float("inf") if hi_s.lower() == "inf" else int(hi_s)
        if lo < 1 or hi < lo:
            raise ValueError(f"Invalid bucket '{item}'.")
        buckets.append((lo, hi))
    if not buckets:
        raise ValueError("At least one valid length bucket must be provided.")
    return buckets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OLMoTrace (arXiv:2504.07096) over a generation dataset.")
    parser.add_argument("--dataset", default="dummy",
                        help="Named loader from 03_tracing/data_loading.py ('dummy', 'generic'), a JSONL path "
                             "with --is-jsonl, or a generation JSON path with --is-generation-json.")
    parser.add_argument("--is-jsonl", action="store_true",
                        help="Treat --dataset as a JSONL file with the response in --text-field.")
    parser.add_argument("--is-generation-json", action="store_true",
                        help="Treat --dataset as a generation JSON file (response in --generation-text-field).")
    parser.add_argument("--text-field", default="text", help="Response field in the JSONL file.")
    parser.add_argument("--prompt-field", default="prompt",
                        help="User-prompt field in the JSONL file, used in the BM25 query ('' disables).")
    parser.add_argument("--generation-text-field", default="completion",
                        help="Response field in the generation JSON file.")
    parser.add_argument("--generation-prompt-field", default="prompt",
                        help="User-prompt field in the generation JSON file, used in the BM25 query ('' disables).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of generations to process.")
    parser.add_argument("--index-dir", nargs="+", required=True,
                        help="One or more InfiniGram index directories, loaded as one multi-shard index.")
    parser.add_argument("--unigram-probs-path", default="02_unigram_probs/unigram_probs_dummy.json",
                        help="Path to token unigram probabilities JSON.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes (one generation each).")
    parser.add_argument("--find-threads", type=int, default=4,
                        help="Threads per worker running the step-1 suffix FIND queries in parallel.")
    parser.add_argument("--docs-per-span", type=int, default=DOCS_PER_SPAN,
                        help="Maximum document snippets retrieved per kept span (paper: 10).")
    parser.add_argument("--span-fraction", type=float, default=SPAN_FRACTION,
                        help="Keep K = ceil(fraction * L) spans in step 2 (paper: 0.05).")
    parser.add_argument("--snippet-context-tokens", type=int, default=SNIPPET_CONTEXT_TOKENS,
                        help="Context tokens around the span in the displayed snippet (paper: 80).")
    parser.add_argument("--document-context-tokens", type=int, default=DOCUMENT_CONTEXT_TOKENS,
                        help="Extended context tokens around the span, also scored by BM25 (paper: 500).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling snippets of frequent spans.")
    parser.add_argument("--enable-print", action="store_true", help="Print spans and ranked documents to stdout.")
    parser.add_argument("--results-output", default="olmotrace_results_test.jsonl",
                        help="Output filename for JSONL tracing results.")
    parser.add_argument("--summary-output", default="olmotrace_summary_test.json",
                        help="Output filename for evaluation summary JSON.")
    parser.add_argument("--length-buckets", default="1-3,4-6,7-10,11-20,21-30,31-40,41-50,51-100,101-inf",
                        help="Comma-separated token-length buckets as 'lo-hi' (use 'inf' for open upper bound).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_workers < 1 or args.find_threads < 1:
        raise ValueError("--num-workers and --find-threads must be >= 1")
    if args.docs_per_span < 1:
        raise ValueError("--docs-per-span must be >= 1")
    if not 0 < args.span_fraction <= 1:
        raise ValueError("--span-fraction must be in (0, 1]")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    if args.is_jsonl and args.is_generation_json:
        raise ValueError("--is-jsonl and --is-generation-json are mutually exclusive")

    items = load_generation_items(
        args.dataset,
        args.limit,
        is_jsonl=args.is_jsonl,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        is_generation_json=args.is_generation_json,
        generation_text_field=args.generation_text_field,
        generation_prompt_field=args.generation_prompt_field,
    )
    length_buckets = parse_length_buckets(args.length_buckets)
    config = TraceConfig(
        docs_per_span=args.docs_per_span,
        span_fraction=args.span_fraction,
        snippet_context_tokens=args.snippet_context_tokens,
        document_context_tokens=args.document_context_tokens,
        seed=args.seed,
    )
    index_dir = args.index_dir[0] if len(args.index_dir) == 1 else args.index_dir

    results = launch_olmotrace(
        index_dir,
        items,
        args.unigram_probs_path,
        config,
        num_workers=args.num_workers,
        find_threads=args.find_threads,
        enable_print=args.enable_print,
    )
    summary = evaluate_results(results, length_buckets, config, args.summary_output)

    print("\n" + "=" * 40 + " EVALUATION SUMMARY " + "=" * 40)
    for k, v in summary.items():
        print(f"{k}: {v}")
    save_results(results, args.results_output)


if __name__ == "__main__":
    main()

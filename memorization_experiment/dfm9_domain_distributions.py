#!/usr/bin/env python3
"""Resolve SimpleTrace document IDs and report DFM9 domain distributions.

The SimpleTrace result files use JSONL content even though their suffix is
usually ``.json``.  Each retrieved document retains its dataset ID but not the
full index metadata.  This command resolves only IDs needed by the requested
analyses: it re-encodes a saved document excerpt with the Llama tokenizer,
uses InfiniGram ``find()``, retrieves candidate records with
``get_doc_by_rank()``, and accepts a record only when its metadata ID matches.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from infini_gram.engine import InfiniGramEngine
from tqdm import tqdm
from transformers import AutoTokenizer
import zstandard as zstd


MODEL_ID = "meta-llama/Llama-2-7b-hf"
RISK_CATEGORIES = ("A", "B", "C", "D")
DEFAULT_INDEXES_ROOT = Path("/work/olmotrace/mimir_propme/indexes")
DEFAULT_PREPARED_DATA_ROOT = Path(
    "/work/olmotrace/mimir_propme/dfm9_memorisation_sources_propme"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "memorization_experiment" / "data" / "dfm9" / "domain_analysis"
)
DEFAULT_LENGTH_BUCKETS = "1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf"
DEFAULT_DOMAIN_FIELDS = (
    "source_metadata.domain",
    "source_metadata.source",
    "source_metadata.dataset",
    "source_metadata.subset",
    "domain",
    "source",
    "cohort",
    "risk_category",
)
UNKNOWN_DOMAIN = "__unknown__"
UNRESOLVED_DOMAIN = "__unresolved__"


class AnalysisError(RuntimeError):
    """Raised when an input or index record cannot be analyzed safely."""


@dataclass(frozen=True)
class LengthBucket:
    low: int
    high: int | None

    @property
    def label(self) -> str:
        return f"({self.low}, {'inf' if self.high is None else self.high})"

    def contains(self, value: int) -> bool:
        return value >= self.low and (self.high is None or value <= self.high)


@dataclass
class DocumentRequest:
    doc_id: str
    category: str
    excerpts: list[str] = field(default_factory=list)

    def add_excerpt(self, text: str, *, limit: int = 3) -> None:
        if not isinstance(text, str) or not text or text in self.excerpts:
            return
        self.excerpts.append(text)
        self.excerpts.sort(key=len, reverse=True)
        del self.excerpts[limit:]


@dataclass(frozen=True)
class ResolvedDocument:
    doc_id: str
    category: str
    domain: str
    domain_field: str
    shard: int | None
    doc_ix: int | None
    metadata: dict[str, Any]
    resolution_method: str = "index_exact_id"


@dataclass
class ResultObservations:
    path: Path
    generation_count: int = 0
    span_count: int = 0
    document_occurrence_count: int = 0
    spans_with_documents: int = 0
    spans_with_nv_hit: int = 0
    missing_id_occurrences: int = 0
    span_counts_by_bucket: Counter[str] = field(default_factory=Counter)
    spans_with_documents_by_bucket: Counter[str] = field(default_factory=Counter)
    spans_with_nv_hit_by_bucket: Counter[str] = field(default_factory=Counter)
    documents_by_bucket: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    nv_hit_documents: Counter[str] = field(default_factory=Counter)
    nv_hit_documents_by_bucket: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def merge(self, other: "ResultObservations") -> None:
        self.generation_count += other.generation_count
        self.span_count += other.span_count
        self.document_occurrence_count += other.document_occurrence_count
        self.spans_with_documents += other.spans_with_documents
        self.spans_with_nv_hit += other.spans_with_nv_hit
        self.missing_id_occurrences += other.missing_id_occurrences
        self.span_counts_by_bucket.update(other.span_counts_by_bucket)
        self.spans_with_documents_by_bucket.update(other.spans_with_documents_by_bucket)
        self.spans_with_nv_hit_by_bucket.update(other.spans_with_nv_hit_by_bucket)
        for label, counts in other.documents_by_bucket.items():
            self.documents_by_bucket[label].update(counts)
        self.nv_hit_documents.update(other.nv_hit_documents)
        for label, counts in other.nv_hit_documents_by_bucket.items():
            self.nv_hit_documents_by_bucket[label].update(counts)

    def document_ids(self) -> set[str]:
        ids = set(self.nv_hit_documents)
        for counts in self.documents_by_bucket.values():
            ids.update(counts)
        return ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve indexed DFM9 document IDs from one or more SimpleTrace "
            "result JSON files and plot domain distributions."
        )
    )
    parser.add_argument(
        "--results",
        nargs="+",
        type=Path,
        required=True,
        help="One or more SimpleTrace result JSON/JSONL files.",
    )
    parser.add_argument(
        "--indexes-root",
        type=Path,
        default=DEFAULT_INDEXES_ROOT,
        help="Root containing DFM9 A/B/C/D index directories.",
    )
    parser.add_argument(
        "--index-dir",
        action="append",
        default=[],
        metavar="CATEGORY=PATH",
        help="Override an index path for a category; repeat as needed.",
    )
    parser.add_argument(
        "--prepared-data-root",
        type=Path,
        default=DEFAULT_PREPARED_DATA_ROOT,
        help=(
            "Prepared index-input root used for exact-ID fallback when a saved "
            "SimpleTrace window is too common for bounded index lookup."
        ),
    )
    parser.add_argument(
        "--no-prepared-data-fallback",
        action="store_true",
        help="Disable exact-ID fallback through prepared JSONL.ZST index inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for JSON reports, document maps, caches, and figures.",
    )
    parser.add_argument(
        "--length-buckets",
        default=DEFAULT_LENGTH_BUCKETS,
        help="Inclusive token buckets in lo-hi form; use inf for the last upper bound.",
    )
    parser.add_argument(
        "--nv-recall-threshold",
        type=float,
        default=0.5,
        help="Include document occurrences with nv_recall >= this value.",
    )
    parser.add_argument(
        "--domain-fields",
        nargs="+",
        default=list(DEFAULT_DOMAIN_FIELDS),
        help="Ordered dotted metadata fields used to derive domain labels.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=MODEL_ID,
        help="Tokenizer used to build the indexes.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Resolved-document JSONL cache (default: OUTPUT_DIR/document_metadata_cache.jsonl).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore an existing resolved-document cache.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Keep unresolved IDs under the __unresolved__ domain instead of failing.",
    )
    parser.add_argument(
        "--max-candidate-ranks",
        type=int,
        default=10_000,
        help="Maximum index candidates examined for one saved document excerpt.",
    )
    parser.add_argument(
        "--top-domains",
        type=int,
        default=15,
        help="Maximum named domains in each plot; remaining domains become Other.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write JSON reports without rendering PNG figures.",
    )
    return parser


def parse_length_buckets(spec: str) -> list[LengthBucket]:
    buckets: list[LengthBucket] = []
    for raw in spec.split(","):
        token = raw.strip()
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+|inf)", token, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid length bucket {token!r}; expected lo-hi or lo-inf")
        low = int(match.group(1))
        high = None if match.group(2).lower() == "inf" else int(match.group(2))
        if low < 1 or (high is not None and high < low):
            raise ValueError(f"Invalid length bucket bounds: {token!r}")
        buckets.append(LengthBucket(low, high))
    if not buckets:
        raise ValueError("At least one length bucket is required")
    for previous, current in zip(buckets, buckets[1:]):
        if previous.high is None or current.low <= previous.high:
            raise ValueError("Length buckets must be ordered and non-overlapping")
    return buckets


def bucket_for_length(length: int, buckets: Sequence[LengthBucket]) -> str | None:
    for bucket in buckets:
        if bucket.contains(length):
            return bucket.label
    return None


def parse_index_overrides(
    indexes_root: Path,
    overrides: Sequence[str],
) -> dict[str, Path]:
    paths = {category: indexes_root / category for category in RISK_CATEGORIES}
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid --index-dir {raw!r}; expected CATEGORY=PATH")
        category, raw_path = raw.split("=", 1)
        category = category.strip().upper()
        if category not in RISK_CATEGORIES or not raw_path.strip():
            raise ValueError(f"Invalid --index-dir {raw!r}; expected A/B/C/D=PATH")
        paths[category] = Path(raw_path).expanduser()
    return paths


def infer_category(doc_id: str) -> str:
    category = doc_id.split(":", 1)[0].upper()
    if category not in RISK_CATEGORIES:
        raise AnalysisError(
            f"Cannot infer DFM9 category from document ID {doc_id!r}; "
            "expected an A:/B:/C:/D: prefix"
        )
    return category


def iter_result_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise AnalysisError(f"{path}: expected a JSON array")
            for index, record in enumerate(payload, start=1):
                if not isinstance(record, dict):
                    raise AnalysisError(f"{path}: JSON item {index} is not an object")
                yield record
            return

        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise AnalysisError(f"{path}:{line_number}: expected a JSON object")
            yield record


def collect_observations(
    path: Path,
    buckets: Sequence[LengthBucket],
    nv_threshold: float,
    requests: dict[str, DocumentRequest],
) -> ResultObservations:
    observations = ResultObservations(path=path)
    for record_number, record in enumerate(iter_result_records(path), start=1):
        observations.generation_count += 1
        spans = record.get("spans")
        if spans is None:
            spans = record.get("final_spans", [])
        if not isinstance(spans, list):
            raise AnalysisError(f"{path}: record {record_number} has non-list spans")

        for span_number, span in enumerate(spans, start=1):
            if not isinstance(span, dict):
                raise AnalysisError(
                    f"{path}: record {record_number}, span {span_number} is not an object"
                )
            raw_length = span.get("span_length")
            if raw_length is None:
                raw_length = int(span.get("end", 0)) - int(span.get("start", 0))
            try:
                span_length = int(raw_length)
            except (TypeError, ValueError) as exc:
                raise AnalysisError(
                    f"{path}: record {record_number}, span {span_number} has invalid length"
                ) from exc
            bucket_label = bucket_for_length(span_length, buckets)
            if bucket_label is None:
                raise AnalysisError(
                    f"{path}: span length {span_length} is outside configured buckets"
                )

            observations.span_count += 1
            observations.span_counts_by_bucket[bucket_label] += 1
            docs = span.get("docs", [])
            if not isinstance(docs, list):
                raise AnalysisError(
                    f"{path}: record {record_number}, span {span_number} has non-list docs"
                )
            if docs:
                observations.spans_with_documents += 1
                observations.spans_with_documents_by_bucket[bucket_label] += 1

            span_has_nv_hit = False
            for doc_number, doc in enumerate(docs, start=1):
                if not isinstance(doc, dict):
                    raise AnalysisError(
                        f"{path}: record {record_number}, span {span_number}, "
                        f"doc {doc_number} is not an object"
                    )
                doc_id = str(doc.get("id", "")).strip()
                if not doc_id:
                    observations.missing_id_occurrences += 1
                    raise AnalysisError(
                        f"{path}: record {record_number}, span {span_number}, "
                        f"doc {doc_number} has no dataset ID"
                    )
                category = infer_category(doc_id)
                request = requests.get(doc_id)
                if request is None:
                    request = DocumentRequest(doc_id=doc_id, category=category)
                    requests[doc_id] = request
                elif request.category != category:
                    raise AnalysisError(f"Conflicting categories for document ID {doc_id}")
                request.add_excerpt(doc.get("text", ""))

                observations.document_occurrence_count += 1
                observations.documents_by_bucket[bucket_label][doc_id] += 1
                try:
                    nv_recall = float(doc.get("nv_recall", 0.0))
                except (TypeError, ValueError) as exc:
                    raise AnalysisError(
                        f"{path}: document {doc_id} has invalid nv_recall"
                    ) from exc
                if nv_recall >= nv_threshold:
                    span_has_nv_hit = True
                    observations.nv_hit_documents[doc_id] += 1
                    observations.nv_hit_documents_by_bucket[bucket_label][doc_id] += 1

            if span_has_nv_hit:
                observations.spans_with_nv_hit += 1
                observations.spans_with_nv_hit_by_bucket[bucket_label] += 1
    return observations


def parse_index_metadata(raw_metadata: Any) -> dict[str, Any]:
    parsed = raw_metadata
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_metadata)
            except (ValueError, SyntaxError) as exc:
                raise AnalysisError("Index metadata is neither JSON nor a Python literal") from exc
    if not isinstance(parsed, dict):
        raise AnalysisError("Index metadata did not decode to an object")
    metadata = parsed.get("metadata", parsed)
    if not isinstance(metadata, dict):
        raise AnalysisError("Index metadata wrapper does not contain an object")
    return metadata


def dotted_value(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def metadata_domain(
    metadata: Mapping[str, Any],
    category: str,
    domain_fields: Sequence[str],
) -> tuple[str, str]:
    for field_path in domain_fields:
        value = dotted_value(metadata, field_path)
        if isinstance(value, str) and value.strip():
            return value.strip(), field_path
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), field_path
    return category or UNKNOWN_DOMAIN, "category_fallback"


def load_cache(
    path: Path,
    domain_fields: Sequence[str],
) -> dict[str, ResolvedDocument]:
    resolved: dict[str, ResolvedDocument] = {}
    if not path.exists():
        return resolved
    for record in iter_result_records(path):
        doc_id = str(record.get("id", "")).strip()
        metadata = record.get("metadata")
        category = str(record.get("category", "")).upper()
        if not doc_id or category not in RISK_CATEGORIES or not isinstance(metadata, dict):
            continue
        domain, domain_field = metadata_domain(metadata, category, domain_fields)
        resolved[doc_id] = ResolvedDocument(
            doc_id=doc_id,
            category=category,
            domain=domain,
            domain_field=domain_field,
            shard=record.get("shard") if isinstance(record.get("shard"), int) else None,
            doc_ix=record.get("doc_ix") if isinstance(record.get("doc_ix"), int) else None,
            metadata=metadata,
            resolution_method=str(record.get("resolution_method", "index_exact_id")),
        )
    return resolved


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temp_path, path)


def write_cache(path: Path, resolved: Mapping[str, ResolvedDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(resolved):
            item = resolved[doc_id]
            handle.write(
                json.dumps(
                    {
                        "id": item.doc_id,
                        "category": item.category,
                        "shard": item.shard,
                        "doc_ix": item.doc_ix,
                        "metadata": item.metadata,
                        "resolution_method": item.resolution_method,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temp_path, path)


def flatten_ranks(find_result: Mapping[str, Any]) -> list[tuple[int, int]]:
    segments = find_result.get("segment_by_shard", [])
    return [
        (shard, rank)
        for shard, segment in enumerate(segments)
        for rank in range(int(segment[0]), int(segment[1]))
    ]


def lookup_query_variants(token_ids: Sequence[int]) -> Iterator[list[int]]:
    """Yield full and deterministic interior queries, longest first.

    SimpleTrace saves decoded document windows. Re-encoding a complete window
    can change boundary tokens (especially when the window starts mid-word),
    even though its interior still has the exact indexed token sequence. The
    interior probes recover those documents without weakening the final exact
    metadata-ID check.
    """
    ids = list(token_ids)
    if not ids:
        return
    seen: set[tuple[int, ...]] = set()

    def emit(query: list[int]) -> Iterator[list[int]]:
        key = tuple(query)
        if query and key not in seen:
            seen.add(key)
            yield query

    yield from emit(ids)
    # Short SimpleTrace windows cannot support fixed-width probes. Trim a few
    # potentially altered boundary tokens while keeping as much context as possible.
    for trim in (1, 2, 3):
        if len(ids) > trim:
            yield from emit(ids[trim:])
            yield from emit(ids[:-trim])
        if len(ids) > 2 * trim:
            yield from emit(ids[trim:-trim])

    for width in (128, 64, 32, 16, 8):
        if len(ids) <= width:
            continue
        available = len(ids) - width
        # Avoid both decoded window boundaries. Multiple positions also make
        # it likely that at least one probe is distinctive rather than boilerplate.
        for fraction in (0.25, 0.5, 0.75):
            start = round(available * fraction)
            yield from emit(ids[start : start + width])


def lookup_document(
    request: DocumentRequest,
    engine: InfiniGramEngine,
    tokenizer: Any,
    domain_fields: Sequence[str],
    max_candidate_ranks: int,
) -> tuple[ResolvedDocument | None, str]:
    if not request.excerpts:
        return None, "no saved document excerpt"

    attempted_counts: list[int] = []
    for excerpt in request.excerpts:
        encoded = tokenizer.encode(excerpt, add_special_tokens=False)
        for query_ids in lookup_query_variants(encoded):
            find_result = engine.find(query_ids)
            if "error" in find_result:
                continue
            candidate_count = int(find_result.get("cnt", 0))
            attempted_counts.append(candidate_count)
            if candidate_count == 0 or candidate_count > max_candidate_ranks:
                continue
            for shard, rank in flatten_ranks(find_result):
                raw_doc = engine.get_doc_by_rank(s=shard, rank=rank, max_disp_len=1)
                if "error" in raw_doc:
                    continue
                metadata = parse_index_metadata(raw_doc.get("metadata"))
                candidate_id = str(
                    metadata.get("id")
                    or metadata.get("doc_id")
                    or metadata.get("source")
                    or metadata.get("url")
                    or ""
                ).strip()
                if candidate_id != request.doc_id:
                    continue
                domain, domain_field = metadata_domain(
                    metadata,
                    request.category,
                    domain_fields,
                )
                doc_ix = raw_doc.get("doc_ix")
                return (
                    ResolvedDocument(
                        doc_id=request.doc_id,
                        category=request.category,
                        domain=domain,
                        domain_field=domain_field,
                        shard=shard,
                        doc_ix=int(doc_ix) if isinstance(doc_ix, int) else None,
                        metadata=metadata,
                    ),
                    "",
                )
    return None, f"ID not found among candidate counts {attempted_counts}"


def load_tokenizer(model_id: str) -> Any:
    return AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN"),
        add_bos_token=False,
        add_eos_token=False,
    )


def resolve_documents(
    requests: Mapping[str, DocumentRequest],
    existing: dict[str, ResolvedDocument],
    index_paths: Mapping[str, Path],
    tokenizer: Any,
    domain_fields: Sequence[str],
    max_candidate_ranks: int,
) -> tuple[dict[str, ResolvedDocument], dict[str, str]]:
    resolved = dict(existing)
    pending = [request for doc_id, request in requests.items() if doc_id not in resolved]
    engines: dict[str, InfiniGramEngine] = {}
    failures: dict[str, str] = {}
    progress = tqdm(pending, desc="Resolving indexed document IDs", unit="doc")
    for request in progress:
        index_path = index_paths[request.category]
        if not index_path.is_dir():
            failures[request.doc_id] = f"missing index directory {index_path}"
            continue
        engine = engines.get(request.category)
        if engine is None:
            engine = InfiniGramEngine(
                index_dir=str(index_path),
                eos_token_id=tokenizer.eos_token_id,
                precompute_unigram_logprobs=False,
            )
            engines[request.category] = engine
        item, reason = lookup_document(
            request,
            engine,
            tokenizer,
            domain_fields,
            max_candidate_ranks,
        )
        if item is None:
            failures[request.doc_id] = reason
        else:
            resolved[request.doc_id] = item
    return resolved, failures


def source_key_from_id(doc_id: str) -> str:
    parts = doc_id.split(":")
    if len(parts) < 5 or not parts[2]:
        raise AnalysisError(f"Cannot extract source key from DFM9 ID {doc_id!r}")
    return parts[2]


def prepared_files_by_source_key(prepared_root: Path) -> dict[str, list[Path]]:
    completed_dir = prepared_root / "_reports" / "completed"
    if not completed_dir.is_dir():
        return {}
    root = prepared_root.resolve()
    result: dict[str, list[Path]] = defaultdict(list)
    for report_path in sorted(completed_dir.glob("*.json")):
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        keys = {
            str(item.get("source_key", "")).strip()
            for item in report.get("source_map", [])
            if isinstance(item, dict) and str(item.get("source_key", "")).strip()
        }
        output_paths = []
        for item in report.get("output_files", []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            output_path = Path(item["path"]).resolve()
            try:
                output_path.relative_to(root)
            except ValueError as exc:
                raise AnalysisError(
                    f"Prepared artifact path escapes --prepared-data-root: {output_path}"
                ) from exc
            output_paths.append(output_path)
        for key in keys:
            result[key].extend(output_paths)
    return {key: list(dict.fromkeys(paths)) for key, paths in result.items()}


def resolve_from_prepared_data(
    requests: Mapping[str, DocumentRequest],
    failures: Mapping[str, str],
    prepared_root: Path,
    domain_fields: Sequence[str],
) -> tuple[dict[str, ResolvedDocument], dict[str, str]]:
    """Resolve exact IDs from the JSONL.ZST records that were indexed.

    This bounded fallback is needed only when a decoded SimpleTrace document
    window is too short/common for practical reverse lookup through InfiniGram.
    It scans only artifact groups named by unresolved IDs' source keys.
    """
    if not failures:
        return {}, {}
    files_by_key = prepared_files_by_source_key(prepared_root)
    targets_by_key: dict[str, set[str]] = defaultdict(set)
    for doc_id in failures:
        targets_by_key[source_key_from_id(doc_id)].add(doc_id)

    resolved: dict[str, ResolvedDocument] = {}
    remaining = dict(failures)
    for source_key, target_ids in sorted(targets_by_key.items()):
        pending = set(target_ids)
        paths = files_by_key.get(source_key, [])
        if not paths:
            for doc_id in pending:
                remaining[doc_id] = f"no prepared artifact found for source key {source_key}"
            continue
        for path in paths:
            if not path.is_file():
                continue
            with zstd.open(path, mode="rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise AnalysisError(f"{path}:{line_number}: invalid JSON") from exc
                    doc_id = str(record.get("id", "")).strip()
                    if doc_id not in pending:
                        continue
                    request = requests[doc_id]
                    metadata = {key: value for key, value in record.items() if key != "text"}
                    domain, domain_field = metadata_domain(
                        metadata, request.category, domain_fields
                    )
                    resolved[doc_id] = ResolvedDocument(
                        doc_id=doc_id,
                        category=request.category,
                        domain=domain,
                        domain_field=domain_field,
                        shard=None,
                        doc_ix=None,
                        metadata=metadata,
                        resolution_method="prepared_data_exact_id",
                    )
                    pending.remove(doc_id)
                    remaining.pop(doc_id, None)
                    if not pending:
                        break
            if not pending:
                break
        for doc_id in pending:
            remaining[doc_id] = (
                f"exact ID absent from prepared artifacts for source key {source_key}"
            )
    return resolved, remaining


def distribution_payload(
    document_counts: Mapping[str, int],
    resolved: Mapping[str, ResolvedDocument],
) -> dict[str, Any]:
    occurrence_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    for doc_id, count in document_counts.items():
        domain = resolved[doc_id].domain
        occurrence_counts[domain] += int(count)
        unique_counts[domain] += 1

    total_occurrences = sum(occurrence_counts.values())
    total_unique = sum(unique_counts.values())

    def rows(counts: Counter[str], total: int) -> list[dict[str, Any]]:
        return [
            {
                "domain": domain,
                "count": count,
                "ratio": count / total if total else 0.0,
            }
            for domain, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    return {
        "total_document_occurrences": total_occurrences,
        "document_occurrences_by_domain": rows(occurrence_counts, total_occurrences),
        "total_unique_documents": total_unique,
        "unique_documents_by_domain": rows(unique_counts, total_unique),
    }


def build_report(
    observations: ResultObservations,
    resolved: Mapping[str, ResolvedDocument],
    buckets: Sequence[LengthBucket],
    nv_threshold: float,
    domain_fields: Sequence[str],
    index_paths: Mapping[str, Path],
    document_map_path: Path,
) -> dict[str, Any]:
    bucket_reports: dict[str, Any] = {}
    nv_bucket_reports: dict[str, Any] = {}
    for bucket in buckets:
        label = bucket.label
        bucket_reports[label] = {
            "span_count": observations.span_counts_by_bucket[label],
            "spans_with_documents": observations.spans_with_documents_by_bucket[label],
            **distribution_payload(observations.documents_by_bucket[label], resolved),
        }
        nv_bucket_reports[label] = {
            "spans_with_nv_recall_hit": observations.spans_with_nv_hit_by_bucket[label],
            **distribution_payload(observations.nv_hit_documents_by_bucket[label], resolved),
        }

    return {
        "schema_version": 1,
        "result_file": str(observations.path),
        "document_map": str(document_map_path),
        "selection_semantics": {
            "span_lengths": "Llama-2 tokenizer tokens stored by SimpleTrace",
            "domain_counting_unit": "retrieved document occurrence within a span",
            "unique_document_counting_unit": "distinct SimpleTrace dataset ID within each group",
            "nv_recall_scope": "per retrieved generation-document pair",
            "nv_recall_operator": ">=",
            "nv_recall_threshold": nv_threshold,
        },
        "configuration": {
            "length_buckets": [bucket.label for bucket in buckets],
            "domain_fields": list(domain_fields),
            "index_paths": {key: str(value) for key, value in index_paths.items()},
            "metadata_retrieval": (
                "exact metadata ID via InfiniGram find + get_doc_by_rank; "
                "exact-ID prepared index-input fallback for overly common windows"
            ),
        },
        "input_counts": {
            "generations": observations.generation_count,
            "spans": observations.span_count,
            "spans_with_documents": observations.spans_with_documents,
            "retrieved_document_occurrences": observations.document_occurrence_count,
            "unique_retrieved_document_ids": len(observations.document_ids()),
            "spans_with_nv_recall_hit": observations.spans_with_nv_hit,
            "nv_recall_hit_document_occurrences": sum(observations.nv_hit_documents.values()),
            "unique_nv_recall_hit_document_ids": len(observations.nv_hit_documents),
            "missing_id_occurrences": observations.missing_id_occurrences,
        },
        "span_bucket_domain_distributions": bucket_reports,
        "nv_recall_domain_distribution": distribution_payload(
            observations.nv_hit_documents,
            resolved,
        ),
        "span_bucket_nv_recall_domain_distributions": nv_bucket_reports,
    }


def write_document_map(
    path: Path,
    resolved: Mapping[str, ResolvedDocument],
    required_ids: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(set(required_ids)):
            item = resolved[doc_id]
            handle.write(
                json.dumps(
                    {
                        "id": item.doc_id,
                        "category": item.category,
                        "domain": item.domain,
                        "domain_field": item.domain_field,
                        "index_shard": item.shard,
                        "index_doc_ix": item.doc_ix,
                        "resolution_method": item.resolution_method,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(temp_path, path)


def _plot_modules():
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "dfm9_domain_distribution_plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def _domain_count_map(group: Mapping[str, Any], key: str) -> dict[str, int]:
    return {str(row["domain"]): int(row["count"]) for row in group.get(key, [])}


def plot_bucket_distributions(
    report: Mapping[str, Any],
    section_name: str,
    output_path: Path,
    top_domains: int,
    title: str,
) -> None:
    plt, np = _plot_modules()
    groups = report[section_name]
    bucket_labels = list(groups)
    occurrence_by_bucket = {
        label: _domain_count_map(groups[label], "document_occurrences_by_domain")
        for label in bucket_labels
    }
    unique_by_bucket = {
        label: _domain_count_map(groups[label], "unique_documents_by_domain")
        for label in bucket_labels
    }
    global_counts: Counter[str] = Counter()
    for counts in occurrence_by_bucket.values():
        global_counts.update(counts)
    domains = [domain for domain, _ in global_counts.most_common(top_domains)]
    if set(global_counts) - set(domains):
        domains.append("Other")

    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(bucket_labels) * 1.5), 10))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(domains), 1)))
    for ax, data, subtitle in (
        (axes[0], occurrence_by_bucket, "Retrieved document occurrences"),
        (axes[1], unique_by_bucket, "Unique dataset IDs"),
    ):
        x = np.arange(len(bucket_labels))
        bottom = np.zeros(len(bucket_labels), dtype=float)
        totals = np.array([sum(data[label].values()) for label in bucket_labels], dtype=float)
        for index, domain in enumerate(domains):
            if domain == "Other":
                values = np.array(
                    [sum(v for k, v in data[label].items() if k not in domains) for label in bucket_labels],
                    dtype=float,
                )
            else:
                values = np.array([data[label].get(domain, 0) for label in bucket_labels], dtype=float)
            ratios = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
            ax.bar(x, ratios, bottom=bottom, color=colors[index], label=domain, width=0.78)
            bottom += ratios
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Domain proportion")
        ax.set_title(subtitle)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Domain", loc="upper center", ncol=min(5, max(1, len(labels))))
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nv_distribution(
    report: Mapping[str, Any],
    output_path: Path,
    top_domains: int,
) -> None:
    plt, np = _plot_modules()
    group = report["nv_recall_domain_distribution"]
    occurrence_counts = _domain_count_map(group, "document_occurrences_by_domain")
    unique_counts = _domain_count_map(group, "unique_documents_by_domain")
    domains = [domain for domain, _ in Counter(occurrence_counts).most_common(top_domains)]
    if set(occurrence_counts) - set(domains):
        domains.append("Other")

    def collapse(counts: Mapping[str, int]) -> list[int]:
        values = []
        for domain in domains:
            if domain == "Other":
                values.append(sum(value for key, value in counts.items() if key not in domains))
            else:
                values.append(counts.get(domain, 0))
        return values

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(domains) * 0.42)))
    for ax, counts, subtitle in (
        (axes[0], occurrence_counts, "Retrieved document occurrences"),
        (axes[1], unique_counts, "Unique dataset IDs"),
    ):
        values = collapse(counts)
        total = sum(values)
        ratios = [value / total if total else 0.0 for value in values]
        y = np.arange(len(domains))
        bars = ax.barh(y, ratios, color=plt.cm.tab20(np.linspace(0, 1, max(len(domains), 1))))
        ax.set_yticks(y)
        ax.set_yticklabels(domains)
        ax.invert_yaxis()
        ax.set_xlim(0, max(ratios, default=0.0) + 0.08)
        ax.set_xlabel("Domain proportion")
        ax.set_title(subtitle)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    threshold = report["selection_semantics"]["nv_recall_threshold"]
    fig.suptitle(f"Domain distribution for document-level NV recall ≥ {threshold:g}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_report_plots(
    report: Mapping[str, Any],
    output_dir: Path,
    stem: str,
    top_domains: int,
) -> list[Path]:
    span_path = output_dir / f"{stem}_span_bucket_domains.png"
    nv_path = output_dir / f"{stem}_nv_recall_domains.png"
    span_nv_path = output_dir / f"{stem}_span_bucket_nv_recall_domains.png"
    plot_bucket_distributions(
        report,
        "span_bucket_domain_distributions",
        span_path,
        top_domains,
        "Domain distribution by SimpleTrace span-length bucket",
    )
    plot_nv_distribution(report, nv_path, top_domains)
    plot_bucket_distributions(
        report,
        "span_bucket_nv_recall_domain_distributions",
        span_nv_path,
        top_domains,
        f"Domain distribution by span bucket for NV recall ≥ {report['selection_semantics']['nv_recall_threshold']:g}",
    )
    return [span_path, nv_path, span_nv_path]


def unique_stems(paths: Sequence[Path]) -> list[str]:
    counts = Counter(path.stem for path in paths)
    seen: Counter[str] = Counter()
    stems = []
    for path in paths:
        stem = path.stem
        seen[stem] += 1
        stems.append(stem if counts[stem] == 1 else f"{stem}_{seen[stem]}")
    return stems


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.nv_recall_threshold <= 1.0):
        raise ValueError("--nv-recall-threshold must be between 0 and 1")
    if args.max_candidate_ranks < 1:
        raise ValueError("--max-candidate-ranks must be >= 1")
    if args.top_domains < 1:
        raise ValueError("--top-domains must be >= 1")
    if not args.domain_fields or any(not field.strip() for field in args.domain_fields):
        raise ValueError("--domain-fields must contain non-empty dotted field names")
    missing = [str(path) for path in args.results if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    buckets = parse_length_buckets(args.length_buckets)
    results = [path.resolve() for path in args.results]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.cache_path or output_dir / "document_metadata_cache.jsonl").resolve()
    index_paths = {
        category: path.resolve()
        for category, path in parse_index_overrides(
            args.indexes_root.resolve(),
            args.index_dir,
        ).items()
    }

    requests: dict[str, DocumentRequest] = {}
    observations = [
        collect_observations(path, buckets, args.nv_recall_threshold, requests)
        for path in results
    ]
    print(
        f"Collected {len(requests):,} unique indexed document IDs from "
        f"{len(results):,} result file(s).",
        flush=True,
    )

    cached = {} if args.refresh_cache else load_cache(cache_path, args.domain_fields)
    required_cached = {doc_id: cached[doc_id] for doc_id in requests if doc_id in cached}
    tokenizer = load_tokenizer(args.tokenizer_model)
    resolved, failures = resolve_documents(
        requests,
        required_cached,
        index_paths,
        tokenizer,
        args.domain_fields,
        args.max_candidate_ranks,
    )
    if failures and not args.no_prepared_data_fallback:
        fallback_resolved, failures = resolve_from_prepared_data(
            requests,
            failures,
            args.prepared_data_root.resolve(),
            args.domain_fields,
        )
        resolved.update(fallback_resolved)
        if fallback_resolved:
            print(
                f"Resolved {len(fallback_resolved):,} ambiguous IDs by exact match in "
                f"prepared index inputs.",
                flush=True,
            )
    if resolved:
        write_cache(cache_path, {**cached, **resolved})

    failures_path = output_dir / "resolution_failures.json"
    atomic_write_json(
        failures_path,
        {
            "unresolved_count": len(failures),
            "failures": [
                {"id": doc_id, "reason": reason}
                for doc_id, reason in sorted(failures.items())
            ],
        },
    )
    if failures:
        if not args.allow_unresolved:
            raise AnalysisError(
                f"Failed to resolve {len(failures):,} document IDs; see {failures_path}. "
                "Use --allow-unresolved to include them under __unresolved__."
            )
        for doc_id in failures:
            request = requests[doc_id]
            resolved[doc_id] = ResolvedDocument(
                doc_id=doc_id,
                category=request.category,
                domain=UNRESOLVED_DOMAIN,
                domain_field="resolution_failure",
                shard=None,
                doc_ix=None,
                metadata={},
                resolution_method="unresolved",
            )

    document_map_path = output_dir / "resolved_document_domains.jsonl"
    write_document_map(document_map_path, resolved, requests)

    stems = unique_stems(results)
    reports: list[tuple[str, dict[str, Any]]] = []
    for stem, item in zip(stems, observations, strict=True):
        report = build_report(
            item,
            resolved,
            buckets,
            args.nv_recall_threshold,
            args.domain_fields,
            index_paths,
            document_map_path,
        )
        report_path = output_dir / f"{stem}_domain_distribution.json"
        atomic_write_json(report_path, report)
        reports.append((stem, report))
        print(f"Wrote {report_path}", flush=True)
        if not args.no_plots:
            for plot_path in render_report_plots(report, output_dir, stem, args.top_domains):
                print(f"Wrote {plot_path}", flush=True)

    if len(observations) > 1:
        combined = ResultObservations(path=Path("<combined>"))
        for item in observations:
            combined.merge(item)
        combined_report = build_report(
            combined,
            resolved,
            buckets,
            args.nv_recall_threshold,
            args.domain_fields,
            index_paths,
            document_map_path,
        )
        combined_report["result_files"] = [str(path) for path in results]
        combined_path = output_dir / "combined_domain_distribution.json"
        atomic_write_json(combined_path, combined_report)
        print(f"Wrote {combined_path}", flush=True)
        if not args.no_plots:
            for plot_path in render_report_plots(
                combined_report,
                output_dir,
                "combined",
                args.top_domains,
            ):
                print(f"Wrote {plot_path}", flush=True)

    print(
        f"Resolved {len(requests) - len(failures):,}/{len(requests):,} IDs "
        f"({len(required_cached):,} from cache).",
        flush=True,
    )


if __name__ == "__main__":
    main()

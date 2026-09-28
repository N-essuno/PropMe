#!/usr/bin/env python3
"""Prepare the DFM9 memorisation-source bundle for Infini-gram indexing.

The source bundle is a heterogeneous, manifest-scoped collection.  This
program applies the manifest selectors, converts supported records to the
``{"text": ..., ...metadata...}`` contract required by Infini-gram, and
writes one independently indexable tree per legal-risk category (A-D).

The input bundle is never modified.  Large files are streamed and output is
written as bounded JSONL.ZST parts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq
import zstandard as zstd
# Three D-10 artifacts are JSON arrays despite their `.json` suffix. Keeping
# this exception explicit avoids treating standalone JSON manifests as data.
LEGACY_JSON_ARRAY_NAMES = frozenset(
    {"data.json", "glaive-function-calling-v2.json", "xlam_function_calling_60k.json"}
)

from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path(
    "/work/olmotrace/mimir_propme/dfm9_memorisation_sources"
)
SUPPORTED_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst")
RISK_CATEGORIES = ("A", "B", "C", "D")
EXPECTED_SELECTED_ROWS = {
    "A-06": 1_104,
    "B-03": 5_169,
    "B-04": 335_117,
    "C-09": 89_982,
    "C-10": 35_357,
    "C-11": 35_380,
    "C-12": 2_688,
    "C-13": 42_026,
    "C-14": 22_280,
}
EXPECTED_B03_DOCUMENTS = 2_607
B03_DATASETS = frozenset(
    {"instruction-generation", "instruction-generation-ifeval"}
)
A06_SUBSETS = frozenset({"dkmedier", "odense", "danskerhverv"})
OPENHERMES_SELECTORS = {
    "C-11": "airoboros2.2",
    "C-12": "caseus_custom",
    "C-13": "cot_alpaca_gpt4",
    "C-14": "platypus",
}
CONTENT_FIELDS = frozenset(
    {
        "text",
        "instruction",
        "response",
        "problem",
        "generated_solution",
        "solution",
        "question",
        "answer",
        "prompt",
        "messages",
        "conversations",
        "conversation_a",
        "conversation_b",
        "system_prompt",
        "system_prompt_a",
        "system_prompt_b",
        "opening_msg",
        "query",
        "golden_answer",
        "trace",
        "tools",
        "response_content",
        "question_content",
        "system",
        "chat",
        "answers",
    }
)
DOCUMENT_RE = re.compile(r"<document>(.*?)</document>", re.IGNORECASE | re.DOTALL)


class PreparationError(RuntimeError):
    """Raised when source scope or content cannot be handled safely."""


@dataclass(frozen=True)
class ManifestEntry:
    row_number: int
    category: str
    cohort: str
    basis: str
    material_role: str
    selector: str
    source_path: str
    assembled_path: str
    source_kind: str
    size_bytes: str = ""
    note: str = ""


@dataclass(frozen=True)
class RouteSpec:
    entry: ManifestEntry
    mode: str
    expected_selected_rows: int | None = None
    allowed_values: tuple[str, ...] = ()
    audit_hash_rows: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class SourceFile:
    path: str
    relative_path: str
    source_key: str


@dataclass(frozen=True)
class GroupTask:
    group_key: str
    canonical_source_path: str
    files: tuple[SourceFile, ...]
    routes: tuple[RouteSpec, ...]


@dataclass(frozen=True)
class WorkerConfig:
    output_root: str
    target_part_bytes: int
    compression_level: int
    parquet_batch_size: int
    resume: bool


@dataclass
class RouteStats:
    category: str
    cohort: str
    source_path: str
    assembled_path: str
    selector: str
    material_role: str
    input_rows: int = 0
    selected_rows: int = 0
    emitted_rows: int = 0
    empty_rows: int = 0
    duplicate_rows: int = 0
    direct_audit_hash_matches: int = 0
    direct_audit_hash_mismatches: int = 0


@dataclass
class OutputFileStats:
    path: str
    records: int
    uncompressed_bytes: int
    compressed_bytes: int
    sha256: str


@dataclass
class GroupResult:
    group_key: str
    canonical_source_path: str
    route_stats: list[dict[str, Any]] = field(default_factory=list)
    output_files: list[dict[str, Any]] = field(default_factory=list)
    source_map: list[dict[str, str]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)
    resumed: bool = False


@dataclass
class PreparedDocument:
    variant: str
    text: str
    consumed_fields: set[str]
    structural_metadata: dict[str, Any] = field(default_factory=dict)


def problem_rows(
    entries: Sequence[ManifestEntry],
    *,
    phase: str,
    problem_type: str,
    message: str,
    canonical_source_path: str | None = None,
    expected: int | str | None = None,
    actual: int | str | None = None,
    traceback_text: str = "",
) -> list[dict[str, Any]]:
    """Create one report row per affected manifest route."""
    return [
        {
            "severity": "error",
            "phase": phase,
            "problem_type": problem_type,
            "category": entry.category,
            "cohort": entry.cohort,
            "canonical_source_path": canonical_source_path or entry.source_path,
            "assembled_path": entry.assembled_path,
            "message": message,
            "expected": expected,
            "actual": actual,
            "traceback": traceback_text,
        }
        for entry in entries
    ]


def stable_key(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def to_json_safe(value: Any) -> Any:
    """Convert Arrow/numpy/Python values to deterministic JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {
            str(key): to_json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return sorted((to_json_safe(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if hasattr(value, "as_py"):
        return to_json_safe(value.as_py())
    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def textual_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool, Decimal)):
        return str(value)
    if isinstance(value, list):
        parts = [textual_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        for key in ("content", "value", "text"):
            if key in value and value[key] is not None:
                return textual_value(value[key])
        return canonical_json(value)
    return str(value)


def message_text(messages: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(messages, list):
        raise PreparationError(f"Expected a message list, got {type(messages).__name__}")
    text_parts: list[str] = []
    structure: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            part = textual_value(message)
            if part:
                text_parts.append(part)
            structure.append({"record_type": type(message).__name__})
            continue

        stripped: dict[str, Any] = {}
        for key, value in message.items():
            if key in {"content", "value", "text"}:
                part = textual_value(value)
                if part:
                    text_parts.append(part)
            elif key in {"tool_calls", "function_call", "reasoning", "analysis"}:
                if value not in (None, "", [], {}):
                    text_parts.append(canonical_json(value))
            else:
                stripped[str(key)] = to_json_safe(value)
        structure.append(stripped)
    return "\n\n".join(text_parts), structure


def serialize_record(record: Mapping[str, Any]) -> list[PreparedDocument]:
    """Convert one heterogeneous source record into one or more documents."""
    # A non-empty top-level text is authoritative and remains byte-for-byte
    # unchanged. Empty text falls through to any usable alternative content.
    if isinstance(record.get("text"), str) and record["text"].strip():
        structural: dict[str, Any] = {}
        if isinstance(record.get("messages"), list):
            _, structural["message_structure"] = message_text(record["messages"])
        return [
            PreparedDocument(
                variant="doc",
                text=record["text"],
                consumed_fields={key for key in CONTENT_FIELDS if key in record},
                structural_metadata=structural,
            )
        ]

    if "conversation_a" in record or "conversation_b" in record:
        documents: list[PreparedDocument] = []
        for variant, key in (("a", "conversation_a"), ("b", "conversation_b")):
            value = record.get(key)
            if value in (None, "", []):
                continue
            text, structure = (
                message_text(value)
                if isinstance(value, list)
                else (textual_value(value), [])
            )
            prefix_fields = [
                name
                for name in ("system_prompt", f"system_prompt_{variant}", "opening_msg")
                if record.get(name) not in (None, "", [], {})
            ]
            parts = [textual_value(record[name]) for name in prefix_fields]
            if text:
                parts.append(text)
            documents.append(
                PreparedDocument(
                    variant=variant,
                    text="\n\n".join(parts),
                    consumed_fields={key for key in CONTENT_FIELDS if key in record},
                    structural_metadata={
                        "conversation_variant": variant,
                        "message_structure": structure,
                    },
                )
            )
        if documents:
            return documents

    for field_name in ("messages", "conversations"):
        value = record.get(field_name)
        if isinstance(value, list) and value:
            text, structure = message_text(value)
            parts: list[str] = []
            if record.get("system") not in (None, "", [], {}):
                parts.append(textual_value(record["system"]))
            if record.get("tools") not in (None, [], {}):
                parts.append(canonical_json(record["tools"]))
            if text:
                parts.append(text)
            return [
                PreparedDocument(
                    variant="doc",
                    text="\n\n".join(parts),
                    consumed_fields={key for key in CONTENT_FIELDS if key in record},
                    structural_metadata={"message_structure": structure},
                )
            ]

    pair_candidates = (
        ("instruction", "response"),
        ("problem", "generated_solution"),
        ("problem", "solution"),
        ("question", "answer"),
        ("question", "response"),
        ("prompt", "response"),
        ("query", "golden_answer"),
        ("query", "answers"),
        ("system", "chat"),
    )
    for left, right in pair_candidates:
        if left in record and right in record:
            fields = [left, right]
            if "system_prompt" in record:
                fields.insert(0, "system_prompt")
            parts: list[str] = []
            if record.get("tools") not in (None, [], {}):
                parts.append(canonical_json(record["tools"]))
            parts.extend(textual_value(record.get(key)) for key in fields)
            return [
                PreparedDocument(
                    variant="doc",
                    text="\n\n".join(part for part in parts if part),
                    consumed_fields={key for key in CONTENT_FIELDS if key in record},
                )
            ]

    for single in ("trace", "query", "prompt", "response_content", "question_content"):
        if single in record and record[single] not in (None, "", [], {}):
            return [
                PreparedDocument(
                    variant="doc",
                    text=textual_value(record[single]),
                    consumed_fields={single},
                )
            ]

    if isinstance(record.get("text"), str):
        # This is a recognized text schema with no usable content in an
        # alternative field. Return it so the common emission path records it
        # in empty_rows and drops it.
        return [
            PreparedDocument(
                variant="doc",
                text=record["text"],
                consumed_fields={key for key in CONTENT_FIELDS if key in record},
            )
        ]

    raise PreparationError(
        "Unrecognized record schema with keys: " + ", ".join(sorted(record))
    )


def source_metadata(
    record: Mapping[str, Any], document: PreparedDocument
) -> dict[str, Any]:
    metadata = {
        str(key): to_json_safe(value)
        for key, value in record.items()
        if key not in document.consumed_fields
    }
    metadata.update(document.structural_metadata)
    return metadata


def is_supported_data_path(path: Path) -> bool:
    value = path.name.lower()
    return any(value.endswith(suffix) for suffix in SUPPORTED_SUFFIXES) or (
        value in LEGACY_JSON_ARRAY_NAMES and "D-10" in path.parts
    )


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without retaining the complete file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            block = handle.read(chunk_size)
            if block:
                buffer += block
            else:
                eof = True

        def ensure_character() -> bool:
            while position >= len(buffer) and not eof:
                refill()
            return position < len(buffer)

        def skip_whitespace() -> None:
            nonlocal position
            while ensure_character() and buffer[position].isspace():
                position += 1

        def finish_array() -> None:
            nonlocal position
            position += 1
            trailing = buffer[position:] + handle.read()
            if trailing.strip():
                raise PreparationError(f"{path}: data after top-level JSON array")

        refill()
        skip_whitespace()
        if not ensure_character() or buffer[position] != "[":
            raise PreparationError(f"{path}: expected a top-level JSON array")
        position += 1
        item_number = 0
        first = True

        while True:
            skip_whitespace()
            if not ensure_character():
                raise PreparationError(f"{path}: unterminated JSON array")

            if buffer[position] == "]":
                finish_array()
                return

            if not first:
                if buffer[position] != ",":
                    raise PreparationError(
                        f"{path}: expected a comma before array item {item_number}"
                    )
                position += 1
                skip_whitespace()
                if not ensure_character():
                    raise PreparationError(f"{path}: unterminated JSON array")
                if buffer[position] == "]":
                    raise PreparationError(f"{path}: trailing comma in JSON array")

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise PreparationError(
                            f"{path}: malformed JSON array item {item_number}: {exc}"
                        ) from exc
                    refill()
            if not isinstance(value, dict):
                raise PreparationError(
                    f"{path}: array item {item_number} is {type(value).__name__}, "
                    "expected object"
                )
            yield value
            item_number += 1
            first = False


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".jsonl.gz"):
        import gzip

        handle: Any = gzip.open(path, "rt", encoding="utf-8")
    elif path.name.endswith(".jsonl.zst"):
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        handle = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise PreparationError(
                    f"{path}:{line_number}: expected JSON object, got {type(value).__name__}"
                )
            yield value
    finally:
        handle.close()


def iter_records(path: Path, parquet_batch_size: int) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=parquet_batch_size):
            yield from batch.to_pylist()
        return
    if path.name.lower() in LEGACY_JSON_ARRAY_NAMES and "D-10" in path.parts:
        yield from iter_json_array(path)
        return
    yield from iter_jsonl(path)


class HashingSink:
    def __init__(self, path: Path):
        self.handle = path.open("wb")
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return self.handle.write(data)

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def writable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self.handle.closed


class RollingZstdWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        target_uncompressed_bytes: int,
        compression_level: int,
    ):
        self.output_dir = output_dir
        self.target = target_uncompressed_bytes
        self.level = compression_level
        self.part_number = 0
        self.sink: HashingSink | None = None
        self.stream: Any = None
        self.temp_path: Path | None = None
        self.final_path: Path | None = None
        self.current_records = 0
        self.current_bytes = 0
        self.files: list[OutputFileStats] = []

    def _open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = f"part-{self.part_number:05d}.jsonl.zst"
        self.final_path = self.output_dir / name
        self.temp_path = self.output_dir / f".{name}.{os.getpid()}.tmp"
        self.sink = HashingSink(self.temp_path)
        self.stream = zstd.ZstdCompressor(level=self.level).stream_writer(
            self.sink, closefd=False
        )

    def write(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if self.current_records and self.current_bytes + len(encoded) > self.target:
            self._finish_part()
        if self.stream is None:
            self._open()
        self.stream.write(encoded)
        self.current_records += 1
        self.current_bytes += len(encoded)

    def _finish_part(self) -> None:
        if self.stream is None or self.sink is None:
            return
        self.stream.flush(zstd.FLUSH_FRAME)
        self.stream.close()
        self.sink.close()
        assert self.temp_path is not None and self.final_path is not None
        os.replace(self.temp_path, self.final_path)
        self.files.append(
            OutputFileStats(
                path=str(self.final_path),
                records=self.current_records,
                uncompressed_bytes=self.current_bytes,
                compressed_bytes=self.final_path.stat().st_size,
                sha256=self.sink.digest.hexdigest(),
            )
        )
        self.part_number += 1
        self.sink = None
        self.stream = None
        self.temp_path = None
        self.final_path = None
        self.current_records = 0
        self.current_bytes = 0

    def close(self) -> list[OutputFileStats]:
        self._finish_part()
        return self.files


def read_manifest(source_root: Path) -> list[ManifestEntry]:
    path = source_root / "manifest.tsv"
    if not path.is_file():
        raise PreparationError(f"Missing manifest: {path}")
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "category",
            "cohort",
            "basis",
            "material_role",
            "selector",
            "source_path",
            "assembled_path",
            "source_kind",
            "size_bytes",
            "note",
        }
        if set(reader.fieldnames or ()) != expected:
            raise PreparationError(
                f"Unexpected manifest columns: {reader.fieldnames}; expected {sorted(expected)}"
            )
        for row_number, row in enumerate(reader, start=2):
            entries.append(ManifestEntry(row_number=row_number, **row))
    return entries


def load_b03_audit(source_root: Path) -> tuple[tuple[str, int, int], ...]:
    path = (
        source_root
        / "B/B-03/legal__registers__dfm9-euroblocks-embedded-seed-documents.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(
            (
                row["document_sha256"],
                int(row["document_chars"]),
                int(row["occurrences"]),
            )
            for row in csv.DictReader(handle)
        )
    if len(rows) != EXPECTED_B03_DOCUMENTS:
        raise PreparationError(f"Expected 2,607 B-03 audit rows, found {len(rows)}")
    return rows


def load_b04_datasets(source_root: Path) -> tuple[str, ...]:
    path = source_root / "B/B-04/legal__registers__dfm9-euroblocks-seed-risk.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = tuple(
            row["subset"]
            for row in csv.DictReader(handle)
            if row["subset"] not in B03_DATASETS
        )
    if not values:
        raise PreparationError("B-04 audit register contains no routable subsets")
    return values


def route_for_entry(entry: ManifestEntry, source_root: Path) -> RouteSpec | None:
    if entry.material_role == "audit_evidence":
        return None
    if entry.cohort == "D-05":
        return None
    if entry.cohort == "A-06":
        return RouteSpec(entry, "a06", EXPECTED_SELECTED_ROWS[entry.cohort])
    if entry.cohort == "B-03":
        return RouteSpec(
            entry,
            "b03",
            EXPECTED_SELECTED_ROWS[entry.cohort],
            audit_hash_rows=load_b03_audit(source_root),
        )
    if entry.cohort == "B-04":
        return RouteSpec(
            entry,
            "b04",
            EXPECTED_SELECTED_ROWS[entry.cohort],
            allowed_values=load_b04_datasets(source_root),
        )
    if entry.cohort == "C-09":
        return RouteSpec(entry, "c09", EXPECTED_SELECTED_ROWS[entry.cohort])
    if entry.cohort == "C-10":
        return RouteSpec(entry, "c10", EXPECTED_SELECTED_ROWS[entry.cohort])
    if entry.cohort in OPENHERMES_SELECTORS:
        return RouteSpec(
            entry,
            "openhermes",
            EXPECTED_SELECTED_ROWS[entry.cohort],
            allowed_values=(OPENHERMES_SELECTORS[entry.cohort],),
        )
    if entry.cohort == "D-01":
        return RouteSpec(entry, "d01")
    if entry.cohort == "D-02":
        return RouteSpec(entry, "d02")
    if entry.selector == "all records":
        return RouteSpec(entry, "all")
    if entry.cohort == "B-05" and entry.selector.startswith("upstream_repo="):
        return RouteSpec(entry, "all")
    raise PreparationError(
        f"Unknown selector at manifest row {entry.row_number}: "
        f"{entry.cohort} / {entry.selector!r}"
    )


def artifact_inventory(path: Path, source_kind: str) -> list[tuple[str, int]]:
    if not path.exists():
        raise PreparationError(f"Manifest artifact does not exist: {path}")
    if source_kind == "file":
        if not path.is_file():
            raise PreparationError(f"Expected file artifact: {path}")
        if not is_supported_data_path(path):
            raise PreparationError(f"Non-data file routed for indexing: {path}")
        return [(path.name, path.stat().st_size)]
    if source_kind != "directory" or not path.is_dir():
        raise PreparationError(f"Expected directory artifact: {path}")
    result: list[tuple[str, int]] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or not is_supported_data_path(candidate):
            continue
        if ".cache" in candidate.relative_to(path).parts:
            continue
        result.append((candidate.relative_to(path).as_posix(), candidate.stat().st_size))
    if not result:
        raise PreparationError(f"No supported data files under: {path}")
    return sorted(result)


def build_tasks(
    source_root: Path, entries: Sequence[ManifestEntry], categories: set[str]
) -> tuple[list[GroupTask], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[ManifestEntry, RouteSpec]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for entry in entries:
        if entry.category not in categories:
            continue
        if entry.category not in RISK_CATEGORIES:
            raise PreparationError(f"Invalid risk category: {entry.category!r}")
        if entry.material_role == "audit_evidence":
            skipped.append({**asdict(entry), "reason": "audit_evidence"})
            continue
        if entry.cohort == "D-05":
            skipped.append(
                {
                    **asdict(entry),
                    "reason": "unresolved_selector: Mixture-of-Thoughts rows lack row provenance",
                }
            )
            continue
        try:
            route = route_for_entry(entry, source_root)
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, FileNotFoundError):
                problem_type = "missing_supporting_artifact"
            elif "Unknown selector" in message:
                problem_type = "unsupported_selector"
            else:
                problem_type = "route_configuration_error"
            problems.extend(
                problem_rows(
                    [entry],
                    phase="task_build",
                    problem_type=problem_type,
                    message=message,
                )
            )
            continue
        if route is None:
            problems.extend(
                problem_rows(
                    [entry],
                    phase="task_build",
                    problem_type="unrouted_manifest_entry",
                    message=f"Unexpected unrouted entry: {entry}",
                )
            )
            continue
        grouped[entry.source_path].append((entry, route))

    tasks: list[GroupTask] = []
    for canonical_source, pairs in sorted(grouped.items()):
        try:
            inventories: list[tuple[ManifestEntry, Path, list[tuple[str, int]]]] = []
            for entry, _ in pairs:
                assembled = source_root / entry.assembled_path
                inventories.append(
                    (entry, assembled, artifact_inventory(assembled, entry.source_kind))
                )
            baseline = inventories[0][2]
            for entry, _, inventory in inventories[1:]:
                if inventory != baseline:
                    raise PreparationError(
                        f"Repeated artifact inventory differs for {canonical_source}: "
                        f"manifest row {entry.row_number}"
                    )
        except Exception as exc:
            message = str(exc)
            if "does not exist" in message:
                problem_type = "missing_manifest_artifact"
            elif "No supported data files" in message:
                problem_type = "no_supported_data_files"
            elif "inventory differs" in message:
                problem_type = "artifact_inventory_mismatch"
            else:
                problem_type = "artifact_inventory_error"
            problems.extend(
                problem_rows(
                    [entry for entry, _ in pairs],
                    phase="task_build",
                    problem_type=problem_type,
                    message=message,
                    canonical_source_path=canonical_source,
                )
            )
            continue
        chosen_entry, chosen_path, inventory = inventories[0]
        group_key = stable_key(canonical_source)
        files: list[SourceFile] = []
        for relative_path, _ in inventory:
            actual_path = (
                chosen_path
                if chosen_entry.source_kind == "file"
                else chosen_path / relative_path
            )
            key_material = f"{canonical_source}\0{relative_path}"
            files.append(
                SourceFile(
                    path=str(actual_path),
                    relative_path=relative_path,
                    source_key=stable_key(key_material),
                )
            )
        routes = tuple(route for _, route in pairs)
        route_cohorts = [route.entry.cohort for route in routes]
        if len(route_cohorts) != len(set(route_cohorts)):
            problems.extend(
                problem_rows(
                    [entry for entry, _ in pairs],
                    phase="task_build",
                    problem_type="duplicate_cohort_route",
                    message=(
                        "Multiple index routes for one cohort/source: "
                        f"{canonical_source}"
                    ),
                    canonical_source_path=canonical_source,
                )
            )
            continue
        tasks.append(
            GroupTask(
                group_key=group_key,
                canonical_source_path=canonical_source,
                files=tuple(files),
                routes=routes,
            )
        )
    return tasks, skipped, problems

def route_matches(route: RouteSpec, record: Mapping[str, Any]) -> bool:
    if route.mode in {"all", "d01", "d02"}:
        return True
    if route.mode == "a06":
        return record.get("subset") in A06_SUBSETS
    if route.mode == "b03":
        return record.get("dataset") in B03_DATASETS
    if route.mode == "b04":
        return record.get("dataset") in route.allowed_values
    if route.mode == "c09":
        return record.get("source") == "ai2-adapt-dev/flan_v2_converted"
    if route.mode == "c10":
        value = record.get("dataset")
        return isinstance(value, str) and value.startswith("science.")
    if route.mode == "openhermes":
        return record.get("openhermes_source") == route.allowed_values[0]
    raise PreparationError(f"Unknown route mode: {route.mode}")


def d01_documents(record: Mapping[str, Any]) -> list[PreparedDocument]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise PreparationError("D-01 record has no messages list")
    _, complete_structure = message_text(messages)
    documents: list[PreparedDocument] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        if role not in {"user", "human"}:
            continue
        value = message.get("content", message.get("value", message.get("text")))
        documents.append(
            PreparedDocument(
                variant=f"m{index:03d}",
                text=textual_value(value),
                consumed_fields={key for key in CONTENT_FIELDS if key in record},
                structural_metadata={
                    "message_structure": complete_structure,
                    "selected_message_index": index,
                    "selected_message_role": role,
                },
            )
        )
    return documents


def extract_b03_document(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise PreparationError("B-03 record has no conversations list")
    joined = "\n".join(
        textual_value(message.get("value", message.get("content")))
        if isinstance(message, Mapping)
        else textual_value(message)
        for message in conversations
    )
    matches = DOCUMENT_RE.findall(joined)
    if len(matches) != 1:
        raise PreparationError(
            f"Expected exactly one <document> block in B-03 row, found {len(matches)}"
        )
    return matches[0]


def normalized_record(
    route: RouteSpec,
    source_file: SourceFile,
    row_index: int,
    record: Mapping[str, Any],
    document: PreparedDocument,
) -> dict[str, Any]:
    entry = route.entry
    identifier = (
        f"{entry.category}:{entry.cohort}:{source_file.source_key}:"
        f"{row_index:09d}:{document.variant}"
    )
    return {
        "text": document.text,
        "id": identifier,
        "risk_category": entry.category,
        "cohort": entry.cohort,
        "basis": entry.basis,
        "material_role": entry.material_role,
        "selector": entry.selector,
        "manifest_source_path": entry.source_path,
        "manifest_assembled_path": entry.assembled_path,
        "manifest_source_kind": entry.source_kind,
        "manifest_size_bytes": entry.size_bytes,
        "manifest_note": entry.note,
        "manifest_row_number": entry.row_number,
        "source_key": source_file.source_key,
        "source_file": source_file.relative_path,
        "source_record_index": row_index,
        "source_metadata": source_metadata(record, document),
    }


def marker_path(output_root: Path, task: GroupTask) -> Path:
    route_scope = "\0".join(
        sorted(
            f"{route.entry.category}:{route.entry.cohort}:{route.entry.row_number}"
            for route in task.routes
        )
    )
    return output_root / "_reports" / "completed" / f"{task.group_key}-{stable_key(route_scope, 12)}.json"


def route_output_dir(output_root: Path, route: RouteSpec, group_key: str) -> Path:
    return output_root / route.entry.category / route.entry.cohort / group_key


def process_group(task: GroupTask, config: WorkerConfig) -> GroupResult:
    output_root = Path(config.output_root)
    marker = marker_path(output_root, task)
    problems: list[dict[str, Any]] = []
    if config.resume and marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            for item in payload["output_files"]:
                path = Path(item["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != item["compressed_bytes"]
                    or sha256_file(path) != item["sha256"]
                ):
                    raise PreparationError(
                        f"Resume marker references missing/changed file: {path}"
                    )
            payload["resumed"] = True
            return GroupResult(**payload)
        except Exception as exc:
            problems.extend(
                problem_rows(
                    [route.entry for route in task.routes],
                    phase="resume",
                    problem_type="invalid_resume_marker",
                    message=f"Regenerated source group after resume check failed: {exc}",
                    canonical_source_path=task.canonical_source_path,
                )
            )
            marker.unlink(missing_ok=True)

    for route in task.routes:
        target = route_output_dir(output_root, route, task.group_key)
        if target.exists():
            shutil.rmtree(target)

    writers = {
        route.entry.cohort: RollingZstdWriter(
            route_output_dir(output_root, route, task.group_key),
            target_uncompressed_bytes=config.target_part_bytes,
            compression_level=config.compression_level,
        )
        for route in task.routes
    }
    stats = {
        route.entry.cohort: RouteStats(
            category=route.entry.category,
            cohort=route.entry.cohort,
            source_path=route.entry.source_path,
            assembled_path=route.entry.assembled_path,
            selector=route.entry.selector,
            material_role=route.entry.material_role,
        )
        for route in task.routes
    }
    source_map: list[dict[str, str]] = []
    b03_docs: dict[str, dict[str, Any]] = {}
    b03_counts: Counter[str] = Counter()

    try:
        for source_file in task.files:
            source_map.append(
                {
                    "source_key": source_file.source_key,
                    "canonical_source_path": task.canonical_source_path,
                    "source_file": source_file.relative_path,
                    "actual_path": source_file.path,
                }
            )
            for row_index, record in enumerate(
                iter_records(Path(source_file.path), config.parquet_batch_size)
            ):
                for route in task.routes:
                    route_stat = stats[route.entry.cohort]
                    route_stat.input_rows += 1
                    if not route_matches(route, record):
                        continue
                    route_stat.selected_rows += 1

                    if route.mode == "b03":
                        text = extract_b03_document(record)
                        audit_text = text.strip()
                        digest = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
                        b03_counts[digest] += 1
                        if digest in b03_docs:
                            route_stat.duplicate_rows += 1
                            continue
                        document = PreparedDocument(
                            variant="document",
                            text=text,
                            consumed_fields={"conversations"},
                        )
                        b03_docs[digest] = normalized_record(
                            route,
                            source_file,
                            row_index,
                            record,
                            document,
                        )
                        continue

                    documents = (
                        d01_documents(record)
                        if route.mode == "d01"
                        else serialize_record(record)
                    )
                    for document in documents:
                        if not document.text.strip():
                            route_stat.empty_rows += 1
                            continue
                        writers[route.entry.cohort].write(
                            normalized_record(
                                route, source_file, row_index, record, document
                            )
                        )
                        route_stat.emitted_rows += 1

        for route in task.routes:
            if route.mode != "b03":
                continue
            route_stat = stats[route.entry.cohort]
            audit_hashes = {row[0] for row in route.audit_hash_rows}
            direct_matches = set(b03_docs) & audit_hashes
            route_stat.direct_audit_hash_matches = len(direct_matches)
            route_stat.direct_audit_hash_mismatches = len(b03_docs) - len(direct_matches)

            actual_shape = Counter(
                (len(b03_docs[digest]["text"].strip()), occurrences)
                for digest, occurrences in b03_counts.items()
            )
            audit_shape = Counter(
                (length, occurrences)
                for _, length, occurrences in route.audit_hash_rows
            )
            if actual_shape != audit_shape:
                problems.extend(
                    problem_rows(
                        [route.entry],
                        phase="validation",
                        problem_type="b03_audit_shape_mismatch",
                        message=(
                            "B-03 extracted document length/occurrence multiset differs "
                            "from audit"
                        ),
                    )
                )
            if len(b03_docs) != EXPECTED_B03_DOCUMENTS:
                problems.extend(
                    problem_rows(
                        [route.entry],
                        phase="validation",
                        problem_type="b03_unique_document_count_mismatch",
                        message=(
                            f"Expected {EXPECTED_B03_DOCUMENTS} unique B-03 documents, "
                            f"found {len(b03_docs)}"
                        ),
                        expected=EXPECTED_B03_DOCUMENTS,
                        actual=len(b03_docs),
                    )
                )
            for digest in sorted(b03_docs):
                output = b03_docs[digest]
                output["source_metadata"]["document_sha256"] = hashlib.sha256(
                    output["text"].encode("utf-8")
                ).hexdigest()
                output["source_metadata"]["audit_normalized_document_sha256"] = digest
                output["source_metadata"]["document_occurrences"] = b03_counts[digest]
                writers[route.entry.cohort].write(output)
                route_stat.emitted_rows += 1

        output_files: list[dict[str, Any]] = []
        for writer in writers.values():
            output_files.extend(asdict(item) for item in writer.close())

        for route in task.routes:
            route_stat = stats[route.entry.cohort]
            if (
                route.expected_selected_rows is not None
                and route_stat.selected_rows != route.expected_selected_rows
            ):
                problems.extend(
                    problem_rows(
                        [route.entry],
                        phase="validation",
                        problem_type="selected_row_count_mismatch",
                        message=(
                            f"{route.entry.cohort}: expected "
                            f"{route.expected_selected_rows:,} selected rows, found "
                            f"{route_stat.selected_rows:,}"
                        ),
                        expected=route.expected_selected_rows,
                        actual=route_stat.selected_rows,
                    )
                )

        result = GroupResult(
            group_key=task.group_key,
            canonical_source_path=task.canonical_source_path,
            route_stats=[asdict(value) for value in stats.values()],
            output_files=output_files,
            source_map=source_map,
            problems=problems,
        )
        atomic_write_json(marker, asdict(result))
        return result
    except Exception:
        for writer in writers.values():
            try:
                if writer.stream is not None:
                    writer.stream.close()
                if writer.sink is not None:
                    writer.sink.close()
            except Exception:
                pass
        # A failed group may already have atomically completed parts. Remove
        # all route directories so incomplete data cannot be indexed.
        for route in task.routes:
            try:
                shutil.rmtree(
                    route_output_dir(output_root, route, task.group_key),
                    ignore_errors=True,
                )
            except Exception:
                pass
        raise


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: to_json_safe(row.get(column)) for column in columns})
    os.replace(temp_path, path)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in ("pyarrow", "zstandard", "transformers", "infini-gram"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def index_commands(
    output_root: Path,
    categories: Sequence[str],
    problems: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = ["# DFM9 category indexing commands", ""]
    if problems:
        lines.extend(
            [
                "**Warning:** preparation reported problems. Review `problems.tsv` ",
                "before indexing; affected category inputs may be incomplete.",
                "",
            ]
        )
    for category in categories:
        shards = 2 if category == "C" else 1
        category_problem_count = sum(
            str(problem.get("category", "")) == category for problem in problems
        )
        lines.extend([f"## Category {category}", ""])
        if category_problem_count:
            lines.extend(
                [
                    f"This category has {category_problem_count} reported problem(s).",
                    "",
                ]
            )
        lines.extend(
            [
                "```bash",
                "conda run -n olmotrace_test python -m infini_gram.indexing \\",
                f"  --data_dir {output_root / category} \\",
                f"  --save_dir <index-root>/{category} \\",
                f"  --temp_dir <scratch-root>/{category} \\",
                "  --tokenizer llama \\",
                "  --cpus 128 \\",
                "  --mem 350 \\",
                f"  --shards {shards} \\",
                "  --batch_size 1024 \\",
                "  --add_metadata \\",
                "  --ulimit 1048576",
                "```",
                "",
                "```bash",
                "conda run -n olmotrace_test python 02_unigram_probs/compute_unigrams.py \\",
                f"  --index-dir <index-root>/{category} \\",
                f"  --output-path 02_unigram_probs/unigram_probs_dfm9_{category}.json \\",
                "  --tokenizer-model meta-llama/Llama-2-7b-hf \\",
                "  --example-token a \\",
                "  --top-k 10",
                "```",
                "",
            ]
        )
    return "\n".join(lines)

def excluded_files_markdown(
    source_root: Path,
    skipped: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
) -> str:
    """Render manifest artifacts that are wholly absent from index inputs."""
    rows_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in skipped:
        category = str(row["category"])
        assembled_path = str(row["assembled_path"])
        artifact = source_root / assembled_path
        file_names: list[str]
        if artifact.is_file():
            file_names = [assembled_path]
        elif artifact.is_dir():
            file_names = [
                (Path(assembled_path) / candidate.relative_to(artifact)).as_posix()
                for candidate in sorted(artifact.rglob("*"))
                if candidate.is_file()
                and ".cache" not in candidate.relative_to(artifact).parts
                and is_supported_data_path(candidate)
            ]
            if not file_names:
                file_names = [f"{assembled_path}/ (no data files found)"]
        else:
            file_names = [f"{assembled_path} (missing)"]
        for file_name in file_names:
            rows_by_category[category].append(
                {
                    "cohort": row["cohort"],
                    "path": file_name,
                    "reason": row["reason"],
                }
            )

    lines = [
        "# Files fully excluded from DFM9 index inputs",
        "",
        "This report lists manifest artifacts for which no records are emitted. ",
        "Files used with row- or message-level selectors are not listed as fully ",
        "excluded, because some of their records still enter an index.",
        "",
    ]
    for category in categories:
        lines.extend([f"## {category}", ""])
        rows = sorted(
            rows_by_category.get(category, []),
            key=lambda row: (str(row["cohort"]), str(row["path"])),
        )
        if not rows:
            lines.extend(["No manifest data files are fully excluded.", ""])
            continue
        current_cohort = None
        for row in rows:
            if row["cohort"] != current_cohort:
                current_cohort = row["cohort"]
                lines.extend([f"### {current_cohort}", ""])
            lines.append(f"- `{row['path']}` — {row['reason']}")
        lines.append("")
    lines.extend(
        [
            "## Scope note",
            "",
            "Missing or malformed manifest artifacts are listed in `problems.tsv`, ",
            "not as intentional exclusions in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    source_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    results: Sequence[GroupResult],
    skipped: Sequence[Mapping[str, Any]],
    problems: Sequence[Mapping[str, Any]],
) -> None:
    reports = output_root / "_reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = source_root / "manifest.tsv"
    gaps = source_root / "gaps.tsv"
    shutil.copy2(manifest, reports / "manifest.tsv")
    if gaps.is_file():
        shutil.copy2(gaps, reports / "gaps.tsv")

    route_rows = [row for result in results for row in result.route_stats]
    output_rows = [row for result in results for row in result.output_files]
    source_rows = [row for result in results for row in result.source_map]
    all_problems = [dict(row) for row in problems]
    all_problems.extend(row for result in results for row in result.problems)
    all_problems.sort(
        key=lambda row: (
            str(row.get("category", "")),
            str(row.get("cohort", "")),
            str(row.get("canonical_source_path", "")),
            str(row.get("problem_type", "")),
        )
    )
    write_tsv(
        reports / "counts.tsv",
        sorted(route_rows, key=lambda row: (row["category"], row["cohort"], row["source_path"])),
        [
            "category",
            "cohort",
            "source_path",
            "assembled_path",
            "selector",
            "material_role",
            "input_rows",
            "selected_rows",
            "emitted_rows",
            "empty_rows",
            "duplicate_rows",
            "direct_audit_hash_matches",
            "direct_audit_hash_mismatches",
        ],
    )
    write_tsv(
        reports / "problems.tsv",
        all_problems,
        [
            "severity",
            "phase",
            "problem_type",
            "category",
            "cohort",
            "canonical_source_path",
            "assembled_path",
            "message",
            "expected",
            "actual",
            "traceback",
        ],
    )
    skipped_columns = list(asdict(read_manifest(source_root)[0])) + ["reason"]
    write_tsv(reports / "skipped.tsv", skipped, skipped_columns)
    atomic_write_text(
        reports / "excluded_files.md",
        excluded_files_markdown(source_root, skipped, args.categories),
    )
    write_tsv(
        reports / "source_map.tsv",
        sorted(source_rows, key=lambda row: row["source_key"]),
        ["source_key", "canonical_source_path", "source_file", "actual_path"],
    )
    write_tsv(
        reports / "output_files.tsv",
        sorted(output_rows, key=lambda row: row["path"]),
        ["path", "records", "uncompressed_bytes", "compressed_bytes", "sha256"],
    )
    config = {
        "arguments": to_json_safe(vars(args)),
        "environment": package_versions(),
        "manifest_sha256": sha256_file(manifest),
        "gaps_sha256": sha256_file(gaps) if gaps.is_file() else None,
    }
    atomic_write_json(reports / "run_config.json", config)
    problem_source_groups = {
        (row.get("phase"), row.get("canonical_source_path")) for row in all_problems
    }
    failed_source_groups = {
        row.get("canonical_source_path")
        for row in all_problems
        if row.get("problem_type") == "source_group_processing_failed"
    }
    summary = {
        "status": "completed_with_problems" if all_problems else "completed",
        "groups": len(results),
        "resumed_groups": sum(result.resumed for result in results),
        "failed_source_groups": len(failed_source_groups),
        "problem_source_groups": len(problem_source_groups),
        "problems": len(all_problems),
        "routes": len(route_rows),
        "input_rows": sum(row["input_rows"] for row in route_rows),
        "selected_rows": sum(row["selected_rows"] for row in route_rows),
        "emitted_rows": sum(row["emitted_rows"] for row in route_rows),
        "empty_rows": sum(row["empty_rows"] for row in route_rows),
        "duplicate_rows": sum(row["duplicate_rows"] for row in route_rows),
        "output_files": len(output_rows),
        "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in output_rows),
        "compressed_bytes": sum(row["compressed_bytes"] for row in output_rows),
        "skipped_manifest_entries": len(skipped),
        "categories": {
            category: {
                "selected_rows": sum(
                    row["selected_rows"] for row in route_rows if row["category"] == category
                ),
                "emitted_rows": sum(
                    row["emitted_rows"] for row in route_rows if row["category"] == category
                ),
                "problems": sum(
                    str(row.get("category", "")) == category for row in all_problems
                ),
            }
            for category in args.categories
        },
    }
    atomic_write_json(reports / "summary.json", summary)
    atomic_write_text(
        reports / "index_commands.md",
        index_commands(output_root, args.categories, all_problems),
    )


def validate_output(output_root: Path, categories: Sequence[str]) -> dict[str, Any]:
    required = {
        "text",
        "id",
        "risk_category",
        "cohort",
        "basis",
        "material_role",
        "selector",
        "manifest_source_path",
        "manifest_assembled_path",
        "manifest_source_kind",
        "manifest_size_bytes",
        "manifest_note",
        "manifest_row_number",
        "source_key",
        "source_file",
        "source_record_index",
        "source_metadata",
    }
    report_rows: dict[str, dict[str, str]] = {}
    report_path = output_root / "_reports" / "output_files.tsv"
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            report_rows = {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}

    total_records = 0
    total_files = 0
    seen_output_paths: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="dfm9-id-validation-") as temp_name:
        bucket_dir = Path(temp_name)
        bucket_handles = [(bucket_dir / f"{i:03d}.bin").open("wb") for i in range(256)]
        try:
            for category in categories:
                category_dir = output_root / category
                if not category_dir.is_dir():
                    raise PreparationError(f"Missing category directory: {category_dir}")
                bad_files = [
                    path
                    for path in category_dir.rglob("*")
                    if path.is_file() and not path.name.endswith(".jsonl.zst")
                ]
                if bad_files:
                    raise PreparationError(f"Non-index data file under {category_dir}: {bad_files[0]}")
                for path in sorted(category_dir.rglob("*.jsonl.zst")):
                    total_files += 1
                    seen_output_paths.add(str(path))
                    actual_sha256 = sha256_file(path)
                    raw_reader = path.open("rb")
                    zreader = zstd.ZstdDecompressor().stream_reader(raw_reader)
                    text_reader = io.TextIOWrapper(zreader, encoding="utf-8")
                    file_records = 0
                    try:
                        for line_number, line in enumerate(text_reader, start=1):
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise PreparationError(f"{path}:{line_number}: {exc}") from exc
                            missing = required - set(record)
                            if missing:
                                raise PreparationError(f"{path}:{line_number}: missing {sorted(missing)}")
                            if not isinstance(record["text"], str) or not record["text"].strip():
                                raise PreparationError(f"{path}:{line_number}: empty text")
                            if record["risk_category"] != category:
                                raise PreparationError(
                                    f"{path}:{line_number}: category {record['risk_category']!r}"
                                )
                            if (
                                not isinstance(record["cohort"], str)
                                or not record["cohort"].startswith(f"{category}-")
                                or record["cohort"] not in path.parts
                            ):
                                raise PreparationError(
                                    f"{path}:{line_number}: invalid cohort {record['cohort']!r}"
                                )
                            if not isinstance(record["source_metadata"], dict):
                                raise PreparationError(
                                    f"{path}:{line_number}: source_metadata is not an object"
                                )
                            identifier = record["id"]
                            if not isinstance(identifier, str) or not identifier:
                                raise PreparationError(f"{path}:{line_number}: invalid id")
                            digest = hashlib.blake2b(
                                identifier.encode("utf-8"), digest_size=16
                            ).digest()
                            bucket_handles[digest[0]].write(digest[1:])
                            file_records += 1
                    finally:
                        text_reader.close()
                        raw_reader.close()
                    total_records += file_records
                    expected = report_rows.get(str(path))
                    if expected:
                        if file_records != int(expected["records"]):
                            raise PreparationError(f"Record count differs for {path}")
                        if actual_sha256 != expected["sha256"]:
                            raise PreparationError(f"SHA-256 differs for {path}")
        finally:
            for handle in bucket_handles:
                handle.close()

        if report_rows:
            category_roots = [output_root / category for category in categories]
            expected_output_paths = {
                value
                for value in report_rows
                if any(Path(value).is_relative_to(root) for root in category_roots)
            }
            if seen_output_paths != expected_output_paths:
                missing = sorted(expected_output_paths - seen_output_paths)
                unreported = sorted(seen_output_paths - expected_output_paths)
                raise PreparationError(
                    "Output/report file inventory differs: "
                    f"missing={missing[:1]}, unreported={unreported[:1]}"
                )

        for bucket in sorted(bucket_dir.glob("*.bin")):
            seen: set[bytes] = set()
            with bucket.open("rb") as handle:
                while digest := handle.read(15):
                    if len(digest) != 15:
                        raise PreparationError(f"Corrupt ID validation bucket: {bucket}")
                    if digest in seen:
                        raise PreparationError("Duplicate document ID detected")
                    seen.add(digest)

    problems_path = output_root / "_reports" / "problems.tsv"
    reported_problems = 0
    if problems_path.is_file():
        with problems_path.open("r", encoding="utf-8", newline="") as handle:
            reported_problems = sum(
                row.get("category") in categories
                for row in csv.DictReader(handle, delimiter="\t")
            )
    return {
        "files": total_files,
        "records": total_records,
        "categories": list(categories),
        "reported_problems": reported_problems,
    }


def output_is_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def reset_selected_outputs(
    output_root: Path,
    source_root: Path,
    categories: Sequence[str],
) -> None:
    """Remove only selected category outputs and their completion markers."""
    if output_root == Path(output_root.anchor):
        raise PreparationError(f"Refusing to force-overwrite filesystem root: {output_root}")
    if output_root == source_root or source_root.is_relative_to(output_root):
        raise PreparationError(
            f"Refusing to force-overwrite {output_root} because it contains the source root"
        )

    selected = set(categories)
    for category in selected:
        target = output_root / category
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)

    completed = output_root / "_reports" / "completed"
    if completed.is_dir():
        for marker in completed.glob("*.json"):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                marker_categories = {
                    row.get("category") for row in payload.get("route_stats", [])
                }
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if selected & marker_categories:
                marker.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare manifest-filtered DFM9 data for separate Infini-gram indexes."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--categories", nargs="+", default=list(RISK_CATEGORIES), choices=RISK_CATEGORIES
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-part-mib", type=int, default=512)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace outputs only for the selected categories before processing",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise PreparationError("--workers must be >= 1")
    if args.target_part_mib < 1:
        raise PreparationError("--target-part-mib must be >= 1")
    if args.parquet_batch_size < 1:
        raise PreparationError("--parquet-batch-size must be >= 1")
    if args.dry_run and args.validate_only:
        raise PreparationError("--dry-run and --validate-only are mutually exclusive")
    if args.force and args.resume:
        raise PreparationError("--force and --resume are mutually exclusive")
    args.categories = list(dict.fromkeys(args.categories))


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if args.validate_only:
        result = validate_output(output_root, args.categories)
        print(json.dumps(result, indent=2))
        return

    entries = read_manifest(source_root)
    tasks, skipped, problems = build_tasks(
        source_root, entries, set(args.categories)
    )
    print(
        f"Validated {len(entries):,} manifest entries; "
        f"prepared {len(tasks):,} unique source tasks; skipped {len(skipped):,}; "
        f"reported {len(problems):,} task-build problems."
    )
    if args.dry_run:
        cohort_counts = Counter(
            route.entry.cohort for task in tasks for route in task.routes
        )
        print(
            json.dumps(
                {
                    "cohorts": dict(sorted(cohort_counts.items())),
                    "problems": problems,
                },
                indent=2,
            )
        )
        return

    if output_is_nonempty(output_root):
        if args.force:
            reset_selected_outputs(output_root, source_root, args.categories)
        elif not args.resume:
            raise PreparationError(
                f"Output directory is not empty: {output_root}; "
                "use --resume, --force, or a new path"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_reports" / "completed").mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        (output_root / category).mkdir(parents=True, exist_ok=True)

    config = WorkerConfig(
        output_root=str(output_root),
        target_part_bytes=args.target_part_mib * 1024 * 1024,
        compression_level=args.compression_level,
        parquet_batch_size=args.parquet_batch_size,
        resume=args.resume,
    )
    results: list[GroupResult] = []

    def report_group_failure(task: GroupTask, exc: Exception) -> None:
        details = traceback.format_exc()
        problems.extend(
            problem_rows(
                [route.entry for route in task.routes],
                phase="processing",
                problem_type="source_group_processing_failed",
                message=str(exc),
                canonical_source_path=task.canonical_source_path,
                traceback_text=details,
            )
        )
        print(
            f"WARNING: Source group failed and was omitted: "
            f"{task.canonical_source_path}: {exc}",
            file=sys.stderr,
        )

    if args.workers == 1:
        for task in tqdm(tasks, desc="Source groups"):
            try:
                results.append(process_group(task, config))
            except Exception as exc:
                report_group_failure(task, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_group, task, config): task for task in tasks}
            with tqdm(total=len(futures), desc="Source groups") as progress:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        report_group_failure(task, exc)
                    progress.update(1)

    results.sort(key=lambda result: result.canonical_source_path)
    write_reports(source_root, output_root, args, results, skipped, problems)
    summary = json.loads(
        (output_root / "_reports" / "summary.json").read_text(encoding="utf-8")
    )
    print(json.dumps(summary, indent=2))
    if summary["problems"]:
        print(
            f"WARNING: Preparation completed with {summary['problems']:,} reported "
            f"problem(s). Review {output_root / '_reports' / 'problems.tsv'}.",
            file=sys.stderr,
        )
    print(f"Index commands: {output_root / '_reports' / 'index_commands.md'}")


if __name__ == "__main__":
    try:
        main()
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

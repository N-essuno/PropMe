#!/usr/bin/env python3
"""Extract span-level text from SimpleTrace result JSONL.

SimpleTrace result rows contain text at two nesting levels: ``spans[*].text``
describes the matched part of a generation, while ``spans[*].docs[*].text``
contains retrieved source-document excerpts.  This script writes the text of
the longest span from each generation as one ``{"text": ...}`` JSON object per
output line, ordered from longest to shortest span.  Generations without spans
are skipped.

Example:
    python memorization_experiment/extract_span_texts.py \
        --input memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_results.json \
        --output memorization_experiment/data/dfm9/prefix/dfm9_A_prefix_50_span_texts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the longest spans[*].text value from each generation in "
            "SimpleTrace result JSONL and write one {\"text\": ...} record per "
            "line, longest first."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="SimpleTrace result JSONL (often stored with a .json suffix).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file.",
    )
    return parser


def iter_longest_span_text_records(
    input_path: Path,
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield the longest span length and text record from each generation."""
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: expected a JSON object"
                )

            spans = row.get("spans")
            if not isinstance(spans, list):
                raise ValueError(
                    f"{input_path}:{line_number}: expected 'spans' to be a list"
                )

            validated_spans: list[tuple[int, dict[str, Any]]] = []
            for span_index, span in enumerate(spans):
                if not isinstance(span, dict):
                    raise ValueError(
                        f"{input_path}:{line_number}: span {span_index} is not an object"
                    )
                text = span.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"{input_path}:{line_number}: span {span_index} has no string 'text' field"
                    )
                span_length = span.get("span_length")
                if not isinstance(span_length, int) or isinstance(span_length, bool):
                    raise ValueError(
                        f"{input_path}:{line_number}: span {span_index} has no integer "
                        "'span_length' field"
                    )
                validated_spans.append((span_length, span))

            if validated_spans:
                span_length, longest_span = max(
                    validated_spans,
                    key=lambda item: item[0],
                )
                yield span_length, {"text": longest_span["text"]}


def write_span_text_records(input_path: Path, output_path: Path) -> int:
    """Write records longest-first atomically and return their count."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    records = sorted(
        iter_longest_span_text_records(input_path),
        key=lambda item: item[0],
        reverse=True,
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for _, record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return len(records)


def main() -> None:
    args = build_arg_parser().parse_args()
    count = write_span_text_records(args.input, args.output)
    print(f"Wrote {count} span text records to {args.output}")


if __name__ == "__main__":
    main()

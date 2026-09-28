from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("extract_span_texts.py")
SPEC = importlib.util.spec_from_file_location("extract_span_texts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractSpanTextsTests(unittest.TestCase):
    def test_extracts_one_longest_span_per_generation(self):
        rows = [
            {
                "generation": "generation text is ignored",
                "spans": [
                    {
                        "text": "this is longer in characters",
                        "span_length": 2,
                        "docs": [{"text": "document text is ignored"}],
                    },
                    {"text": "short", "span_length": 3, "docs": []},
                ],
            },
            {"generation": "no matches", "spans": []},
            {
                "generation": "another match",
                "spans": [{"text": "last span", "span_length": 1}],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.json"
            output_path = Path(tmpdir) / "texts.jsonl"
            input_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            count = MODULE.write_span_text_records(input_path, output_path)
            output_rows = [
                json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 2)
        self.assertEqual(
            output_rows,
            [
                {"text": "short"},
                {"text": "last span"},
            ],
        )

    def test_invalid_span_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.json"
            output_path = Path(tmpdir) / "texts.jsonl"
            input_path.write_text(
                json.dumps({"spans": [{"docs": [{"text": "nested only"}]}]}) + "\n",
                encoding="utf-8",
            )
            output_path.write_text("existing output\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "span 0 has no string 'text' field"):
                MODULE.write_span_text_records(input_path, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing output\n")
            self.assertFalse(list(Path(tmpdir).glob("*.tmp")))

    def test_rejects_using_input_as_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps({"spans": []}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be different"):
                MODULE.write_span_text_records(path, path)


if __name__ == "__main__":
    unittest.main()

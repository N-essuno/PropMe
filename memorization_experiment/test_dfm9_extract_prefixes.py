from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dfm9_extract_prefixes.py")
SPEC = importlib.util.spec_from_file_location("dfm9_extract_prefixes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class FakeLowLevelEngine:
    def __init__(self, documents):
        self.documents = documents

    def get_total_doc_cnt(self):
        return len(self.documents)


class FakeEngine:
    def __init__(self, documents):
        self.documents = documents
        self.engine = FakeLowLevelEngine(documents)

    def get_doc_by_ix(self, doc_ix, max_disp_len=None):
        document = dict(self.documents[doc_ix])
        if max_disp_len is not None:
            document["token_ids"] = document["token_ids"][:max_disp_len]
        return document


class Dfm9PrefixTests(unittest.TestCase):
    def test_metadata_parsing_and_domain_precedence(self):
        raw = json.dumps(
            {
                "path": "A-01/part.jsonl.zst",
                "metadata": {
                    "risk_category": "A",
                    "cohort": "A-01",
                    "source_metadata": {"source": "lexdk"},
                },
            }
        )
        metadata = MODULE.parse_metadata(raw)
        self.assertEqual(MODULE.metadata_domain(metadata, "A"), "lexdk")
        self.assertEqual(MODULE.metadata_domain({"cohort": "B-03"}, "B"), "B-03")
        self.assertEqual(MODULE.metadata_domain({}, "D"), "D")

    def test_permuted_indices_are_complete_and_deterministic(self):
        import random

        first = list(MODULE.permuted_indices(31, random.Random(42)))
        second = list(MODULE.permuted_indices(31, random.Random(42)))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(31)))

    def test_sampling_filters_short_docs_and_is_category_stable(self):
        documents = [
            {
                "doc_len": length,
                "token_ids": [ord("a") + index] * length,
                "metadata": json.dumps(
                    {"metadata": {"cohort": "A-01", "source_metadata": {"source": f"s{index}"}}}
                ),
            }
            for index, length in enumerate((2, 10, 12, 15, 20))
        ]
        first = MODULE.sample_documents(
            FakeEngine(documents),
            FakeTokenizer(),
            category="A",
            num_docs=3,
            minimum_tokens=10,
            seed=9,
        )
        second = MODULE.sample_documents(
            FakeEngine(documents),
            FakeTokenizer(),
            category="A",
            num_docs=3,
            minimum_tokens=10,
            seed=9,
        )
        self.assertEqual(first, second)
        self.assertEqual(len({document.doc_ix for document in first}), 3)
        self.assertTrue(all(len(document.token_ids) >= 10 for document in first))

    def test_separate_prefix_files_reuse_the_same_documents(self):
        sampled = [
            MODULE.SampledDocument(
                doc_ix=1,
                token_ids=tuple(ord(character) for character in "abcdefghij"),
                metadata={"source_metadata": {"source": "lexdk"}},
            ),
            MODULE.SampledDocument(
                doc_ix=2,
                token_ids=tuple(ord(character) for character in "klmnopqrst"),
                metadata={"cohort": "A-02"},
            ),
        ]
        with tempfile.TemporaryDirectory() as name:
            output_dir = Path(name)
            paths = MODULE.write_prefix_files(
                sampled,
                FakeTokenizer(),
                category="A",
                prefix_lengths=[2, 4, 6],
                output_dir=output_dir,
            )
            self.assertEqual(
                [path.name for path in paths],
                [
                    "dfm9_A_prefix_2_prompts.jsonl",
                    "dfm9_A_prefix_4_prompts.jsonl",
                    "dfm9_A_prefix_6_prompts.jsonl",
                ],
            )
            rows = [
                [json.loads(line) for line in path.read_text().splitlines()]
                for path in paths
            ]
            self.assertEqual([len(values) for values in rows], [2, 2, 2])
            self.assertEqual([values[0]["text"] for values in rows], ["ab", "abcd", "abcdef"])
            self.assertEqual([values[0]["domain"] for values in rows], ["lexdk"] * 3)
            self.assertEqual([values[1]["domain"] for values in rows], ["A-02"] * 3)
            self.assertFalse(list(output_dir.glob("*.tmp")))

    def test_argument_validation(self):
        args = MODULE.build_arg_parser().parse_args(["--categories", "A", "B"])
        MODULE.validate_args(args)
        self.assertEqual(args.categories, ["A", "B"])
        self.assertEqual(args.prefix_lengths, [50, 75, 100])


if __name__ == "__main__":
    unittest.main()

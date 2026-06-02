from __future__ import annotations

import json
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

simple_trace = importlib.import_module("03_tracing.simple_trace")
evaluate_results = simple_trace.evaluate_results
launch_simpletrace = simple_trace.launch_simpletrace
save_results = simple_trace.save_results
trace_generation = simple_trace.trace_generation


INDEX_DIR = REPO_ROOT / "00_data" / "dummy_index"
UNIGRAM_PROBS_PATH = REPO_ROOT / "02_unigram_probs" / "unigram_probs_dummy.json"
DUMMY_DATASET_PATH = REPO_ROOT / "00_data" / "dummy_dataset" / "dummy.jsonl"


class TestSimpleTraceDummyIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"Missing index dir: {INDEX_DIR}. Build it with the command in 01_indexing/README.md."
            )
        if not UNIGRAM_PROBS_PATH.exists():
            raise FileNotFoundError(
                f"Missing unigram probs: {UNIGRAM_PROBS_PATH}. Build it with 02_unigram_probs/compute_unigrams.py."
            )
        if not DUMMY_DATASET_PATH.exists():
            raise FileNotFoundError(f"Missing dataset file: {DUMMY_DATASET_PATH}")

        with open(DUMMY_DATASET_PATH, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        dataset_by_id = {str(row["id"]): row for row in rows}

        cls.query_begin = "The city archive opened before sunrise, and the restoration team entered quietly"
        cls.query_middle = "reconstructed a reliable timeline of migrations, business closures, and school expansions"
        cls.query_end = "photographed shelf labels to prevent indexing mistakes during the next session."
        cls.query_short = "expanded our understanding of the universe."
        cls.query_cross_docs = (
            "zzqvxx01 zzqvxx02 zzqvxx03 zzqvxx04 zzqvxx05 zzqvxx06 zzqvxx07 zzqvxx08 "
            "zzqvxx09 zzqvxx10 zzqvxx11 zzqvxx12 zzqvxx13 zzqvxx14 zzqvxx15 zzqvxx16. "
            "Space exploration has expanded our understanding of the universe. "
            "zzqvxx17 zzqvxx18 zzqvxx19 zzqvxx20 zzqvxx21 zzqvxx22 zzqvxx23 zzqvxx24 "
            "zzqvxx25 zzqvxx26 zzqvxx27 zzqvxx28 zzqvxx29 zzqvxx30 zzqvxx31 zzqvxx32. "
            "Economic policies can influence employment rates."
        )
        cls.query_none = "zzqvxx_nonexistent_phrase_for_validation_12345"
        cls.query_full_doc43 = dataset_by_id["43"]["text"]

        cls.generations = [
            cls.query_begin,
            cls.query_middle,
            cls.query_end,
            cls.query_short,
            cls.query_cross_docs,
            cls.query_none,
            cls.query_full_doc43,
        ]

        # High docs_per_span reduces random subsampling in tiny test index.
        cls.results = cls._run_simpletrace_for_tests(
            generations=cls.generations,
            docs_per_span=1000,
        )

    @classmethod
    def _run_simpletrace_for_tests(cls, generations: list[str], docs_per_span: int) -> dict:
        """Run SimpleTrace in multiprocessing mode when available, else in-process.

        Some restricted environments (including certain sandboxes/CI) deny
        semaphore syscalls used by ProcessPoolExecutor. In that case we still
        validate SimpleTrace behavior by running trace_generation directly with
        the same components and arguments.
        """
        try:
            return launch_simpletrace(
                index_dir=str(INDEX_DIR),
                generations=generations,
                unigram_probs_path=str(UNIGRAM_PROBS_PATH),
                num_workers=1,
                docs_per_span=docs_per_span,
                enable_print=False,
            )
        except PermissionError:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                add_bos_token=False,
                add_eos_token=False,
            )
            engine = InfiniGramEngine(
                index_dir=str(INDEX_DIR),
                eos_token_id=tokenizer.eos_token_id,
                precompute_unigram_logprobs=False,
            )
            with open(UNIGRAM_PROBS_PATH, "r", encoding="utf-8") as f:
                unigram_probs = {int(k): v["prob"] for k, v in json.load(f).items()}

            results: dict[str, dict] = {}
            for generation in generations:
                results[generation] = trace_generation(
                    generation=generation,
                    engine=engine,
                    enc=tokenizer,
                    unigram_probs=unigram_probs,
                    docs_per_span=docs_per_span,
                )
            return results

    def _doc_ids_for_generation(self, generation: str) -> set[str]:
        result = self.results.get(generation, {})
        doc_ids = set()
        for span in result.get("final_spans", []):
            for doc in span.get("docs", []):
                doc_id = str(doc.get("id", ""))
                if doc_id:
                    doc_ids.add(doc_id)
        return doc_ids

    def _span_texts_for_generation(self, generation: str) -> list[str]:
        result = self.results.get(generation, {})
        return [span.get("text", "") for span in result.get("final_spans", [])]

    def _assert_span_and_doc_present(
        self,
        generation: str,
        expected_span_text: str,
        expected_doc_id: str,
    ) -> None:
        span_texts = self._span_texts_for_generation(generation)
        self.assertIn(
            expected_span_text,
            span_texts,
            msg=(
                f"Expected exact span not found.\n"
                f"Expected: {expected_span_text!r}\n"
                f"Found spans: {span_texts!r}"
            ),
        )
        self.assertIn(expected_doc_id, self._doc_ids_for_generation(generation))

    def _evaluate_single_generation(self, generation: str, nv_recall_threshold: float = 0.0) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            return evaluate_results(
                {generation: self.results[generation]},
                summary_output_path=str(summary_path),
                nv_recall_threshold=nv_recall_threshold,
            )

    def test_beginning_substring_extracts_doc_43(self):
        self._assert_span_and_doc_present(
            generation=self.query_begin,
            expected_span_text=self.query_begin,
            expected_doc_id="43",
        )

    def test_middle_substring_extracts_doc_43(self):
        self._assert_span_and_doc_present(
            generation=self.query_middle,
            expected_span_text=self.query_middle,
            expected_doc_id="43",
        )

    def test_ending_substring_extracts_doc_43(self):
        self._assert_span_and_doc_present(
            generation=self.query_end,
            expected_span_text=self.query_end,
            expected_doc_id="43",
        )

    def test_known_short_phrase_extracts_doc_8(self):
        self._assert_span_and_doc_present(
            generation=self.query_short,
            expected_span_text=self.query_short,
            expected_doc_id="8",
        )

    def test_cross_document_string_extracts_both_docs(self):
        span_texts = self._span_texts_for_generation(self.query_cross_docs)
        self.assertIn("Space exploration has expanded our understanding of the universe.", span_texts)
        self.assertIn("Economic policies can influence employment rates.", span_texts)

        doc_ids = self._doc_ids_for_generation(self.query_cross_docs)
        self.assertIn("8", doc_ids)
        self.assertIn("23", doc_ids)

    def test_full_document_query_retrieves_doc_43(self):
        result = self.results.get(self.query_full_doc43, {})
        self.assertGreaterEqual(len(result.get("final_spans", [])), 1)
        self.assertIn("43", self._doc_ids_for_generation(self.query_full_doc43))

    def test_nonexistent_query_has_no_final_spans(self):
        result = self.results.get(self.query_none, {})
        self.assertEqual(result.get("final_spans", []), [])

    def test_evaluation_full_partial_counts_per_query(self):
        expectations = [
            (self.query_begin, 1, 0, ["43"]),
            (self.query_middle, 1, 0, ["43"]),
            (self.query_end, 1, 0, ["43"]),
            (self.query_short, 1, 0, ["8"]),
            (self.query_cross_docs, 0, 2, []),
            (self.query_none, 0, 0, []),
            (self.query_full_doc43, 1, 0, ["43"]),
        ]

        for generation, expected_full, expected_partial, expected_full_doc_ids in expectations:
            with self.subTest(generation=generation[:80]):
                summary = self._evaluate_single_generation(generation)
                self.assertEqual(summary["full_exact_matches"], expected_full)
                self.assertEqual(summary["partial_matches"], expected_partial)
                self.assertEqual(summary["total_docs"], expected_full + expected_partial)
                self.assertEqual(
                    sorted(str(doc_id) for doc_id in summary["full_exact_match_doc_ids"]),
                    sorted(expected_full_doc_ids),
                )

    def test_evaluation_threshold_tracks_distinct_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary = evaluate_results(
                self.results,
                summary_output_path=str(summary_path),
                nv_recall_threshold=0.5,
            )

            self.assertIn("43", summary["doc_ids_above_nv_recall_threshold"])
            self.assertEqual(
                summary["docs_above_nv_recall_threshold"],
                len(summary["doc_ids_above_nv_recall_threshold"]),
            )

    def test_evaluation_additional_generation_and_span_metrics(self):
        synthetic_results = {
            "g1": {
                "final_spans": [
                    {
                        "start": 0,
                        "end": 10,
                        "text": "s1",
                        "docs": [
                            {"id": "d1", "text": "prefix g1 suffix", "nv_recall": 1.0, "nv_matched_words": 1},
                            {"id": "d2", "text": "partial hit only", "nv_recall": 0.0, "nv_matched_words": 0},
                        ],
                    },
                    {
                        "start": 10,
                        "end": 70,  # len = 60
                        "text": "s2",
                        "docs": [
                            {"id": "d2", "text": "partial hit only", "nv_recall": 0.0, "nv_matched_words": 0},
                            {"id": "d3", "text": "another partial", "nv_recall": 0.0, "nv_matched_words": 0},
                        ],
                    },
                ]
            },
            "g2": {
                "final_spans": [
                    {
                        "start": 0,
                        "end": 59,
                        "text": "s3",
                        "docs": [
                            {"id": "d4", "text": "this contains g2", "nv_recall": 0.5, "nv_matched_words": 1},
                        ],
                    }
                ]
            },
            "g3": {"final_spans": []},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary = evaluate_results(synthetic_results, summary_output_path=str(summary_path))

        self.assertEqual(summary["unique_partial_matches"], 2)  # d2, d3
        self.assertAlmostEqual(summary["average_span_length"], (60 + 59 + 0) / 3)
        self.assertEqual(summary["min_span_length"], 10)
        self.assertEqual(summary["max_span"], 60)
        self.assertEqual(summary["unique_total_docs"], 4)  # d1, d2, d3, d4
        self.assertEqual(summary["unique_full_exact_matches"], 2)  # d1, d4
        self.assertAlmostEqual(summary["generations_with_60_token_span_ratio"], 1 / 3)
        self.assertAlmostEqual(summary["generations_full_matches_ratio"], 2 / 3)
        self.assertEqual(
            summary["spans_length_counts_distribution"],
            {"(1, 3)": 0, "(4, 6)": 0, "(7, 10)": 2, "(11, 20)": 0, "(21, inf)": 3},
        )
        self.assertEqual(
            summary["spans_length_distribution"],
            {"(1, 3)": 0.0, "(4, 6)": 0.0, "(7, 10)": 0.4, "(11, 20)": 0.0, "(21, inf)": 0.6},
        )

    def test_saved_results_include_span_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.jsonl"
            save_results(self.results, str(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            for row in rows:
                for span in row.get("spans", []):
                    self.assertIn("span_length", span)
                    self.assertEqual(span["span_length"], span["end"] - span["start"])

    def test_saved_results_spans_are_sorted_by_length_desc(self):
        results = {
            "dummy generation": {
                "final_spans": [
                    {"start": 1, "end": 3, "text": "b", "docs": []},   # len 2
                    {"start": 2, "end": 8, "text": "c", "docs": []},   # len 6
                    {"start": 0, "end": 1, "text": "a", "docs": []},   # len 1
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.jsonl"
            save_results(results, str(output_path))
            with open(output_path, "r", encoding="utf-8") as f:
                row = json.loads(next(line for line in f if line.strip()))

        lengths = [span["span_length"] for span in row["spans"]]
        self.assertEqual(lengths, [6, 2, 1])


if __name__ == "__main__":
    unittest.main()

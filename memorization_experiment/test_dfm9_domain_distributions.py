import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("dfm9_domain_distributions.py")
SPEC = importlib.util.spec_from_file_location("dfm9_domain_distributions", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class FakeEngine:
    def find(self, input_ids):
        return {"cnt": 2, "segment_by_shard": [(10, 12)]}

    def get_doc_by_rank(self, *, s, rank, max_disp_len):
        doc_id = "A:A-01:wrong:0:doc" if rank == 10 else "A:A-01:key:1:doc"
        return {
            "doc_ix": rank + 100,
            "metadata": json.dumps(
                {
                    "metadata": {
                        "id": doc_id,
                        "risk_category": "A",
                        "cohort": "A-01",
                        "source_metadata": {"source": "books"},
                    }
                }
            ),
        }


class FakeBoundaryEngine:
    def find(self, input_ids):
        if len(input_ids) > 32:
            return {"cnt": 0, "segment_by_shard": [(0, 0)]}
        return {"cnt": 1, "segment_by_shard": [(7, 8)]}

    def get_doc_by_rank(self, *, s, rank, max_disp_len):
        return {
            "doc_ix": 123,
            "metadata": json.dumps(
                {
                    "metadata": {
                        "id": "B:B-01:key:2:doc",
                        "source_metadata": {"dataset": "fallback-domain"},
                    }
                }
            ),
        }


class DomainDistributionTests(unittest.TestCase):
    def test_length_buckets_match_memorization_defaults(self):
        buckets = MODULE.parse_length_buckets(MODULE.DEFAULT_LENGTH_BUCKETS)
        self.assertEqual(
            [bucket.label for bucket in buckets],
            [
                "(1, 3)",
                "(4, 6)",
                "(7, 10)",
                "(11, 20)",
                "(21, 50)",
                "(51, 100)",
                "(101, 150)",
                "(151, inf)",
            ],
        )
        self.assertEqual(MODULE.bucket_for_length(6, buckets), "(4, 6)")
        self.assertEqual(MODULE.bucket_for_length(151, buckets), "(151, inf)")

    def test_domain_precedence(self):
        metadata = {
            "domain": "outer",
            "cohort": "A-01",
            "source_metadata": {
                "domain": "inner-domain",
                "source": "inner-source",
            },
        }
        self.assertEqual(
            MODULE.metadata_domain(metadata, "A", MODULE.DEFAULT_DOMAIN_FIELDS),
            ("inner-domain", "source_metadata.domain"),
        )
        self.assertEqual(
            MODULE.metadata_domain({"cohort": "B-04"}, "B", MODULE.DEFAULT_DOMAIN_FIELDS),
            ("B-04", "cohort"),
        )

    def test_id_lookup_verifies_candidate_metadata(self):
        request = MODULE.DocumentRequest(
            doc_id="A:A-01:key:1:doc",
            category="A",
            excerpts=["distinct excerpt"],
        )
        resolved, reason = MODULE.lookup_document(
            request,
            FakeEngine(),
            FakeTokenizer(),
            MODULE.DEFAULT_DOMAIN_FIELDS,
            max_candidate_ranks=10,
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.doc_id, request.doc_id)
        self.assertEqual(resolved.domain, "books")
        self.assertEqual(resolved.shard, 0)
        self.assertEqual(resolved.doc_ix, 111)

    def test_short_lookup_queries_trim_boundaries(self):
        queries = list(MODULE.lookup_query_variants([1, 2, 3, 4, 5]))
        self.assertEqual(queries[0], [1, 2, 3, 4, 5])
        self.assertIn([2, 3, 4, 5], queries)
        self.assertIn([1, 2, 3, 4], queries)
        self.assertIn([2, 3, 4], queries)

    def test_id_lookup_uses_interior_window_after_boundary_mismatch(self):
        request = MODULE.DocumentRequest(
            doc_id="B:B-01:key:2:doc",
            category="B",
            excerpts=["x" * 200],
        )
        resolved, reason = MODULE.lookup_document(
            request,
            FakeBoundaryEngine(),
            FakeTokenizer(),
            MODULE.DEFAULT_DOMAIN_FIELDS,
            max_candidate_ranks=10,
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.domain, "fallback-domain")
        self.assertEqual(resolved.doc_ix, 123)

    def test_prepared_fallback_resolves_exact_id(self):
        import zstandard as zstd

        doc_id = "A:A-01:sourcekey:000000001:doc"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "A" / "A-01" / "group" / "part-00000.jsonl.zst"
            output_path.parent.mkdir(parents=True)
            with zstd.open(output_path, mode="wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "text": "indexed text",
                            "id": doc_id,
                            "risk_category": "A",
                            "cohort": "A-01",
                            "source_key": "sourcekey",
                            "source_metadata": {"source": "exact-domain"},
                        }
                    )
                    + "\n"
                )
            completed = root / "_reports" / "completed"
            completed.mkdir(parents=True)
            (completed / "group.json").write_text(
                json.dumps(
                    {
                        "source_map": [{"source_key": "sourcekey"}],
                        "output_files": [{"path": str(output_path)}],
                    }
                )
            )
            requests = {
                doc_id: MODULE.DocumentRequest(doc_id=doc_id, category="A")
            }
            resolved, failures = MODULE.resolve_from_prepared_data(
                requests,
                {doc_id: "ambiguous index window"},
                root,
                MODULE.DEFAULT_DOMAIN_FIELDS,
            )
        self.assertFalse(failures)
        self.assertEqual(resolved[doc_id].domain, "exact-domain")
        self.assertEqual(resolved[doc_id].resolution_method, "prepared_data_exact_id")
        self.assertNotIn("text", resolved[doc_id].metadata)

    def test_observation_report_counts_occurrences_and_unique_ids(self):
        payload = [
            {
                "generation": "g1",
                "spans": [
                    {
                        "start": 0,
                        "end": 3,
                        "span_length": 3,
                        "docs": [
                            {
                                "id": "A:A-01:key:1:doc",
                                "text": "alpha excerpt",
                                "nv_recall": 0.5,
                            },
                            {
                                "id": "B:B-01:key:2:doc",
                                "text": "beta excerpt",
                                "nv_recall": 0.0,
                            },
                        ],
                    },
                    {
                        "start": 0,
                        "end": 8,
                        "span_length": 8,
                        "docs": [
                            {
                                "id": "A:A-01:key:1:doc",
                                "text": "longer alpha excerpt",
                                "nv_recall": 0.75,
                            }
                        ],
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "results.json"
            result_path.write_text("\n".join(json.dumps(row) for row in payload) + "\n")
            requests = {}
            buckets = MODULE.parse_length_buckets(MODULE.DEFAULT_LENGTH_BUCKETS)
            observations = MODULE.collect_observations(
                result_path,
                buckets,
                0.5,
                requests,
            )
            resolved = {
                "A:A-01:key:1:doc": MODULE.ResolvedDocument(
                    doc_id="A:A-01:key:1:doc",
                    category="A",
                    domain="alpha",
                    domain_field="source_metadata.source",
                    shard=0,
                    doc_ix=1,
                    metadata={},
                ),
                "B:B-01:key:2:doc": MODULE.ResolvedDocument(
                    doc_id="B:B-01:key:2:doc",
                    category="B",
                    domain="beta",
                    domain_field="source_metadata.source",
                    shard=0,
                    doc_ix=2,
                    metadata={},
                ),
            }
            report = MODULE.build_report(
                observations,
                resolved,
                buckets,
                0.5,
                MODULE.DEFAULT_DOMAIN_FIELDS,
                {"A": Path("A"), "B": Path("B")},
                Path("document_map.jsonl"),
            )

        self.assertEqual(report["input_counts"]["retrieved_document_occurrences"], 3)
        self.assertEqual(report["input_counts"]["unique_retrieved_document_ids"], 2)
        self.assertEqual(report["input_counts"]["nv_recall_hit_document_occurrences"], 2)
        short = report["span_bucket_domain_distributions"]["(1, 3)"]
        self.assertEqual(short["total_document_occurrences"], 2)
        self.assertEqual(short["total_unique_documents"], 2)
        hits = report["nv_recall_domain_distribution"]
        self.assertEqual(hits["total_document_occurrences"], 2)
        self.assertEqual(hits["total_unique_documents"], 1)
        self.assertEqual(hits["document_occurrences_by_domain"][0]["domain"], "alpha")

    def test_plots_are_rendered(self):
        buckets = MODULE.parse_length_buckets(MODULE.DEFAULT_LENGTH_BUCKETS)
        observations = MODULE.ResultObservations(path=Path("synthetic.json"))
        observations.generation_count = 1
        observations.span_count = 1
        observations.document_occurrence_count = 1
        observations.span_counts_by_bucket["(4, 6)"] = 1
        observations.spans_with_documents_by_bucket["(4, 6)"] = 1
        observations.spans_with_nv_hit = 1
        observations.spans_with_nv_hit_by_bucket["(4, 6)"] = 1
        observations.documents_by_bucket["(4, 6)"]["A:A-01:key:1:doc"] = 1
        observations.nv_hit_documents["A:A-01:key:1:doc"] = 1
        observations.nv_hit_documents_by_bucket["(4, 6)"]["A:A-01:key:1:doc"] = 1
        resolved = {
            "A:A-01:key:1:doc": MODULE.ResolvedDocument(
                doc_id="A:A-01:key:1:doc",
                category="A",
                domain="alpha",
                domain_field="cohort",
                shard=0,
                doc_ix=1,
                metadata={},
            )
        }
        report = MODULE.build_report(
            observations,
            resolved,
            buckets,
            0.5,
            MODULE.DEFAULT_DOMAIN_FIELDS,
            {"A": Path("A")},
            Path("document_map.jsonl"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = MODULE.render_report_plots(report, Path(tmpdir), "test", 5)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()

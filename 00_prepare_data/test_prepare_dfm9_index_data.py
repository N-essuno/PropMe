from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import io
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd


MODULE_PATH = Path(__file__).with_name("prepare_dfm9_index_data.py")
SPEC = importlib.util.spec_from_file_location("prepare_dfm9_index_data", MODULE_PATH)
PREP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREP
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


def entry(**overrides):
    values = {
        "row_number": 2,
        "category": "A",
        "cohort": "A-01",
        "basis": "agreement",
        "material_role": "dataset_source",
        "selector": "all records",
        "source_path": "/canonical/source",
        "assembled_path": "A/A-01/data.jsonl",
        "source_kind": "file",
        "size_bytes": "10",
        "note": "fixture",
    }
    values.update(overrides)
    return PREP.ManifestEntry(**values)


def read_zstd_jsonl(path: Path):
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            return [json.loads(line) for line in reader.read().decode().splitlines()]


class StreamingInputTests(unittest.TestCase):
    def test_jsonl_gzip_zstd_parquet_and_json_array(self):
        records = [{"text": "one", "n": 1}, {"text": "two", "n": 2}]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            jsonl = root / "data.jsonl"
            jsonl.write_text("".join(json.dumps(row) + "\n" for row in records))
            gz = root / "data.jsonl.gz"
            with gzip.open(gz, "wt", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            zst = root / "data.jsonl.zst"
            with zst.open("wb") as raw:
                with zstd.ZstdCompressor().stream_writer(raw) as writer:
                    writer.write(
                        "".join(json.dumps(row) + "\n" for row in records).encode()
                    )
            parquet = root / "data.parquet"
            pq.write_table(pa.Table.from_pylist(records), parquet)
            array_dir = root / "D-10"
            array_dir.mkdir()
            array = array_dir / "data.json"
            array.write_text(json.dumps(records, indent=2))

            for path in (jsonl, gz, zst, parquet):
                self.assertEqual(list(PREP.iter_records(path, 1)), records)
            self.assertEqual(list(PREP.iter_json_array(array, chunk_size=3)), records)
            self.assertEqual(list(PREP.iter_records(array, 1)), records)

    def test_malformed_inputs_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bad_jsonl = root / "bad.jsonl"
            bad_jsonl.write_text('{"text": 1}\nnot-json\n')
            with self.assertRaises(PREP.PreparationError):
                list(PREP.iter_jsonl(bad_jsonl))
            array_dir = root / "D-10"
            array_dir.mkdir()
            bad_array = array_dir / "data.json"
            bad_array.write_text('[{"text": 1}')
            with self.assertRaises(PREP.PreparationError):
                list(PREP.iter_json_array(bad_array, chunk_size=2))


class SerializationTests(unittest.TestCase):
    def test_text_is_exact_and_metadata_is_non_content(self):
        source = {"text": "  exact\nspace  ", "messages": [{"role": "user", "content": "x"}], "tag": 7}
        document = PREP.serialize_record(source)[0]
        self.assertEqual(document.text, source["text"])
        metadata = PREP.source_metadata(source, document)
        self.assertEqual(metadata["tag"], 7)
        self.assertNotIn("text", metadata)
        self.assertNotIn("messages", metadata)
        self.assertEqual(metadata["message_structure"], [{"role": "user"}])

    def test_empty_text_is_recognized_for_drop_accounting(self):
        document = PREP.serialize_record({"text": "", "source": "lexdk"})[0]
        self.assertEqual(document.text, "")
        fallback = PREP.serialize_record(
            {"text": "", "instruction": "Do", "response": "Done"}
        )[0]
        self.assertEqual(fallback.text, "Do\n\nDone")
        self.assertIn("text", fallback.consumed_fields)

    def test_pairs_messages_tools_and_math(self):
        self.assertEqual(
            PREP.serialize_record({"instruction": "Do", "response": "Done"})[0].text,
            "Do\n\nDone",
        )
        self.assertEqual(
            PREP.serialize_record({"problem": "p", "generated_solution": "s"})[0].text,
            "p\n\ns",
        )
        tool_record = {
            "system": "system",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer", "tool_calls": [{"name": "f"}]},
            ],
            "tools": [{"name": "f", "description": "tool"}],
            "source": "fixture",
        }
        document = PREP.serialize_record(tool_record)[0]
        self.assertIn("system", document.text)
        self.assertIn('"description":"tool"', document.text)
        self.assertIn('"name":"f"', document.text)
        metadata = PREP.source_metadata(tool_record, document)
        self.assertEqual(metadata["source"], "fixture")
        self.assertEqual(
            metadata["message_structure"],
            [{"role": "user"}, {"role": "assistant"}],
        )

    def test_tool_dataset_variants(self):
        glaive = PREP.serialize_record({"system": "tools", "chat": "USER: hi"})[0]
        self.assertEqual(glaive.text, "tools\n\nUSER: hi")
        xlam = PREP.serialize_record(
            {"query": "q", "answers": [{"name": "f"}], "tools": [{"name": "f"}]}
        )[0]
        self.assertEqual(xlam.text, '[{"name":"f"}]\n\nq\n\n{"name":"f"}')
        self.assertIn('"name":"f"', xlam.text)
        self.assertIn("q", xlam.text)

    def test_arena_emits_two_documents_and_d01_only_prompts(self):
        arena = {
            "system_prompt_a": "sa",
            "system_prompt_b": "sb",
            "opening_msg": "open",
            "conversation_a": [{"role": "user", "content": "qa"}],
            "conversation_b": [{"role": "assistant", "content": "ab"}],
            "winner": "a",
        }
        documents = PREP.serialize_record(arena)
        self.assertEqual([document.variant for document in documents], ["a", "b"])
        self.assertEqual(documents[0].text, "sa\n\nopen\n\nqa")
        self.assertEqual(documents[1].text, "sb\n\nopen\n\nab")
        prompts = PREP.d01_documents(
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                    {"from": "human", "value": "h"},
                ]
            }
        )
        self.assertEqual([document.text for document in prompts], ["u", "h"])

    def test_unknown_schema_and_json_safety(self):
        with self.assertRaises(PREP.PreparationError):
            PREP.serialize_record({"metadata_only": 1})
        value = {
            "nan": math.nan,
            "when": dt.datetime(2026, 1, 2, 3, 4, 5),
            "bytes": b"abc",
            "set": {"b", "a"},
        }
        first = PREP.canonical_json(value)
        self.assertEqual(first, PREP.canonical_json(value))
        safe = json.loads(first)
        self.assertIsNone(safe["nan"])
        self.assertEqual(safe["set"], ["a", "b"])


class SelectorAndInventoryTests(unittest.TestCase):
    def test_special_selectors_and_expected_counts(self):
        self.assertEqual(PREP.EXPECTED_SELECTED_ROWS["A-06"], 1_104)
        self.assertEqual(PREP.EXPECTED_SELECTED_ROWS["B-03"], 5_169)
        self.assertEqual(PREP.EXPECTED_SELECTED_ROWS["B-04"], 335_117)
        self.assertEqual(PREP.EXPECTED_B03_DOCUMENTS, 2_607)
        expected = {
            "C-09": 89_982,
            "C-10": 35_357,
            "C-11": 35_380,
            "C-12": 2_688,
            "C-13": 42_026,
            "C-14": 22_280,
        }
        self.assertEqual({key: PREP.EXPECTED_SELECTED_ROWS[key] for key in expected}, expected)

        routes = [
            (PREP.RouteSpec(entry(cohort="A-06"), "a06"), {"subset": "odense"}, True),
            (PREP.RouteSpec(entry(cohort="B-03"), "b03"), {"dataset": "instruction-generation"}, True),
            (PREP.RouteSpec(entry(cohort="B-04"), "b04", allowed_values=("safe",)), {"dataset": "unsafe"}, False),
            (PREP.RouteSpec(entry(cohort="C-09"), "c09"), {"source": "ai2-adapt-dev/flan_v2_converted"}, True),
            (PREP.RouteSpec(entry(cohort="C-10"), "c10"), {"dataset": "science.bio"}, True),
            (PREP.RouteSpec(entry(cohort="C-11"), "openhermes", allowed_values=("airoboros2.2",)), {"openhermes_source": "airoboros2.2"}, True),
        ]
        for route, record, expected_match in routes:
            self.assertEqual(PREP.route_matches(route, record), expected_match)

    def test_unknown_selector_is_fatal_and_d05_is_skipped(self):
        with self.assertRaises(PREP.PreparationError):
            PREP.route_for_entry(entry(selector="new selector"), Path("/unused"))
        self.assertIsNone(PREP.route_for_entry(entry(category="D", cohort="D-05"), Path("/unused")))

    def test_inventory_excludes_cache_and_json_manifests(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / ".cache").mkdir()
            (root / "metadata").mkdir()
            (root / "data.jsonl").write_text('{"text":"ok"}\n')
            (root / ".cache" / "hidden.jsonl").write_text('{"text":"no"}\n')
            (root / "metadata" / "manifest.json").write_text("{}")
            inventory = PREP.artifact_inventory(root, "directory")
            self.assertEqual([row[0] for row in inventory], ["data.jsonl"])

            legacy = root / "D-10"
            legacy.mkdir()
            (legacy / "data.json").write_text("[]")
            self.assertTrue(PREP.is_supported_data_path(legacy / "data.json"))
            self.assertFalse(PREP.is_supported_data_path(root / "data.json"))

    def test_shared_source_is_streamed_as_one_task(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for assembled in ("B/B-03/shared", "B/B-04/shared"):
                target = root / assembled
                target.mkdir(parents=True)
                (target / "part.jsonl").write_text('{"text":"x"}\n')
            entries = [
                entry(category="B", cohort="B-03", selector="source-retaining rows; deduplicate the 2,607 embedded document hashes", source_path="/same", assembled_path="B/B-03/shared", source_kind="directory"),
                entry(category="B", cohort="B-04", selector="seed-derived rows without embedded document", source_path="/same", assembled_path="B/B-04/shared", source_kind="directory"),
            ]
            with mock.patch.object(PREP, "load_b03_audit", return_value=()), mock.patch.object(PREP, "load_b04_datasets", return_value=("safe",)):
                tasks, skipped, problems = PREP.build_tasks(root, entries, {"B"})
            self.assertEqual(len(tasks), 1)
            self.assertEqual({route.entry.cohort for route in tasks[0].routes}, {"B-03", "B-04"})
            self.assertEqual(skipped, [])
            self.assertEqual(problems, [])

    def test_missing_artifact_is_reported_while_other_tasks_are_built(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            available = root / "A/A-02/data.jsonl"
            available.parent.mkdir(parents=True)
            available.write_text('{"text":"available"}\n')
            entries = [
                entry(
                    cohort="A-01",
                    source_path="/missing",
                    assembled_path="A/A-01/missing.jsonl",
                ),
                entry(
                    cohort="A-02",
                    source_path="/available",
                    assembled_path="A/A-02/data.jsonl",
                ),
            ]

            tasks, skipped, problems = PREP.build_tasks(root, entries, {"A"})

            self.assertEqual([task.canonical_source_path for task in tasks], ["/available"])
            self.assertEqual(skipped, [])
            self.assertEqual(len(problems), 1)
            self.assertEqual(problems[0]["cohort"], "A-01")
            self.assertEqual(problems[0]["problem_type"], "missing_manifest_artifact")
            self.assertIn("does not exist", problems[0]["message"])


    def test_excluded_files_markdown_is_grouped_by_category(self):
        rows = [
            {
                **PREP.asdict(entry(category="B", cohort="B-01", assembled_path="B/audit.csv")),
                "reason": "audit_evidence",
            },
            {
                **PREP.asdict(entry(category="D", cohort="D-05", assembled_path="D/proxy.parquet")),
                "reason": "unresolved_selector",
            },
        ]
        report = PREP.excluded_files_markdown(Path("/missing"), rows, ["A", "B", "C", "D"])
        self.assertIn("## A", report)
        self.assertIn("No manifest data files are fully excluded.", report)
        self.assertIn("### B-01", report)
        self.assertIn("`B/audit.csv (missing)`", report)
        self.assertIn("### D-05", report)
        self.assertIn("`D/proxy.parquet (missing)`", report)



class OutputAndRecoveryTests(unittest.TestCase):
    def test_rollover_roundtrip_atomic_files_and_infini_gram_loader(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            writer = PREP.RollingZstdWriter(root, target_uncompressed_bytes=80, compression_level=3)
            for index in range(6):
                writer.write({"text": f"document-{index}", "id": str(index)})
            stats = writer.close()
            self.assertGreater(len(stats), 1)
            self.assertFalse(list(root.glob("*.tmp")))
            rows = [row for item in stats for row in read_zstd_jsonl(Path(item.path))]
            self.assertEqual([row["id"] for row in rows], [str(i) for i in range(6)])
            from infini_gram.indexing import load_file

            self.assertTrue(load_file(stats[0].path))

    def test_b03_audit_dedup_and_resume_integrity(self):
        records = [
            {"dataset": "instruction-generation", "conversations": [{"value": "<document> alpha </document>"}], "meta": 1},
            {"dataset": "instruction-generation", "conversations": [{"value": "<document> alpha </document>"}], "meta": 2},
            {"dataset": "instruction-generation-ifeval", "conversations": [{"value": "<document> beta </document>"}], "meta": 3},
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in records))
            route = PREP.RouteSpec(
                entry=entry(category="B", cohort="B-03", basis="article_3", selector="source-retaining rows; deduplicate the 2,607 embedded document hashes"),
                mode="b03",
                expected_selected_rows=3,
                audit_hash_rows=(
                    (hashlib.sha256(b"alpha").hexdigest(), 5, 2),
                    (hashlib.sha256(b"beta").hexdigest(), 4, 1),
                ),
            )
            task = PREP.GroupTask(
                group_key="group",
                canonical_source_path="/canonical",
                files=(PREP.SourceFile(str(source), source.name, "sourcekey"),),
                routes=(route,),
            )
            config = PREP.WorkerConfig(str(root / "output"), 1024, 3, 1, False)
            (root / "output" / "_reports" / "completed").mkdir(parents=True)
            with mock.patch.object(PREP, "EXPECTED_B03_DOCUMENTS", 2):
                result = PREP.process_group(task, config)
            stats = result.route_stats[0]
            self.assertEqual(stats["selected_rows"], 3)
            self.assertEqual(stats["emitted_rows"], 2)
            self.assertEqual(stats["duplicate_rows"], 1)
            output_rows = [row for item in result.output_files for row in read_zstd_jsonl(Path(item["path"]))]
            ids = [row["id"] for row in output_rows]
            self.assertEqual({row["text"] for row in output_rows}, {" alpha ", " beta "})
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(identifier.startswith("B:B-03:sourcekey:") for identifier in ids))

            resumed_config = PREP.WorkerConfig(str(root / "output"), 1024, 3, 1, True)
            resumed = PREP.process_group(task, resumed_config)
            self.assertTrue(resumed.resumed)
            Path(result.output_files[0]["path"]).write_bytes(b"corrupt")
            with mock.patch.object(PREP, "EXPECTED_B03_DOCUMENTS", 2):
                regenerated = PREP.process_group(task, resumed_config)
            self.assertFalse(regenerated.resumed)
            self.assertEqual(
                [problem["problem_type"] for problem in regenerated.problems],
                ["invalid_resume_marker"],
            )
            self.assertEqual(
                {row["text"] for item in regenerated.output_files for row in read_zstd_jsonl(Path(item["path"]))},
                {" alpha ", " beta "},
            )

    def test_selected_count_mismatch_finishes_and_is_reported(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source.jsonl"
            source.write_text('{"text":"one"}\n')
            route = PREP.RouteSpec(
                entry=entry(category="C", cohort="C-11"),
                mode="all",
                expected_selected_rows=2,
            )
            task = PREP.GroupTask(
                group_key="group",
                canonical_source_path="/canonical",
                files=(PREP.SourceFile(str(source), source.name, "sourcekey"),),
                routes=(route,),
            )
            output_root = root / "output"
            (output_root / "_reports" / "completed").mkdir(parents=True)
            config = PREP.WorkerConfig(str(output_root), 1024, 3, 1, False)

            result = PREP.process_group(task, config)

            self.assertEqual(result.route_stats[0]["selected_rows"], 1)
            self.assertEqual(len(result.output_files), 1)
            self.assertEqual(len(result.problems), 1)
            problem = result.problems[0]
            self.assertEqual(problem["problem_type"], "selected_row_count_mismatch")
            self.assertEqual(problem["expected"], 2)
            self.assertEqual(problem["actual"], 1)

    def test_reports_include_problems_summary_and_index_warning(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source_root = root / "source"
            output_root = root / "output"
            source_root.mkdir()
            columns = [
                "category", "cohort", "basis", "material_role", "selector",
                "source_path", "assembled_path", "source_kind", "size_bytes", "note",
            ]
            row = entry(category="C", cohort="C-11")
            with (source_root / "manifest.tsv").open("w", newline="") as handle:
                writer = PREP.csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
                writer.writeheader()
                writer.writerow({column: getattr(row, column) for column in columns})
            problem = PREP.problem_rows(
                [row],
                phase="validation",
                problem_type="selected_row_count_mismatch",
                message="expected 2, found 1",
                expected=2,
                actual=1,
            )[0]
            args = argparse.Namespace(categories=["C"], source_root=source_root, output_root=output_root)

            result = PREP.GroupResult(
                group_key="group",
                canonical_source_path="/canonical",
                problems=[problem],
            )
            PREP.write_reports(source_root, output_root, args, [result], [], [])

            with (output_root / "_reports/problems.tsv").open(newline="") as handle:
                rows = list(PREP.csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["problem_type"], "selected_row_count_mismatch")
            summary = json.loads((output_root / "_reports/summary.json").read_text())
            self.assertEqual(summary["status"], "completed_with_problems")
            self.assertEqual(summary["problems"], 1)
            commands = (output_root / "_reports/index_commands.md").read_text()
            self.assertIn("Review `problems.tsv`", commands)
            self.assertIn("Category C", commands)

    def test_main_isolates_bad_group_and_finishes_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source_root = root / "source"
            output_root = root / "output"
            output_root.mkdir()
            stale_output = output_root / "stale.txt"
            stale_output.write_text("unrelated root file")
            stale_selected = output_root / "A/old.txt"
            stale_unselected = output_root / "B/keep.txt"
            stale_selected.parent.mkdir()
            stale_unselected.parent.mkdir()
            stale_selected.write_text("old A output")
            stale_unselected.write_text("existing B output")
            good = source_root / "A/A-01/good.jsonl"
            bad = source_root / "A/A-02/bad.jsonl"
            good.parent.mkdir(parents=True)
            bad.parent.mkdir(parents=True)
            good.write_text('{"text":"good"}\n')
            bad.write_text('not-json\n')
            columns = [
                "category", "cohort", "basis", "material_role", "selector",
                "source_path", "assembled_path", "source_kind", "size_bytes", "note",
            ]
            rows = [
                entry(cohort="A-01", source_path="/good", assembled_path="A/A-01/good.jsonl"),
                entry(cohort="A-02", source_path="/bad", assembled_path="A/A-02/bad.jsonl"),
            ]
            with (source_root / "manifest.tsv").open("w", newline="") as handle:
                writer = PREP.csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: getattr(row, column) for column in columns})
            argv = [
                str(MODULE_PATH),
                "--source-root", str(source_root),
                "--output-root", str(output_root),
                "--categories", "A",
                "--workers", "2",
                "--target-part-mib", "1",
                "--force",
            ]

            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                PREP.main()

            self.assertTrue(stale_output.exists())
            self.assertFalse(stale_selected.exists())
            self.assertTrue(stale_unselected.exists())
            summary = json.loads((output_root / "_reports/summary.json").read_text())
            self.assertEqual(summary["status"], "completed_with_problems")
            self.assertEqual(summary["groups"], 1)
            self.assertEqual(summary["failed_source_groups"], 1)
            with (output_root / "_reports/problems.tsv").open(newline="") as handle:
                problems = list(PREP.csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(problems[0]["cohort"], "A-02")
            self.assertEqual(problems[0]["problem_type"], "source_group_processing_failed")
            self.assertTrue(list((output_root / "A/A-01").rglob("*.jsonl.zst")))
            self.assertFalse((output_root / "A/A-02").exists())
            validated = PREP.validate_output(output_root, ["A"])
            self.assertEqual(validated["records"], 1)
            self.assertEqual(validated["reported_problems"], 1)

    def test_validate_detects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            category = root / "A" / "A-01" / "group"
            record = {
                "text": "x",
                "id": "duplicate",
                "risk_category": "A",
                "cohort": "A-01",
                "basis": "agreement",
                "material_role": "dataset_source",
                "selector": "all records",
                "manifest_source_path": "/source",
                "manifest_assembled_path": "A/A-01/source",
                "manifest_source_kind": "file",
                "manifest_size_bytes": "1",
                "manifest_note": "",
                "manifest_row_number": 2,
                "source_key": "key",
                "source_file": "source.jsonl",
                "source_record_index": 0,
                "source_metadata": {},
            }
            writer = PREP.RollingZstdWriter(category, target_uncompressed_bytes=1024, compression_level=3)
            writer.write(record)
            writer.write(record)
            writer.close()
            with self.assertRaisesRegex(PREP.PreparationError, "Duplicate document ID"):
                PREP.validate_output(root, ["A"])

    def test_nonempty_output_and_cli_validation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertFalse(PREP.output_is_nonempty(root))
            (root / "existing").write_text("x")
            self.assertTrue(PREP.output_is_nonempty(root))
        args = argparse.Namespace(
            workers=0,
            target_part_mib=1,
            parquet_batch_size=1,
            dry_run=False,
            validate_only=False,
            force=False,
            resume=False,
            categories=["A"],
        )
        with self.assertRaises(PREP.PreparationError):
            PREP.validate_args(args)

        args.workers = 1
        args.force = True
        args.resume = True
        with self.assertRaisesRegex(PREP.PreparationError, "mutually exclusive"):
            PREP.validate_args(args)

    def test_force_only_removes_selected_categories_and_markers(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            source_root = root / "source"
            output_root = root / "output"
            source_root.mkdir()
            for category in ("A", "B"):
                target = output_root / category
                target.mkdir(parents=True)
                (target / "data.txt").write_text(category)
            completed = output_root / "_reports/completed"
            completed.mkdir(parents=True)
            for category in ("A", "B"):
                (completed / f"{category}.json").write_text(
                    json.dumps({"route_stats": [{"category": category}]})
                )
            unrelated = output_root / "keep.txt"
            unrelated.write_text("keep")

            PREP.reset_selected_outputs(output_root, source_root, ["A"])

            self.assertFalse((output_root / "A").exists())
            self.assertTrue((output_root / "B/data.txt").exists())
            self.assertFalse((completed / "A.json").exists())
            self.assertTrue((completed / "B.json").exists())
            self.assertTrue(unrelated.exists())

    def test_force_rejects_output_root_containing_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            source_root = root / "source"
            source_root.mkdir()
            with self.assertRaisesRegex(PREP.PreparationError, "contains the source root"):
                PREP.reset_selected_outputs(root, source_root, ["A"])


if __name__ == "__main__":
    unittest.main()

# Prepare DFM9 data for separate Infini-gram indexes

`prepare_dfm9_index_data.py` converts the heterogeneous DFM9 memorisation
source bundle into four independent Infini-gram input trees. It treats
`manifest.tsv` as the routing authority, preserves A/B/C/D and cohort
boundaries, streams large inputs, and leaves the source bundle unchanged.

## Environment

Use the repository's `olmotrace_test` environment for every command:

```bash
conda run -n olmotrace_test python \
  00_prepare_data/prepare_dfm9_index_data.py --help
```

The implementation is tested with Python 3.11, PyArrow 23.0.1, zstandard
0.25.0, transformers 4.57.5, and infini-gram 2.6.0.

## Validate the source inventory

Run inventory and manifest validation before allocating space for output:

```bash
conda run -n olmotrace_test python \
  00_prepare_data/prepare_dfm9_index_data.py \
  --source-root /work/olmotrace/mimir_propme/dfm9_memorisation_sources \
  --output-root /absolute/path/dfm9_index_inputs \
  --categories A B C D \
  --dry-run
```

`--dry-run` writes nothing. It prints the tasks that can run and structured
problem entries for unknown selectors, missing manifest artifacts, different
inventories for repeated canonical sources, and directories with no indexable
data. Those entries are omitted while independent source groups remain usable.

At implementation time, A/B/C pass full inventory validation. The supplied D
bundle has two manifest entries with no data file:

- `D/D-10/data__downloads__datasets__natural_reasoning`
- `D/D-10/data__downloads__datasets__omni_math`

Both contain only README/cache metadata and are not declared in `gaps.tsv`.
An all-category or D run records them as `no_supported_data_files` in
`_reports/problems.tsv` and continues with the available D sources. The
preprocessor does not silently replace them. D-05 remains the sole allowed
unresolved selector and is reported as skipped.

Three other D-10 artifacts are large top-level JSON arrays with a `.json`
suffix. They are explicitly whitelisted and streamed incrementally:
`toolace/data.json`, `glaive-function-calling-v2.json`, and
`xlam_function_calling_60k.json`. All other standalone JSON manifests remain
excluded.

## Prepare the index inputs

After source validation succeeds:

```bash
conda run -n olmotrace_test python \
  00_prepare_data/prepare_dfm9_index_data.py \
  --source-root /work/olmotrace/mimir_propme/dfm9_memorisation_sources \
  --output-root /absolute/path/dfm9_index_inputs \
  --categories A B C D \
  --workers 8 \
  --target-part-mib 512
```

The output root must be absent or empty. The result is:

```text
dfm9_index_inputs/
  A/A-01/<source-group>/part-00000.jsonl.zst
  B/B-03/<source-group>/part-00000.jsonl.zst
  C/C-14/<source-group>/part-00000.jsonl.zst
  D/D-10/<source-group>/part-00000.jsonl.zst
  _reports/
```

Only `.jsonl.zst` index inputs occur below category directories. Reports,
snapshots, completed-artifact markers, source maps, and exclusions are under
`_reports` and therefore never enter an index whose `--data_dir` is one of the
category directories.

Parquet is read in bounded Arrow batches. JSONL, JSONL.GZ, and JSONL.ZST are
read incrementally. Output uses Zstandard level 3 and rolls at approximately
512 MiB of uncompressed JSONL. File-level source groups are processed in up to
eight worker processes. Entries sharing one canonical `source_path` are read
once and routed to all applicable cohorts.

## Record contract

Every emitted line has a non-empty `text`, deterministic `id`, the risk
category and cohort, full manifest provenance, stable source mapping, source
row, and JSON-safe `source_metadata`. Consumed content is not duplicated in
metadata; message roles and other structural fields are retained. Original
text whitespace is not globally normalized. Empty documents are counted and
dropped, while malformed input and unknown schemas fail their source group.

B-03 is the only content-level deduplication. Its embedded documents are
deduplicated by SHA-256 of the audit-normalized (wrapper-boundary-trimmed)
value and checked against the supplied audit register's exact
document-length/occurrence multiset. The emitted `text` retains the raw payload
whitespace, and both raw and audit-normalized hashes are metadata. Direct hash
matches and mismatches are reported separately because the audit register's
historical hashes do not all match hashes of the literal extracted text. An
audit or expected-count discrepancy is also written to `problems.tsv`; it does
not discard otherwise usable output. No other cohort is globally deduplicated.

## Recovery and validation

Each output part is first written as a hidden `.tmp` file and atomically
renamed. A completed source group receives a marker containing output sizes,
record counts, and SHA-256 hashes. To resume an interrupted run:

```bash
conda run -n olmotrace_test python \
  00_prepare_data/prepare_dfm9_index_data.py \
  --source-root /work/olmotrace/mimir_propme/dfm9_memorisation_sources \
  --output-root /absolute/path/dfm9_index_inputs \
  --categories A B C D \
  --workers 8 \
  --target-part-mib 512 \
  --resume
```

Resume verifies both size and SHA-256 before accepting a completed group. An
incomplete group or invalid marker is regenerated in its cohort-specific
directory, and a marker problem is recorded. Without `--resume`, a non-empty
output root is rejected. Use `--force` for an intentional fresh run of the
categories named by `--categories`. It removes and recreates only those
category directories and preserves every unselected category tree; `--force`
and `--resume` cannot be used together.

A malformed source group is cleaned from every affected cohort and recorded in
`problems.tsv`; other worker futures continue. The command writes all reports
and exits successfully after recoverable task-build, source-group, and expected
count problems. Review `summary.json` (`status` is `completed_with_problems`)
and `problems.tsv` before indexing.

Validate every output line, category, ID, record count, and file hash before
indexing:

```bash
conda run -n olmotrace_test python \
  00_prepare_data/prepare_dfm9_index_data.py \
  --output-root /absolute/path/dfm9_index_inputs \
  --categories A B C D \
  --validate-only
```

Validation uses disk-backed ID buckets so it does not retain all IDs in
memory. A duplicate ID is fatal.

## Reports and indexing handoff

`_reports` contains:

- `manifest.tsv`, `gaps.tsv`, and their SHA-256 values in `run_config.json`
- `counts.tsv`, `source_map.tsv`, `output_files.tsv`, and `skipped.tsv`
- `problems.tsv` with missing inputs, validation discrepancies, recoveries, and
  isolated source-group failures (including tracebacks)
- package versions, effective arguments, and an aggregate `summary.json`
- `completed/*.json` recovery records
- `index_commands.md` with category-specific indexing and unigram commands;
  affected categories carry a prominent problem warning

The generated A/B/D commands start with one shard; C starts with two. For
example:

```bash
- `excluded_files.md`, grouped by A/B/C/D, for wholly excluded manifest files
conda run -n olmotrace_test python -m infini_gram.indexing \
  --data_dir /absolute/path/dfm9_index_inputs/A \
  --save_dir /absolute/path/index/A \
  --temp_dir /absolute/path/scratch/A \
  --tokenizer llama \
  --cpus 128 \
  --mem 350 \
  --shards 1 \
  --batch_size 1024 \
  --add_metadata \
  --ulimit 1048576
```

Use the generated command separately for A, B, C, and D, then run each
generated `compute_unigrams.py` command against its corresponding index.

## Tests

```bash
conda run -n olmotrace_test python -m unittest \
  00_prepare_data/test_prepare_dfm9_index_data.py -v
```

The suite covers streaming formats, schema conversion, metadata preservation,
selectors, B-03 deduplication, shared-source routing, rollover, Infini-gram's
installed `.zst` loader, resume integrity, and duplicate-ID validation.

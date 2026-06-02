# Validation

This folder contains validation scripts for `SimpleTrace`, the tracing component used by the broader `PropMe` pipeline.

The goal of these scripts is not just to test raw index lookup, but to validate the actual end-to-end tracing behavior that downstream memorization and propensity analyses depend on.


## Validation entry points

There are currently three validation scripts:

- `04_validation/validation.py`: deterministic unit-style checks on the local `00_data/dummy_index`.
- `04_validation/validation_full_dynaword.py`: randomized large-scale validation on a Dynaword index.
- `04_validation/validation_full_commonpile.py`: randomized large-scale validation on a Common Pile index.

## Quick start

### 1. Fast local sanity check

Use this first when you want a lightweight correctness check of the tracing pipeline:

```bash
python 04_validation/validation.py -v
```

This validates known dummy examples such as:

- exact substring retrieval,
- full-document retrieval,
- cross-document matches,
- negative queries with no final spans,
- summary accounting for full vs partial matches.

### 2. Randomized Dynaword validation

Use this when you want to test `SimpleTrace` retrieval on randomly sampled source documents from a Dynaword index:

```bash
python 04_validation/validation_full_dynaword.py \
  --index-dir 00_data/dynaword_index \
  --unigram-probs-path 02_unigram_probs/unigram_probs_dynaword.json \
  --num-samples 50 \
  --docs-per-span 10
```

### 3. Randomized Common Pile validation

Use this when you want the same style of randomized validation on a Common Pile index:

```bash
python 04_validation/validation_full_commonpile.py \
  --index-dir /work/olmotrace/common_pile_train/indexes/common_pile_train_index \
  --unigram-probs-path 02_unigram_probs/unigram_probs_common_pile_train.json \
  --num-samples 50 \
  --docs-per-span 10
```

## Common flags for the full validators

Both randomized validators support the same core controls:

- `--num-samples` or `-n`: number of source documents to validate.
- `--query-token-len`: maximum token length for full-document queries; `None` uses the full sampled document window.
- `--partial-query-tokens`: maximum token length for partial start/middle/end queries.
- `--min-query-tokens`: minimum token length required for a usable query window.
- `--no-full`: disable full-document queries.
- `--no-partials`: disable partial start/middle/end queries.
- `--docs-per-span`: maximum retrieved documents per traced span.
- `--tokenizer-model`: tokenizer used for decoding and re-encoding query windows.
- `--seed`: random seed for reproducible sampling.
- `--max-attempts-multiplier`: how hard the script tries to find usable source documents.

Both scripts also expose output path flags for sampled queries, raw traces, per-query metrics, summaries, and failure logs.

## Useful run variants

### Partials only

```bash
python 04_validation/validation_full_dynaword.py \
  --index-dir 00_data/dynaword_index \
  --unigram-probs-path 02_unigram_probs/unigram_probs_dynaword.json \
  --num-samples 50 \
  --no-full
```

### Full queries only

```bash
python 04_validation/validation_full_dynaword.py \
  --index-dir 00_data/dynaword_index \
  --unigram-probs-path 02_unigram_probs/unigram_probs_dynaword.json \
  --num-samples 50 \
  --no-partials
```

The same flag combinations also work for `validation_full_commonpile.py`.

## Details

### What `validation.py` checks

`validation.py` is the smallest and fastest validator. It uses the local dummy dataset and dummy index to exercise known cases with deterministic expectations.

It verifies:

- exact retrieval from the beginning, middle, and end of a source document,
- retrieval of a known short phrase,
- a cross-document query that should retrieve multiple documents,
- a nonexistent query that should produce no final spans,
- evaluation-summary behavior such as `full_exact_matches`, `partial_matches`, and tracked document IDs.

This is the best entry point when you want confidence that a local change did not break basic `SimpleTrace` behavior.

### What the full validators do

Both randomized validators follow the same high-level protocol:

1. Load a tokenizer, an InfiniGram index, and unigram probabilities.
2. Randomly sample candidate source documents from the indexed corpus.
3. Build one or more validation queries per sampled document.
4. Run `trace_generation(...)` on each query.
5. Check whether the expected source document was retrieved and whether the query text appears in the retrieved document text.
6. Save raw traces, per-query rows, aggregate summaries, and debugging logs.

The two scripts differ mainly in how query windows are chosen and in the extra diagnostics they report.

### Query types

The full validators use up to four query kinds:

- `full`: a full sampled source-document window, or a length-capped prefix of it if `--query-token-len` is set.
- `partial_start`: a partial window anchored near the beginning.
- `partial_middle`: a partial window centered around the middle.
- `partial_end`: a partial window anchored near the end.

### Dynaword validation behavior

`validation_full_dynaword.py` uses "clean traceable" windows for partial queries. These are designed to respect the kinds of spans that `SimpleTrace` itself is likely to preserve, such as:

- starting at word boundaries,
- ending at word boundaries,
- avoiding punctuation patterns that the final span filter would typically reject.

This makes the Dynaword validation stricter and better aligned with real `SimpleTrace` span retention behavior.

### Common Pile validation behavior

`validation_full_commonpile.py` also samples full and partial queries, but its partial window construction is simpler and more positional.

It additionally reports "rescued" cases where:

- the expected source document ID is not returned among retrieved docs,
- but the exact partial span text is still recovered.

That distinction is useful for corpora with many duplicated or near-duplicated documents, where the tracer may recover the right text but not the exact original document identifier within the retrieval budget.

### Source-document acceptance policy

For the full validators, a source document is accepted only if all enabled query kinds can be constructed for it.

That means:

- if full queries are enabled, the `full` query must be constructible,
- if partial queries are enabled, all three partial query types must be constructible.

If the script cannot find enough usable source documents, it raises an error instead of silently returning too few samples.

### Expected query counts

When both query families are enabled:

- each accepted source document produces `4` queries,
- so `--num-samples 50` yields `200` validation queries.

If you disable one query family:

- `--no-full` yields `3 * num_samples` queries,
- `--no-partials` yields `1 * num_samples` queries.

## Outputs

### `validation.py`

`validation.py` is a `unittest` suite and prints normal test output to the terminal.

### `validation_full_dynaword.py`

By default it writes:

- `04_validation/output/validation_full_samples_dynaword.jsonl`: sampled query definitions.
- `04_validation/output/validation_full_traces_dynaword.jsonl`: raw traced spans and retrieved docs.
- `04_validation/output/validation_full_per_query_dynaword.jsonl`: per-query retrieval and text-match rows.
- `04_validation/output/validation_full_summary_dynaword.json`: aggregate summary metrics and run config.
- `04_validation/output/validation_full_missing_doc_logs.txt`: detailed logs for cases where the expected source document ID was not retrieved.

Its core summary metrics include:

- `source_doc_retrieval_rate`
- `exact_text_match_rate`
- `full_doc_retrieval_rate`
- `full_exact_text_match_rate`
- `partial_start_doc_retrieval_rate`
- `partial_start_exact_text_match_rate`
- `partial_middle_doc_retrieval_rate`
- `partial_middle_exact_text_match_rate`
- `partial_end_doc_retrieval_rate`
- `partial_end_exact_text_match_rate`

### `validation_full_commonpile.py`

By default it writes:

- `04_validation/output/validation_full_samples_commonpile.jsonl`: sampled query definitions.
- `04_validation/output/validation_full_traces_commonpile.jsonl`: raw traced spans and retrieved docs.
- `04_validation/output/validation_full_per_query_commonpile.jsonl`: per-query retrieval and validation rows.
- `04_validation/output/validation_full_summary_commonpile.json`: aggregate summary metrics and run config.
- `04_validation/output/validation_full_failed_examples_commonpile.txt`: detailed logs for failed validation examples.

Its summary contains the same retrieval and text-match rates as Dynaword, plus extra fields such as:

- `pass_rate`
- `failed_examples`
- `partial_query_doc_id_not_retrieved_but_span_exact_match_count`
- per-query-kind `*_pass_rate`
- per-query-kind `*_partial_span_exact_query_match_rate`

## When to use which validator

- Use `validation.py` for quick local regression checks.
- Use `validation_full_dynaword.py` when you want stricter randomized validation aligned with `SimpleTrace` span-cleanliness constraints.
- Use `validation_full_commonpile.py` when you want randomized validation on Common Pile and extra visibility into duplicated-document retrieval edge cases.

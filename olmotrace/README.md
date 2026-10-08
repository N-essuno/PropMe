## OLMoTrace (reproduction)

Offline reproduction of OLMoTrace, based only on the paper
[OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens](https://arxiv.org/abs/2504.07096)
(Liu et al., 2025). It uses the same InfiniGram indexes (`--index-dir`) and
unigram tables (`02_unigram_probs/*.json`) as `SimpleTrace`, and the same
command-line shape as `03_tracing/simple_trace.py`.

See [`OLMOTRACE_REPORT.md`](OLMOTRACE_REPORT.md) for implementation notes and the
comparison with SimpleTrace.

### Pipeline (paper §3)

1. **Maximal matching spans.** For every begin-of-word position of the
   Llama-2-tokenized response, one FIND on the whole suffix gives its longest
   matching prefix (neighbouring suffix-array entries when the count is 0; max
   over shards). The prefix is shrunk until it has no period/newline token
   except at its end and is followed by a begin-of-word token; non-maximal
   spans are suppressed. Suffix queries run in parallel (`--find-threads`).
2. **Filter.** Keep the `K = ceil(0.05 * L)` spans with the lowest span unigram
   probability.
3. **Retrieve.** Up to 10 enclosing document snippets per span (random sample
   if the span occurs more often): an 80-token snippet and a 500-token
   extended context.
4. **Merge.** Union overlapping spans; merge snippets from the same document.
5. **Rerank.** BM25 over the retrieved documents with `prompt + response` as
   the query; scores normalized by `0.18 * len(response)` and bucketed into
   high (>= 0.7), medium (>= 0.5) and low relevance. A span's relevance is the
   max over the documents enclosing it.

### Example run

```bash
python olmotrace/olmo_trace.py \
    --dataset dummy \
    --index-dir 00_prepare_data/dummy_index \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dummy.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --results-output olmotrace_results_dummy.jsonl \
    --summary-output olmotrace_summary_dummy.json \
    --length-buckets 1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf
```

Generation JSON produced by `memorization_experiment/generate_vllm.py` (the
`prompt` field of each item is added to the BM25 query):

```bash
python olmotrace/olmo_trace.py \
    --dataset /absolute/path/to/generations.json \
    --is-generation-json \
    --generation-text-field completion \
    --generation-prompt-field prompt \
    --index-dir /absolute/path/to/index_dir \
    --unigram-probs-path 02_unigram_probs/unigram_probs_my_corpus.json \
    --num-workers 8 \
    --results-output outputs/olmotrace_results.jsonl \
    --summary-output outputs/olmotrace_summary.json
```

JSONL input works as in SimpleTrace (`--is-jsonl --text-field text`, plus an
optional `--prompt-field`). Several `--index-dir` values are loaded as one
multi-shard index, e.g. all Dolma3 split indexes.

Flags that set the paper's hyperparameters (defaults are the paper's final
setting): `--docs-per-span 10`, `--span-fraction 0.05`,
`--snippet-context-tokens 80`, `--document-context-tokens 500`. `--seed` makes
the sampling of snippets for frequent spans reproducible.

### Outputs

- Results JSONL, one line per input in input order: `filtered_spans` (step 2,
  with unigram log-probability and corpus count), `spans` (merged highlights
  with relevance and `doc_ranks`), `documents` (ranked by BM25, each with
  `relevance`, `span_indices` and its snippets/contexts) and per-step latency.
- Summary JSON with the statistics reported in the paper: span-length
  distribution after step 2, document and span relevance distributions,
  max BM25 per response character, and latency.

### Tests

```bash
python -m unittest discover -s olmotrace/tests -v
```

The tests use `00_prepare_data/dummy_index` and check, among others, that the
single-FIND longest-prefix computation and Algorithm 1 agree with a brute-force
enumeration of the paper's span definition, on one and two shards.

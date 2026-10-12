## 4. Run SimpleTrace on generations

### Example Run

Example command for the dummy dataset:

```bash
python 03_tracing/simple_trace.py \
    --dataset dummy \
    --index-dir 00_prepare_data/dummy_index \
    --unigram-probs-path 02_unigram_probs/unigram_probs_dummy.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --results-output simpletrace_results_dummy.jsonl \
    --summary-output simpletrace_evaluation_summary_dummy.json \
    --length-buckets 1-3,4-6,7-10,11-20,21-50,51-100,101-150,151-inf
```

### JSONL input

Use this when you have one JSON object per line and the text to trace is in a field such as `text`.

```bash
python 03_tracing/simple_trace.py \
    --dataset /absolute/path/to/generations.jsonl \
    --is-jsonl \
    --text-field text \
    --index-dir /absolute/path/to/index_dir \
    --unigram-probs-path 02_unigram_probs/unigram_probs_my_corpus.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --match-mode mixed \
    --results-output outputs/simpletrace_results.jsonl \
    --summary-output outputs/simpletrace_summary.json
```

### Generation JSON input

Use this when your generations are stored in a JSON file and the generated text is inside a field such as `completion`.

Each generation is traced separately, even when several are identical (e.g. repeated samples of one prompt). Records that have `prompt_id` and `sample_idx` (written by `generate_vllm.py` and `generate_vllm_free.py`) are keyed `<prompt_id>:<sample_idx>`, and these fields are copied into the results file; other inputs are keyed by position and get an `index` field.

```bash
python 03_tracing/simple_trace.py \
    --dataset /absolute/path/to/generations.json \
    --is-generation-json \
    --generation-text-field completion \
    --index-dir /absolute/path/to/index_dir \
    --unigram-probs-path 02_unigram_probs/unigram_probs_my_corpus.json \
    --num-workers 8 \
    --docs-per-span 10 \
    --match-mode mixed \
    --results-output outputs/simpletrace_results.jsonl \
    --summary-output outputs/simpletrace_summary.json
```

Useful flags:

- `--match-mode text`: prose-oriented tracing. A match that runs across a sentence boundary (`!`, `.`, `?`, newline) is trimmed to end at that boundary instead of being discarded.
- `--match-mode mixed`: better for full-text, code, math, markup, and mixed-content generations.
- `--limit N`: trace only the first `N` generations.
- `--max-page-table-gb G` (default 1): a worker reopens its index once its page tables exceed `G` GB (checked after every index call). On huge memory-mapped indexes such as Dolma3, page tables otherwise grow without bound during long runs (up to ~20 MB per uncached lookup across Dolma3's 65 shards) and can exhaust the container's memory. Results are unchanged; total page-table memory is about `--num-workers × G`.
- `--find-threads N` (default 4): threads per worker that run the index queries of one generation concurrently. It only affects speed, not results; total concurrency is `--num-workers × --find-threads`. On the Dolma3 indexes it roughly halved the time of a first (cold-cache) run and was ~10% slower when the index was already cached.
- `--seed S` (default 42): seed for picking `--docs-per-span` documents when a span matches more documents than that. Each generation gets its own RNG seeded with the seed and its text, so results and summaries are identical across reruns, worker and thread counts, and between an index and the same shards loaded another way (e.g. Dolma3 linked vs split). Before this flag the sample was unseeded, and document-based metrics (full matches, NV recall, doc ids) varied by about ±0.5 percentage points between identical runs.
- `--nv-recall-threshold 0.5`: threshold used in summary reporting.
- `--k-eidetic-values 1,5,10`: optional post-hoc k-eidetic memorization evaluation.
- `--n-token-span-ratio 60`: report the fraction of generations with a span of at least `N` tokens.
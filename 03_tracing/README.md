## 4. Run SimpleTrace on generations

### Example Run

Example command for the dummy dataset:

```bash
python 03_tracing/simple_trace.py \
    --dataset dummy \
    --index-dir 00_data/dummy_index \
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

- `--match-mode text`: original prose-oriented tracing.
- `--match-mode mixed`: better for full-text, code, math, markup, and mixed-content generations.
- `--limit N`: trace only the first `N` generations.
- `--nv-recall-threshold 0.5`: threshold used in summary reporting.
- `--k-eidetic-values 1,5,10`: optional post-hoc k-eidetic memorization evaluation.
- `--n-token-span-ratio 60`: report the fraction of generations with a span of at least `N` tokens.
## Precompute unigram probabilities for the indexed corpus

`SimpleTrace` ranks candidate spans by rarity using unigram probabilities computed from the index.

```bash
python 02_unigram_probs/compute_unigrams.py \
    --index-dir /absolute/path/to/index_dir \
    --output-path 02_unigram_probs/unigram_probs_my_corpus.json \
    --tokenizer-model meta-llama/Llama-2-7b-hf \
    --example-token a \
    --top-k 10
```
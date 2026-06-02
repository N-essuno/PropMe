# How to prepare datasets for tracing

## Indexing

Indexing is done through the `infini-gram` library. This library allows us to create an index of the dataset, which can be used for efficient retrieval of examples during tracing.

When indexing large datasets some considerations are needed with respect to resources like CPU and memory. For more details we recommend checking out the [`infini-gram` documentation](https://infini-gram.readthedocs.io/en/latest/indexing.html)

### Indexing the dummy dataset

This dataset is very small so we can index it with a normal laptop.

```bash
python -m infini_gram.indexing \
    --data_dir <dummy_dataset_dir> \
    --save_dir <dummy_index_dir> \
    --tokenizer llama \
    --cpus 1 \
    --mem 16 \
    --shards 1 \
    --add_metadata \
    --ulimit 1048576
```

### Indexing Dynaword

This dataset is larger, so we recommend indexing it on a machine with more resources. The command below is an example of how to index it on a machine with 16 CPUs and 84 GB of memory. Adjust the `--cpus` and `--mem` flags according to the resources available on your machine.

```bash
python -m infini_gram.indexing \
    --data_dir <dynaword_dataset_dir> \
    --save_dir <dynaword_index_dir> \
    --tokenizer llama \
    --cpus 16 \
    --mem 84 \
    --shards 1 \
    --add_metadata \
    --ulimit 1048576
```

### Indexing Common Pile

Common Pile is a relatively large dataset of 521 GB in storage and 463.6B tokens. Indexing it will require a machine with significant resources. In the setting below, we split the dataset into 3 parts and indexed each part separately on a machine with 128 CPUs and 350 GB of memory. Each part took around 3 hours to index.

```bash
python -m infini_gram.indexing \
    --data_dir <path/to/common_pile_train_1> \
    --save_dir <path/to/common_pile_train_1_index> \
    --tokenizer llama \
    --cpus 128 \
    --mem 350 \
    --shards 2 \
    --add_metadata \
    --ulimit 1048576

python -m infini_gram.indexing \
    --data_dir <path/to/common_pile_train_2> \
    --save_dir <path/to/common_pile_train_2_index> \
    --tokenizer llama \
    --cpus 128 \
    --mem 350 \
    --shards 2 \
    --add_metadata \
    --ulimit 1048576

python -m infini_gram.indexing \
    --data_dir <path/to/common_pile_train_3> \
    --save_dir <path/to/common_pile_train_3_index> \
    --tokenizer llama \
    --cpus 128 \
    --mem 350 \
    --shards 2 \
    --add_metadata \
    --ulimit 1048576
```


## Computing unigrams

This will produce a JSON file with probabilities and log probabilities for each token in the indexed dataset. Top-k and example-token are just for inspection purposes. The `--top-k` flag is used to print the top-k most common tokens in the dataset, while the `--example-token` flag is used to print the probability and log probability of a specific token (in this case, the token "a").

```bash
python 02_unigram_probs/compute_unigrams.py \
    --index-dir <index_dir> \
    --output-path <unigram_output_path> \
    --tokenizer-model meta-llama/Llama-2-7b-hf \
    --example-token a \
    --top-k 10
```

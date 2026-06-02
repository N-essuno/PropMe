## Build an InfiniGram index for the corpus you want to trace against

`SimpleTrace` traces generations against a corpus that has already been indexed with `infini-gram`.

Example:

```bash
python -m infini_gram.indexing \
    --data_dir /absolute/path/to/dataset_dir \
    --save_dir /absolute/path/to/index_dir \
    --tokenizer llama \
    --cpus 16 \
    --mem 84 \
    --shards 1 \
    --add_metadata \
    --ulimit 1048576
```

Notes:

- Use absolute paths for `--data_dir` and `--save_dir`.
- Keep `--add_metadata`; downstream tracing uses document metadata when available.
- Resource needs depend heavily on corpus size. See `preprocess.md` for more examples, including Common Pile and Dynaword.
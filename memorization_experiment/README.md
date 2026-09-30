# Memorization Experiments

This folder contains the experiment code and input assets used to run the `SimpleTrace` and `PropMe` memorization evaluations in this repository.

## Code in this folder

The scripts in this folder are:

- `generate_vllm.py`: batch prompt inference through a running OpenAI-compatible vLLM server.
- `sample_docs.py`: sample source documents from an InfiniGram index into `*_sample_docs.jsonl`.
- `commonpile_extract_prefixes.py`: build Common Pile prefix prompts from sampled source documents.
- `dynaword_extract_prefixes.py`: build Dynaword prefix prompts from sampled source documents.
- `dolma3_extract_prefixes.py`: sample the combined Dolma3 index and build prefix prompts.
- `dfm9_extract_prefixes.py`: sample selected DFM9 A/B/C/D indexes and write separate 50/75/100-token prefix prompts.
- `dfm9_domain_distributions.py`: resolve DFM9 SimpleTrace dataset IDs and report domain distributions by span-length bucket and NV-recall threshold.

The rest of the experiment pipeline is using code from other folders in the repository.

## Input assets in this folder

The experiment code uses a few recurring input-file patterns:

- `test_prompts.jsonl`: small prompt sets for ad hoc generation checks.
- `generic_prompts.jsonl`: ordinary non-adversarial generic prompts.
- `specific_prompts.jsonl`: ordinary non-adversarial dataset-specific prompts.
- `*_prefix_prompts.jsonl`: capability-style prefix prompts.
- `*_sample_docs.jsonl`: sampled source documents used to derive prefix prompts.

## Prompt settings

The experiment code follows the same three settings used in the paper and in the main project README:

- `generic`: ordinary, non-adversarial generic prompts.
- `specific`: ordinary, non-adversarial prompts that are more dataset-specific.
- `prefix`: capability-oriented prompts extracted from training-set source documents.

In the PropMe framing:

- `generic` and `specific` are the ordinary-use settings,
- `prefix` is the capability setting used as the adversarial comparison point.

## Generate model completions

Use `generate_vllm.py` to send prompts to a running vLLM server and save the returned completions.

Example:

```bash
vllm serve danish-foundation-models/dfm-decoder-open-v0-7b-pt --host 127.0.0.1 --port 8000

python memorization_experiment/generate_vllm.py \
  --model danish-foundation-models/dfm-decoder-open-v0-7b-pt \
  --api_base http://127.0.0.1:8000/v1 \
  --input_jsonl memorization_experiment/data/dynaword/generic/generic_prompts.jsonl \
  --output_file memorization_experiment/data/dynaword/generic/generic_generations.json
```

CLI options include:

- `--model`: model name served by vLLM.
- `--api_base`: OpenAI-compatible base URL, typically `http://127.0.0.1:8000/v1`.
- `--input_json` or `--input_jsonl`: exactly one input source must be provided.
- `--text_field`: prompt text field for JSONL input. Default is code-defined in the script.
- `--domain_field`: domain field for JSONL input.
- `--output_file`: output JSON file for generations.
- `--batch_size`: request batch size.
- `--max_input_tokens`: prompt truncation cap passed to the server.
- `--max_new_tokens`: maximum generated continuation length.
- `--do_sample`, `--temperature`, `--top_p`: sampling controls.
- `--num_beams`: beam-search control.
- `--repetition_penalty`: repetition penalty forwarded to vLLM.
- `--seed`: reproducibility seed.

## Build prefix prompts

The `prefix` setting is created from sampled source documents rather than handwritten prompts.

First sample source documents from the relevant InfiniGram index. The prefix extraction scripts expect the sampled documents to exist before they run.

Example for Dynaword:

```bash
python memorization_experiment/sample_docs.py \
  --index-dir 00_data/dynaword_index \
  --output-path memorization_experiment/data/dynaword/dynaword_sample_docs.jsonl \
  --num-docs 100 \
  --min-tokens 100 \
  --tokenizer-model meta-llama/Llama-2-7b-hf
```

Then extract prefixes with:

- `commonpile_extract_prefixes.py`
- `dynaword_extract_prefixes.py`

Both scripts currently:

- load sampled source documents from `*_sample_docs.jsonl`,
- tokenize them with `meta-llama/Llama-2-7b-hf`,
- extract the first `50` tokens,
- write JSONL prompt records with `text` and `domain` fields.

Run them from the repository root:

```bash
python memorization_experiment/dynaword_extract_prefixes.py
```

### Dolma3 prefixes

The Dolma3 extractor samples directly from the combined symlink-backed index,
so it does not need a separate `sample_docs.py` step. From the repository root:

```bash
python memorization_experiment/dolma3_extract_prefixes.py --num-docs 100
```

To load the separate split indexes instead, use:

```bash
python memorization_experiment/dolma3_extract_prefixes.py \
  --split-indexes \
  --num-docs 100
```

`--indexes-root` changes the parent directory used by `--split-indexes`.
Alternatively, pass one or more directories explicitly with `--index-dir`.
The split selection includes both split 13 variants and
`dolma3_split26_index_6shards`, matching the linked index.

It writes `memorization_experiment/data/dolma3/dolma3_sample_docs.jsonl`
and `memorization_experiment/data/dolma3/prefix/dolma3_prefix_prompts.jsonl`.
The defaults use the Llama tokenizer, require at least 100 source tokens, and
take a 50-token prefix. Use `--index-dir`, `--output-dir`, `--min-tokens`,
`--prefix-tokens`, or `--seed` to change those settings. The default index is
`/work/pecora/propme_data/indexes/dolma3_index_link`.


### DFM9 category prefixes

For DFM9, the extractor samples directly from any selected category indexes.
The same sampled documents are reused for every requested prefix length:

```bash
HF_HUB_OFFLINE=1 python -u memorization_experiment/dfm9_extract_prefixes.py \
  --indexes-root /work/olmotrace/mimir_propme/indexes \
  --categories A B C D \
  --num-docs 100 \
  --prefix-lengths 50 75 100
```

By default it writes the following files under
`memorization_experiment/data/dfm9/prefix/` for each selected category:

```text
dfm9_A_prefix_50_prompts.jsonl
dfm9_A_prefix_75_prompts.jsonl
dfm9_A_prefix_100_prompts.jsonl
```

The same `50`, `75`, and `100` files are produced for categories B, C, and D.

Every line retains the existing prefix-prompt contract with `text` and
`domain` fields. Documents must have at least as many usable Llama tokens as
the longest requested prefix. Sampling is distinct, deterministic, and uses
100 documents per category by default; use `--seed`, `--num-docs`, and
`--min-tokens` to override those settings.

## Run SimpleTrace on the generated completions

The experiment runner is `run_memorization_experiments.py`.

Inspect the available preset groups and experiments:

```bash
python memorization_experiment/run_memorization_experiments.py --list
```

Example for running experiments on dynaword generic, specific and prefix generations:

```bash
python memorization_experiment/run_memorization_experiments.py dynaword-generations
```

Run the DFM9 generic and A/B/C/D prefix-generation presets with:

```bash
python memorization_experiment/run_memorization_experiments.py dfm9-generations
```

The DFM9 group runs Generic EN and Generic DA separately against each
singular A, B, C, and D index, followed by the four matching prefix traces.
The existing memorization and comparison plot paths are regenerated in place
from these twelve summaries.

The runner is responsible for connecting generated completions to the right tracing configuration:

- it selects the correct generation JSON file,
- forwards the right dataset flag such as `--is-generation-json`,
- selects the appropriate index and unigram files,
- applies the experiment-specific tracing defaults.

Outputs are stored in memorization_experiment/data with subfolders for each experiment.

## Compute PropMe reports

After tracing, use `compute_propensity_metrics.py` to compare the non-prefix settings against the prefix setting.

List available preset groups:

```bash
python 05_propensity_metrics/compute_propensity_metrics.py --list
```

Example for computing metrics and plotting for the dynaword generations experiment:

```bash
python 05_propensity_metrics/compute_propensity_metrics.py dynaword-generations --plot
```

## Plotting code

Use `plot_memorization_results.py` for per-suite plots:

Example for the dynaword generations experiment:

```bash
python memorization_experiment/plot_memorization_results.py dynaword-generations
```

Alongside the existing metric and distribution plots, each suite produces:

- `span_length_distribution_heatmap.png`, an annotated view of the binned span ratios.
- `memorization_metrics_spider.png`, comparing full-generation matches, average NV recall, and average longest span. The span axis is normalized to the 100-token plotting scale.

Use `plot_comparison_overviews.py` for cross-model and cross-stage comparison views:

```bash
python memorization_experiment/plot_comparison_overviews.py dynaword-stages-comparison
```


## Analyze DFM9 domain distributions

After SimpleTrace has produced one or more result files, resolve the retrieved
DFM9 dataset IDs back to their full metadata and calculate domain
compositions:

```bash
python -u memorization_experiment/dfm9_domain_distributions.py \
  --results \
    memorization_experiment/data/dfm9/generic/st_dfm9_generic_en_results.json \
    memorization_experiment/data/dfm9/generic/st_dfm9_generic_da_results.json \
    memorization_experiment/data/dfm9/prefix/st_dfm9_A_prefix_50_results.json \
    memorization_experiment/data/dfm9/prefix/st_dfm9_B_prefix_50_results.json \
    memorization_experiment/data/dfm9/prefix/st_dfm9_C_prefix_50_results.json \
    memorization_experiment/data/dfm9/prefix/st_dfm9_D_prefix_50_results.json \
  --indexes-root /work/olmotrace/mimir_propme/indexes \
  --prepared-data-root /work/olmotrace/mimir_propme/dfm9_memorisation_sources_propme \
  --output-dir memorization_experiment/data/dfm9/domain_analysis \
  --nv-recall-threshold 0.5
```

The command accepts any number of result files. It queries only dataset IDs
present in those inputs, verifies exact metadata IDs returned by InfiniGram,
and caches successful resolutions. Decoded SimpleTrace windows can sometimes
be too short or common for bounded reverse lookup; in that case, the command
scans only the prepared source artifact identified by the ID's source key and
requires an exact ID match. Resolution methods and complete non-text metadata
are retained in `resolved_document_domains.jsonl`.

Each input gets one JSON report and three figures, and multiple inputs also get
a combined report/figure suite. Reports contain both retrieved-document
occurrence distributions and unique-dataset-ID distributions for:

- every standard memorization span bucket (`1-3`, `4-6`, `7-10`, `11-20`,
  `21-50`, `51-100`, `101-150`, and `151-inf` Llama tokens),
- all document occurrences whose per-document `nv_recall >= 0.5`,
- the `nv_recall >= 0.5` subset separately within every span bucket.

Use `--domain-fields` to change metadata precedence, `--top-domains` to change
plot truncation, `--no-plots` for JSON-only output, or `--refresh-cache` to
force fresh index resolution. The command is strict by default; unresolved IDs
are reported and fail the run instead of silently changing denominators.

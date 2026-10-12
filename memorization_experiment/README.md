# Memorization Experiments

This folder contains the experiment code and input assets used to run the `SimpleTrace` and `PropMe` memorization evaluations in this repository.

## Code in this folder

- `run_memorization_experiments.py`: trace the generations with SimpleTrace (see below).

`generation/`:

- `generate_vllm.py`: batch prompt inference through a running OpenAI-compatible vLLM server.
- `run_generations_all.sh`: generate every model's completions with vLLM (prompt sets, prefix prompts and the prompt-free settings), 10 per prompt; see the header of the script for models, settings and outputs.
- `generation_runs.py`: the models, corpora, settings and file layout of those generations, shared by the tracing, propensity and plotting presets below.
- `generate_vllm_free.py`: prompt-free generation through a running vLLM server: `unconditional` (start-of-document token only) or `minimal_cue` (start-of-document token plus one common word, sampled per generation from `wordfreq`).
- `subsample_prompt_sets.py`: write the 1000-prompt samples of the larger prompt sets and of their generations.

`extract_prefixes/`:

- `sample_docs.py`: sample source documents from an InfiniGram index into `*_sample_docs.jsonl`.
- `prefix_extraction.py`: shared index sampling and prefix extraction used by the three scripts below.
- `commonpile_extract_prefixes.py`: sample the Common Pile index and build 2000 prefix prompts.
- `dynaword_extract_prefixes.py`: sample the Dynaword index (excluding kb_administrative_publication, kb_historical_letters, municipality_meetings, hvadvilduhelst, tidsskrift-dk) and build 2000 prefix prompts.
- `dolma3_extract_prefixes.py`: sample the combined Dolma3 index and build 2000 prefix prompts.

`scripts/`:

- `plot_memorization_results.py`: per-suite plots.
- `plot_comparison_overviews.py`: cross-model and cross-stage comparison views.
- `full_match_length_distribution.py`: token-length distributions of fully matched generations, used by both plotting scripts.

The rest of the experiment pipeline is using code from other folders in the repository.

## Input assets in this folder

The experiment code uses a few recurring input-file patterns:

- `test_prompts.jsonl`: small prompt sets for ad hoc generation checks.
- `generic_prompts.jsonl`: ordinary non-adversarial generic prompts.
- `specific_prompts.jsonl`: ordinary non-adversarial dataset-specific prompts.

The generic and specific prompt sets are built in `00_prepare_data/propensity_settings` (see its README).
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

python memorization_experiment/generation/generate_vllm.py \
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
- `--num_generations`: completions per prompt (vLLM's `n`, default 1). Each completion is saved as its own record with the prompt's `prompt_id` and its `sample_idx`; SimpleTrace keys its results by these, so it traces all of them, identical completions included.

## Build prefix prompts

The `prefix` setting is created from sampled source documents rather than handwritten prompts.

The Common Pile, Dynaword, and Dolma3 extractors sample documents directly from
their InfiniGram index, so they do not need a separate `sample_docs.py` step.
All three share the logic in `prefix_extraction.py`:

- sample distinct documents with a seeded, lazy shuffle over the whole index,
- keep documents with at least `100` index tokens (`--min-tokens`),
- tokenize them with `meta-llama/Llama-2-7b-hf` and extract the first `50` tokens (`--prefix-tokens`),
- stop after `2000` documents (`--num-docs`) and fail if fewer qualify,
- write `<dataset>_sample_docs.jsonl` and `prefix/<dataset>_prefix_prompts.jsonl`
  (records with `text` and `domain` fields) under `--output-dir`.

Run them from the repository root:

```bash
python memorization_experiment/extract_prefixes/commonpile_extract_prefixes.py
python memorization_experiment/extract_prefixes/dynaword_extract_prefixes.py
python memorization_experiment/extract_prefixes/dolma3_extract_prefixes.py
```

Default indexes (`$PROPME_DATA_ROOT` defaults to `propme_data/` in the repository root) and outputs:

| Script | Default index | Output directory |
| --- | --- | --- |
| `commonpile_extract_prefixes.py` | `$PROPME_DATA_ROOT/indexes/commonpile_index/common_pile_train_index` | `memorization_experiment/data/commonpile` |
| `dynaword_extract_prefixes.py` | `$PROPME_DATA_ROOT/indexes/dynaword_index` | `memorization_experiment/data/dynaword` |
| `dolma3_extract_prefixes.py` | `$PROPME_DATA_ROOT/indexes/dolma3_index_link` | `memorization_experiment/data/dolma3` |

Use `--index-dir`, `--output-dir`, `--num-docs`, `--min-tokens`,
`--prefix-tokens`, `--tokenizer-model`, or `--seed` to change those settings.
The Common Pile index metadata carries no `source`, so its prompts get the
`unknown` domain.

### Dynaword excluded sources

The Dynaword extractor never samples documents whose metadata `source` is
`kb_administrative_publication`, `kb_historical_letters`,
`municipality_meetings`, `hvadvilduhelst`, or `tidsskrift-dk`; they are skipped during
sampling, so the output still contains `--num-docs` prompts. Override the list
with `--exclude-sources <source> ...`, or pass `--exclude-sources` with no
values to disable the filter.

### Dolma3 split indexes

To load the separate Dolma3 split indexes instead of the linked index, use:

```bash
python memorization_experiment/extract_prefixes/dolma3_extract_prefixes.py --split-indexes
```

`--indexes-root` changes the parent directory used by `--split-indexes`.
The split selection includes both split 13 variants and
`dolma3_split26_index_6shards`, matching the linked index.


### DFM9 category prefixes

For DFM9, the extractor samples directly from any selected category indexes.
The same sampled documents are reused for every requested prefix length:

```bash
HF_HUB_OFFLINE=1 python -u memorization_experiment/dfm9_extract_prefixes.py \
  --indexes-root $PROPME_DFM9_ROOT/indexes \
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

Each generation run (a model traced against one of its training corpora) has five experiments,
named `<model>-<corpus>-<setting>`: `generic`, `specific`, `unconditional`, `minimal-cue`
and `prefix`. The runs are `dfm-stage1`, `dfm-stage2` and
`dfm-main` on `dynaword` and `commonpile`, `comma-2t` on `commonpile` and `olmo3-32b` on
`dolma3`. Minimal cues are traced in the corpus language (`minimal_cue_da` on Dynaword,
`minimal_cue_en` on Common Pile and Dolma3), unconditional generations against every corpus of
the model. Groups select by tag: a model family (`dfm`), model (`dfm-main`), run
(`dfm-main-dynaword`), corpus (`dolma3`) or setting (`unconditional`).

```bash
python memorization_experiment/run_memorization_experiments.py dfm-main-dynaword
python memorization_experiment/run_memorization_experiments.py olmo3-32b prefix
```

Outputs go to `memorization_experiment/data/propme/<model>/<corpus>/st_<setting>_{results,summary}.json`.

Prompt sets with more than 1000 prompts are traced on a random sample of 1000 (seed 42), with all
generations of each sampled prompt: their generations are read from `<stem>_generations_1000.json`
and their traces written as `st_<setting>_1000_{results,summary}.json`. The samples are built by
`generation/subsample_prompt_sets.py`; set `SAMPLE_SIZE` in `generation/generation_runs.py` to `None` to trace the full sets.

Index paths default to `$PROPME_DATA_ROOT/indexes/...` (`propme_data/` in the repository root if unset), and the
DFM9 indexes to `$PROPME_DFM9_ROOT/indexes/{A,B,C,D}` (`dfm9_data/` if unset).

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

Each run has one preset, named after the run, comparing its five non-prefix settings with its
prefix setting. The comparison presets (`dfm-stages-dynaword`, `dfm-stages-commonpile`,
`dfm-dynaword-vs-commonpile`, `commonpile-dfm-vs-comma`, one per setting) compare a setting
of some runs with the same setting of a reference run instead of the prefix setting.

```bash
python 05_propensity_metrics/compute_propensity_metrics.py dfm-main-dynaword --plot
python 05_propensity_metrics/compute_propensity_metrics.py all-comparisons --plot
```

## Plotting code

Use `plot_memorization_results.py` for per-suite plots:

Each run has one suite (named after the run) with its five settings, and each comparison one
suite per setting (e.g. `dfm-stages-dynaword-prefix`):

```bash
python memorization_experiment/scripts/plot_memorization_results.py dfm-main-dynaword
```

Alongside the existing metric and distribution plots, each suite produces:

- `span_length_distribution_heatmap.png`, an annotated view of the binned span ratios.
- `memorization_metrics_spider.png`, comparing full-generation matches, average NV recall, and average longest span. The span axis is normalized to the 100-token plotting scale.

Use `plot_comparison_overviews.py` for cross-model and cross-stage comparison views:

```bash
python memorization_experiment/scripts/plot_comparison_overviews.py dfm-stages-dynaword
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
  --indexes-root $PROPME_DFM9_ROOT/indexes \
  --prepared-data-root $PROPME_DFM9_ROOT/dfm9_memorisation_sources_propme \
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

# Memorization Experiments

This folder contains the experiment code and input assets used to run the `SimpleTrace` and `PropMe` memorization evaluations in this repository.

## Code in this folder

The scripts in this folder are:

- `generate_vllm.py`: batch prompt inference through a running OpenAI-compatible vLLM server.
- `sample_docs.py`: sample source documents from an InfiniGram index into `*_sample_docs.jsonl`.
- `commonpile_extract_prefixes.py`: build Common Pile prefix prompts from sampled source documents.
- `dynaword_extract_prefixes.py`: build Dynaword prefix prompts from sampled source documents.

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

Use `plot_comparison_overviews.py` for cross-model and cross-stage comparison views:

```bash
python memorization_experiment/plot_comparison_overviews.py dynaword-stages-comparison
```

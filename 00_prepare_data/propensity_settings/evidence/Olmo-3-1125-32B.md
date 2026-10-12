---
license: apache-2.0
language:
- en
library_name: transformers
datasets:
- allenai/dolma3_mix-5.5T-1125
model-index:
- name: Olmo-3-1125-32B
  results:
  - task:
      type: text-generation
    dataset:
      name: Benchmarks
      type: benchmark
    metrics:
    - name: Olmo 3-Eval Math
      type: olmo_3_eval_math
      value: 61.6
    - name: BigCodeBench
      type: bigcodebench
      value: 43.9
    - name: HumanEval
      type: humaneval
      value: 66.5
    - name: DeepSeek LeetCode
      type: deepseek_leetcode
      value: 1.9
    - name: DS 1000
      type: ds_1000
      value: 29.7
    - name: MBPP
      type: mbpp
      value: 60.2
    - name: MultiPL HumanEval
      type: multipl_humaneval
      value: 35.9
    - name: MultiPL MBPPP
      type: multipl_mbppp
      value: 41.8
    - name: Olmo 3-Eval Code
      type: olmo_3_eval_code
      value: 40.0
    - name: ARC MC
      type: arc_mc
      value: 94.7
    - name: MMLU STEM
      type: mmlu_stem
      value: 70.8
    - name: MedMCQA MC
      type: medmcqa_mc
      value: 57.6
    - name: MedQA MC
      type: medqa_mc
      value: 53.8
    - name: SciQ MC
      type: sciq_mc
      value: 95.5
    - name: Olmo 3-Eval MC_STEM
      type: olmo_3_eval_mc_stem
      value: 74.5
    - name: MMLU Humanities
      type: mmlu_humanities
      value: 78.3
    - name: MMLU Social Sci.
      type: mmlu_social_sci.
      value: 83.9
    - name: MMLU Other
      type: mmlu_other
      value: 75.1
    - name: CSQA MC
      type: csqa_mc
      value: 82.3
    - name: PIQA MC
      type: piqa_mc
      value: 85.6
    - name: SocialIQA MC
      type: socialiqa_mc
      value: 83.9
    - name: CoQA Gen2MC MC
      type: coqa_gen2mc_mc
      value: 96.4
    - name: DROP Gen2MC MC
      type: drop_gen2mc_mc
      value: 87.2
    - name: Jeopardy Gen2MC MC
      type: jeopardy_gen2mc_mc
      value: 92.3
    - name: NaturalQs Gen2MC MC
      type: naturalqs_gen2mc_mc
      value: 78.0
    - name: SQuAD Gen2MC MC
      type: squad_gen2mc_mc
      value: 98.2
    - name: Olmo 3-Eval MC_Non-STEM
      type: olmo_3_eval_mc_non_stem
      value: 85.6
    - name: HellaSwag RC
      type: hellaswag_rc
      value: 84.8
    - name: Winogrande RC
      type: winogrande_rc
      value: 90.3
    - name: Lambada
      type: lambada
      value: 75.7
    - name: Basic Skills
      type: basic_skills
      value: 93.5
    - name: DROP
      type: drop
      value: 81.0
    - name: Jeopardy
      type: jeopardy
      value: 75.3
    - name: NaturalQs
      type: naturalqs
      value: 48.7
    - name: SQuAD
      type: squad
      value: 94.5
    - name: CoQA
      type: coqa
      value: 74.1
    - name: Olmo 3-Eval GenQA
      type: olmo_3_eval_genqa
      value: 79.8
    - name: BBH
      type: bbh
      value: 77.6
    - name: MMLU Pro MC
      type: mmlu_pro_mc
      value: 49.6
    - name: Deepmind Math
      type: deepmind_math
      value: 30.1
    - name: LBPP
      type: lbpp
      value: 21.7
    source:
      name: Model README
      url: https://huggingface.co/allenai/Olmo-3-1125-32B
---

## Model Details
<img alt="Logo for Olmo 3 32B Base model" src="olmo-base.png" width="211px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Model Card for Olmo 3 32B

We introduce Olmo 3, a new family of 7B and 32B models. This suite includes Base, Instruct, and Think variants. The Base models were trained using a staged training approach.

Olmo is a series of **O**pen **l**anguage **mo**dels designed to enable the science of language models. 
These models are trained on the Dolma 3 dataset. We are releasing all code, checkpoints, and associated training details. 

| Size   | Training Tokens | Layers | Hidden Size | Q Heads | KV Heads | Context Length |
|--------|-----------------|--------|-------------|---------|----------|----------------|
| [OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | 5.93 Trillion | 32 | 4096 | 32 | 32 | 65,536 |
| [OLMo 3 32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | 5.50 Trillion | 64 | 5120 | 40 | 8 | 65,536 |


The core models released in this batch include the following:

| **Stage**               | **Olmo 3 7B Think** | **Olmo 3 32B Think** | **Olmo 3 7B Instruct** |
|--------------------------|-----------------------|------------------------|---------------------------|
| **Base Model**           | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) |
| **SFT**                  | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) |
| **DPO**                  | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) |
| **Final Models (RLVR)**  | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) |           
                                 

## Installation

Olmo 3 is supported in transformers v4.57.0 or higher:
```bash
pip install transformers>=4.57.0
```

## Inference

You can use OLMo with the standard HuggingFace transformers library:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=0, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
>> 'Language modeling is  a key component of any text-based application, but its effectiveness...'
```

For faster performance, you can quantize the model using the following method:
```python
AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B", 
    torch_dtype=torch.float16, 
    load_in_8bit=True)  # Requires bitsandbytes
```
The quantized model is more sensitive to data types and CUDA operations. To avoid potential issues, it's recommended to pass the inputs directly to CUDA using:
```python
inputs.input_ids.to('cuda')
```

We have released checkpoints for these models. For pretraining, the naming convention is `stage1-stepXXX`.  The conventions for midtraining and long context are `stage2-ingredientY-stepXXX` and `stage3-stepXXX`, respectively.


To load a specific model revision with HuggingFace, simply add the argument `revision`:
```bash
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B", revision="stage1-step10000")
```

Or, you can access all the revisions for the models via the following code snippet:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-1125-32B")
branches = [b.name for b in out.branches]
```

### Fine-tuning
Model fine-tuning can be done from the final checkpoint (the `main` revision of this model) or many intermediate checkpoints. Two recipes for tuning are available.
1. Fine-tune with the OLMo-core repository:
```bash
torchrun --nproc-per-node=8 ./src/scripts/official/OLMo3/OLMo-3-1025-32B-pretrain.py run01
```
You can override most configuration options from the command-line. For example, to override the learning rate you could launch the script like this:

```bash
torchrun --nproc-per-node=8 ./src/scripts/official/OLMo3/OLMo-3-1025-32B-pretrain.py run01 --train_module.optim.lr=6e-4
```
For more documentation, see the [GitHub readme](https://github.com/allenai/OLMo-core).

### Model Description

- **Developed by:** Allen Institute for AI (Ai2)
- **Model type:** a Transformer style autoregressive language model.
- **Language(s) (NLP):** English
- **License:** The code and model are released under Apache 2.0.
- **Contact:** Technical inquiries: `olmo@allenai.org`. Press: `press@allenai.org`
- **Date cutoff:** Dec 2024


### Model Sources

- **Project Page:** https://allenai.org/olmo
- **Repositories:** 
    - Core repo (training, inference, fine-tuning etc.): https://github.com/allenai/OLMo-core
    - Evaluation code: https://github.com/allenai/OLMo-Eval
    - Further fine-tuning code: https://github.com/allenai/open-instruct
- **W&B Report:** https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw
- **Paper:** https://allenai.org/papers/olmo3
<!-- - **Technical blog post:** (URL)  -->
<!-- - **W&B Logs:** [pretraining](), [annealing]() -->


## Evaluation
Core model results for MODELS are found below.

| Model | Olmo 3-Eval Math | BigCodeBench | HumanEval | DeepSeek LeetCode | DS 1000 | MBPP | MultiPL HumanEval | MultiPL MBPPP | Olmo 3-Eval Code | ARC MC | MMLU STEM | MedMCQA MC | MedQA MC | SciQ MC | Olmo 3-Eval MC_STEM | MMLU Humanities | MMLU Social Sci. | MMLU Other | CSQA MC | PIQA MC | SocialIQA MC | CoQA Gen2MC MC | DROP Gen2MC MC | Jeopardy Gen2MC MC | NaturalQs Gen2MC MC | SQuAD Gen2MC MC | Olmo 3-Eval MC_Non-STEM | HellaSwag RC | Winogrande RC | Lambada | Basic Skills | DROP | Jeopardy | NaturalQs | SQuAD | CoQA | Olmo 3-Eval GenQA | BBH | MMLU Pro MC | Deepmind Math | LBPP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Open-weight Models** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Qwen-2.5-32B | 64.7 | 48.1 | 65.6 | 8.0 | 43.3 | 69.8 | 49.7 | 53.6 | 48.3 | 97.0 | 79.7 | 68.8 | 68.4 | 97.1 | 82.2 | 85.0 | 88.4 | 81.2 | 89.9 | 93.3 | 86.6 | 96.8 | 86.6 | 97.0 | 79.9 | 97.9 | 89.3 | 86.3 | 87.5 | 76.2 | 94.2 | 53.7 | 74.0 | 39.3 | 64.9 | 40.4 | 68.5 | 81.1 | 61.1 | 40.7 | 40.3 |
| Gemma-3-27B | 63.2 | 44.0 | 62.1 | 5.8 | 34.3 | 60.0 | 37.7 | 47.2 | 41.6 | 95.8 | 74.9 | 64.7 | 68.7 | 96.8 | 80.2 | 80.5 | 86.2 | 80.2 | 79.0 | 90.3 | 81.2 | 95.8 | 84.6 | 95.9 | 82.0 | 97.7 | 86.7 | 86.0 | 91.3 | 77.5 | 94.9 | 75.9 | 82.1 | 49.2 | 92.4 | 12.4 | 73.5 | 77.4 | 53.1 | 30.4 | 17.7 |
| Mistral-3.1-24B | 59.5 | 46.4 | 65.5 | 0.1 | 36.3 | 61.9 | 39.0 | 47.7 | 42.4 | 96.2 | 70.1 | 68.8 | 70.4 | 96.3 | 81.5 | 82.7 | 88.6 | 81.9 | 80.5 | 91.0 | 81.0 | 94.9 | 86.5 | 97.2 | 84.6 | 97.9 | 87.9 | 86.2 | 90.8 | 79.3 | 91.9 | 74.9 | 80.3 | 45.1 | 92.6 | 61.1 | 78.0 | 81.4 | 58.9 | 35.3 | 30.3 |
| Seed-36B | 15.3 | 50.7 | 71.3 | 13.0 | 44.0 | 72.0 | 69.2 | 63.8 | 54.9 | 97.3 | 82.8 | 69.6 | 70.1 | 97.1 | 83.4 | 85.7 | 90.1 | 82.4 | 81.1 | 92.5 | 84.9 | 96.9 | 90.1 | 96.2 | 81.4 | 98.1 | 89.0 | 84.8 | 89.3 | 76.1 | 96.0 | 76.1 | 77.4 | 30.7 | 89.1 | 64.4 | 76.0 | 85.0 | 62.2 | 31.3 | 42.6 |
| Gemma-2-27B | 57.5 | 43.4 | 57.5 | 4.7 | 29.7 | 61.7 | 40.3 | 49.7 | 41.0 | 94.1 | 65.8 | 61.8 | 61.0 | 95.1 | 75.6 | 79.3 | 85.8 | 76.9 | 78.1 | 89.0 | 81.0 | 94.3 | 66.6 | 92.0 | 74.5 | 97.5 | 83.2 | 86.7 | 90.8 | 76.9 | 93.2 | 73.2 | 80.7 | 47.1 | 93.0 | 14.9 | 72.9 | 74.8 | 47.6 | 27.6 | 19.7 |
| Llama-3.1-70B | 62.0 | 43.4 | 57.4 | 0.2 | 29.5 | 55.5 | 32.2 | 35.9 | 36.3 | 95.2 | 70.0 | 67.8 | 72.3 | 95.4 | 80.1 | 83.4 | 87.4 | 79.4 | 79.0 | 91.5 | 83.5 | 95.1 | 70.3 | 97.1 | 82.4 | 97.7 | 86.1 | 88.4 | 91.7 | 79.6 | 92.4 | 78.3 | 84.0 | 53.1 | 92.9 | 73.9 | 81.6 | 80.8 | 50.4 | 40.3 | 11.8 |
| **Fully-open Models** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Marin-32B | 49.3 | 34.5 | 52.3 | 1.3 | 26.3 | 52.1 | 18.5 | 30.5 | 30.8 | 93.4 | 68.4 | 61.8 | 60.8 | 95.1 | 75.9 | 78.9 | 83.7 | 75.4 | 80.1 | 90.5 | 82.4 | 93.9 | 71.0 | 95.3 | 81.0 | 97.6 | 84.5 | 87.2 | 90.5 | 76.7 | 91.1 | 76.5 | 80.5 | 55.1 | 94.4 | 70.7 | 80.3 | 70.1 | 48.1 | 26.7 | 17.3 |
| Apertus-70B | 39.7 | 24.0 | 32.5 | 1.2 | 17.8 | 37.6 | 18.4 | 31.3 | 23.3 | 90.7 | 57.8 | 55.9 | 52.4 | 93.3 | 70.0 | 74.1 | 79.2 | 70.1 | 76.9 | 79.0 | 79.3 | 87.5 | 56.5 | 93.2 | 71.9 | 95.7 | 78.5 | 84.5 | 87.7 | 74.8 | 87.5 | 56.3 | 77.2 | 43.1 | 90.7 | 72.8 | 75.0 | 58.8 | 39.6 | 20.1 | 8.1 |
| OLMo 2-32B | 53.9 | 22.2 | 29.4 | 0.8 | 20.4 | 37.1 | 10.5 | 23.2 | 20.5 | 94.4 | 64.7 | 60.2 | 62.2 | 95.1 | 75.3 | 79.7 | 84.5 | 75.6 | 81.2 | 87.7 | 82.3 | 94.4 | 68.6 | 96.6 | 78.6 | 97.4 | 84.2 | 87.5 | 89.4 | 77.0 | 88.7 | 76.3 | 79.1 | 51.4 | 94.0 | 68.7 | 79.1 | 64.6 | 46.9 | 22.0 | 8.2 |
| **Olmo 3-32B** | 61.6 | 43.9 | 66.5 | 1.9 | 29.7 | 60.2 | 35.9 | 41.8 | 40.0 | 94.7 | 70.8 | 57.6 | 53.8 | 95.5 | 74.5 | 78.3 | 83.9 | 75.1 | 82.3 | 85.6 | 83.9 | 96.4 | 87.2 | 92.3 | 78.0 | 98.2 | 85.6 | 84.8 | 90.3 | 75.7 | 93.5 | 81.0 | 75.3 | 48.7 | 94.5 | 74.1 | 79.8 | 77.6 | 49.6 | 30.1 | 21.7 |

## Model Details

#### Stage 1: Initial Pretraining
- Dataset: [dolma3_mix-5.5T-1125](https://huggingface.co/datasets/allenai/dolma3_mix-5.5T-1125)
- 5.50T tokens
- Coverage: 94.83%+ of total pretraining budget

#### Stage 2: Mid-training
- Ingredient 1
  - Dataset: [dolma3-dolmino-mix-1125](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125)
  - 100B tokens
  - Mix composition: web pages, code, math/QA/thinking/instruction/PDFs
- Ingredient 2
  - Dataset: [dolma3-dolmino-mix-1125](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125)
  - 100B tokens
  - Mix composition: web pages, code, math/QA/thinking/instruction/PDFs

#### Stage 3: Long Context
- Dataset: [dolma3-longmino-mix-1125](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125)
- 100B tokens
- Mix composition: midtraining data and PDFs

#### Model Merging
- 7B Model: No merging
- 32B Model: 2 versions on 100B mix, merged before starting long context run.  Final checkpoint is merged 4 final checkpoints.

## Bias, Risks, and Limitations
Like any base language model or fine-tuned model without safety filtering, these models can easily be prompted by users to generate harmful and sensitive content. Such content may also be produced unintentionally, especially in cases involving bias, so we recommend that users consider the risks when applying this technology. Additionally, many statements from OLMo or any LLM are often inaccurate, so facts should be verified.

## License
This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

## Citation
A technical manuscript is forthcoming! Find the paper at: https://allenai.org/papers/olmo3

## Model Card Contact
For errors in this model card, contact `olmo@allenai.org`.
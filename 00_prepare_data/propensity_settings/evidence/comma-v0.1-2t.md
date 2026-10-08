---
license: apache-2.0
datasets:
- common-pile/comma_v0.1_training_dataset
language:
- en
---
# Comma v0.1-2T

Comma v0.1-2T is a 7 billion parameter language model trained on 2 trillion tokens from [the Comma v0.1 dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset), comprising of openly licensed text from [the Common Pile](https://huggingface.co/collections/common-pile/common-pile-v01-68307d37df48e36f02717f21).
Comma v0.1-2T is a "base model" that can be used a the starting point for finetuning and post-training.
It performs comparably to budget-matched models (7 billion parameters, 2 trillion tokens) trained on unlicensed data.

| Model         | ARC-C | ARC-E | MMLU | BoolQ | HSwag | OBQA | CSQA | PIQA | SIQA | HEval | MBPP | Avg. |
| ------------- | ----- | ----- | ---- | ----- | ----- | ---- | ---- | ---- | ---- | ----- | ---- | ---- |
| OLMo Twin | 45.2 | 67.5 | 28.2 | 71.7 | 73.4 | 48.0 | 61.8 | 77.9 | 48.5 | 18.2 | 27.5 | 51.6 |
| Llama 2 | 48.5 | 69.5 | 45.8 | 80.2 | 76.2 | 48.4 | 62.8 | 76.7 | 50.8 | 26.1 | 28.5 | 55.8 |
| Comma v0.1 2T | 45.8 | 71.8 | 49.8 | 78.6 | 64.4 | 46.2 | 64.0 | 72.5 | 52.3 | 44.2 | 41.5 | 57.4 |
| DeepSeekLLM | 49.5 | 67.7 | 48.5 | 71.7 | 74.1 | 52.0 | 66.6 | 77.8 | 51.6 | 43.1 | 43.8 | 58.8 |

## Training details

Comma v0.1-2T is a decoder-only transformer that uses the same architecture as Llama 3.
Training was done in two stages: first on 1.93 trillion tokens with a cosine learning rate schedule, and second a "cool-down" training phase on 75.5 billion tokens from high-quality sources.
The final model is the average of 10 checkpoints during this cool-down phase. Both training phases use a batch size of 8.3 million tokens per step.
Training was performed using [lingua](https://github.com/facebookresearch/lingua/) on 512 AMD MI300A GPUs.
Hyperparameters can be found in our [lingua config file](https://huggingface.co/common-pile/comma-v0.1-2t-checkpoints/blob/main/config.yaml).

## Limitations

Comma v0.1-2T was trained only on English-language data and code from the 15 programming languages covered by the [stack-edu classifiers](https://huggingface.co/collections/HuggingFaceTB/the-ultimate-collection-of-code-classifiers-67b5aa3eb8994a4b71453005).
It will likely have poor performance on other languages or programming languages.
While we aimed to train solely on openly licensed data, license laundering and inaccurate metadata can result in erroneous license information in the Common Pile (for further discussion of this limitation, please see [our paper](https://huggingface.co/papers/2506.05209)).
Consequently, we cannot make a guarantee that Comma v0.1-2T was trained exclusively on openly licensed text.
When preparing Comma v0.1's pre-training data, we made use of the Toxicity tagger from [Dolma](https://github.com/allenai/dolma) to attempt to remove problematic content.
However, Comma v0.1-2T may nevertheless reflect social biases present in its training data.
Finally, please note that Comma v0.1-2T is a base model that has not undergone any form of "alignment" and therefore has no guardrails that limit what it may generate.

## Citation

```bibtext
@article{kandpal2025common,
  title={{The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text}},
  author={Nikhil Kandpal and Brian Lester and Colin Raffel and Sebastian Majstorovic and Stella Biderman and Baber Abbasi and Luca Soldaini and Enrico Shippole and A. Feder Cooper and Aviya Skowron and Shayne Longpre and Lintang Sutawika and Alon Albalak and Zhenlin Xu and Guilherme Penedo and Loubna Ben  and Elie Bakouch and John David  and Honglu Fan and Dashiell Stander and Guangyu Song and Aaron Gokaslan and John Kirchenbauer and Tom Goldstein and Brian R and Bhavya Kailkhura and Tyler Murray},
  journal={arXiv preprint},
  year={2025}
}
```
---
license: apache-2.0
datasets:
- danish-foundation-models/danish-dynaword
- common-pile/comma_v0.1_training_dataset
language:
- da
- en
base_model:
- common-pile/comma-v0.1-2t
pipeline_tag: text-generation
---

# DFM-Decoder-open-v0-7b-pt

DFM-Decoder-open-v0-7b-pt is a 7-billion-parameter [open-source](https://opensource.org/ai/open-source-ai-definition) language model. 
DFM-Decoder-open-v0-7b-pt is a base model that can serve as a starting point for fine-tuning and post-training. 
It has not been instruction-tuned and cannot directly be expected to function as a chat model.

| Model | Model Weights | Training Data | Training Code |
|:------|:--------------|:--------------|:--------------|
| Llama | Public with custom license | Private | Private |
| Gemma | Public, openly licensed | Private | Private |
| Apertus | Public, openly licensed | Reproducible, license unspecified | Public, openly licensed |
| **DFM-Decoder-open-v0-7b-pt** (ours) | **Public, openly licensed** | **Public, openly licensed** | **Public, openly licensed** |

## Evaluation

### Performance on Danish

The following plots show the model size on the x-axis and an aggregate performance score for Danish on the y-axis. Each metric is normalized across all evaluated models using min-max normalization to the range [0, 1], and the final score represents the average of all normalized metrics.

<p align="center">
  <img src="./plots/danish-perf.png" width="700"/>
</p>

DFM-Decoder-open-v0-7b-pt was evaluated using the [EuroEval](https://euroeval.com/) framework, which includes benchmarks across seven task types covering more than 15 European languages.

Below we report results for Danish (see English below) for all EuroEval-supported tasks: sentiment classification, named entity recognition, linguistic acceptability, reading comprehension, summarization, and knowledge and common-sense reasoning. In addition, we evaluate the model on DaLA, a Danish linguistic acceptability dataset focusing on real-world common errors.

We compare DFM-Decoder-open-v0-7b-pt at various training stages with its base model [Comma v0.1-2T](https://huggingface.co/common-pile/comma-v0.1-2t) 
and two models from the Pleias family ([Pleias-350M-Preview](https://huggingface.co/PleIAs/Pleias-350m-Preview) and [Pleias-1.2B-Preview](https://huggingface.co/PleIAs/Pleias-1.2b-Preview)). 
All comparison models were trained exclusively on open data, either in the public domain or under a permissive license.

The following tables show the performance on each dataset. 
For each, we report the respective main metric from EuroEval and the confidence interval. 
The latter is calculated as the mean of the metric scores across all evaluation runs ± 1.96 times the standard error of the mean.

| Model                               | scala-da (MCC)| dala (MCC)    | angry-tweets (MCC) | dansk (Micro F1, No Misc) | danske-talemaader (MCC) | danish-citizen-tests (MCC) | multi-wiki-qa-da (F1) | hellaswag-da (MCC) | nordjylland-news (BERTScore) | average  |
| ----------------------------------- | ------------- | ------------- | ------------------ | ------------------------- | ----------------------- | -------------------------- | --------------------- | ------------------ | ---------------------------- | -------  |
| base (comma-v0.1-2t)                | 0.9 ± 0.8     | 0.2 ± 0.6     | 39.8 ± 1.4         | 32.0 ± 2.8                | 3.6 ± 2.3               | 10.7 ± 4.1                 | 66.4 ± 0.8            | 3.8 ± 1.0          | 60.2 ± 1.7                   | 24.2     |
| **Training Stages**                 |               |               |                    |                           |                         |                            |                       |                    |                              |          |
| stage 1 | 13.3 ± 2.9    | 12.7 ± 2.2    | **47.7** ± 1.7     | 40.0 ± 2.4                | 18.1 ± 0.9              | 32.8 ± 1.4                 | **76.6** ± 0.6        | 12.9 ± 1.0         | 66.3 ± 0.7                   | 35.6     |
| stage 2 | 15.8 ± 3.1    | 14.4 ± 2.9    | 47.4 ± 2.3         | 40.4 ± 2.4                | 24.1 ± 1.8              | 36.1 ± 1.8                 | 75.2 ± 0.7            | 13.1 ± 1.1         | 66.5 ± 0.6                   | 37.0     |
| dfm-decoder-open-v0-7b-pt (stage 3) | **16.5** ± 1.4| **15.7** ± 1.7| 46.3 ± 2.1         | **41.1** ± 2.8            | **24.6** ± 2.0          | **36.2** ± 1.7             | 76.0 ± 0.7            | **13.2** ± 1.2     | **66.6** ± 0.6               | **37.4** |
| **Baselines**                       |               |               |                    |                           |                         |                            |                       |                    |                              |          |
| Pleias-350m-Preview                 | -1.0 ± 1.5    | -1.8 ± 1.8    | 10.6 ± 2.9         | 12.9 ± 1.8                | 0.7 ± 2.6               | 4.6 ± 2.3                  | 11.6 ± 0.9            | -0.3 ± 0.7         | 56.3 ± 1.5                   | 10.4     |
| Pleias-1.2b-Preview                 | 0.2 ± 1.1     | 0.7 ± 1.0     | 27.7 ± 2.9         | 27.3 ± 2.2                | -0.6 ± 1.9              | 8.6 ± 3.2                  | 35.2 ± 1.3            | -0.0 ± 1.5         | 60.3 ± 0.9                   | 17.7     |

### Performance on English

<div align="center">
<table>
  <tr>
    <td><img src="./plots/danish-english-perf.png" width="600"/></td>
    <td><img src="./plots/english-perf.png" width="600"/></td>
  </tr>
</table>
</div>

The goal of this section is to demonstrate how performance is maintained deteriorates for English when adapting the model for Danish. Generally, we see only minor performance degradation 
across tasks. 

| Model                                | scala-en (MCC) | sst5 (MCC)    | conll-en (Micro F1 no misc) | life-in-the-uk (MCC) | squad (F1)    | hellaswag (MCC) | cnn-dailymail (BERTScore) | average   |
| ------------------------------------ | -------------  | ------------  | --------------------------- | -------------------- | ------------  | --------------- | ------------------------- | -------   |
| base (comma-v0.1-2t)                 | **29.7** ± 1.9 | **61.8** ± 2.1| **57.5** ± 2.8              | 41.6 ± 2.4           | **90.4** ± 0.4| **16.8** ± 0.6  | **63.3** ± 0.9            | **51.6**  |
| **Training Stages**                  |                |               |                             |                      |               |                 |                           |           |
| stage 1  | 17.1 ± 9.0     | 60.0 ± 1.7    | 56.6 ± 2.2                  | 40.5 ± 1.7           | 90.1 ± 0.3    | 13.7 ± 0.7      | 59.6 ± 1.3                | 48.2      |
| stage 2  | 27.7 ± 2.0     | 59.5 ± 1.6    | 56.6 ± 2.3                  | 41.2 ± 1.7           | 90.2 ± 0.4    | 16.0 ± 0.9      | 60.3 ± 1.6                | 50.2      |
| dfm-decoder-open-v0-7b-pt (stage 3)  | 29.0 ± 2.4     | 60.3 ± 1.4    | 56.9 ± 2.5                  | **41.7** ± 1.8       | 89.9 ± 0.4    | 13.8 ± 0.9      | 59.2 ± 1.7                | 50.1      |
| **Baseline**                         |                |               |                             |                      |               |                 |                           |           |
| Pleias-350m-Preview                  | 0.7 ± 1.8      | 15.4 ± 7.3    | 31.8 ± 3.5                  | -0.7 ± 2.1           | 31.1 ± 2.3    | 0.2 ± 1.4       | 53.8 ± 1.0                | 18.9      |
| Pleias-1.2b-Preview                  | 1.0 ± 2.4      | 48.2 ± 2.6    | 40.9 ± 3.3                  | 2.6 ± 2.8            | 52.9 ± 2.5    | -0.1 ± 1.5      | 60.2 ± 1.6                | 29.4      |

## Training details

DFM-Decoder-open-v0-7b-pt is continually pre-trained from [Comma v0.1-2T](https://huggingface.co/common-pile/comma-v0.1-2t) using 30B tokens, utilizing a mix of [Danish Dynaword 1.2.12](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword) and the [Comma v0.1 dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset), both comprising only public domain and openly licensed data.

DFM-Decoder-open-v0-7b-pt has been trained using the [maester](https://github.com/rlrs/maester) framework developed as part of [Danish Foundation Models](https://foundationmodels.dk/). All training was performed on a single 8x NVIDIA B200 node (the first of its kind in Denmark) as part of the [SDU UCloud](https://cloud.sdu.dk/) research cloud.

The training was performed in three stages, with data mix (open-stageK.py) and maester (open-stageK.toml) configuration files available in each subfolder. The datasets can be created using the `create_dataset.py` script provided in this repository.

The characteristics of the three pre-training stages are detailed in the following table:

| Stage | Batch size (tokens) | Steps | HF path | Data mix | Comments |
|-|-|-|-|-|-|
| stage 1 | 262,144 | 37,852| [revision="stage1"](https://huggingface.co/danish-foundation-models/munin-7b-open-pt/tree/stage1)  | 2/3 [Dynaword](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword/tree/9e230b35e31a510e5ab909112ad5bfc9463b2c23); <br> 1/3 [Common-Pile](https://huggingface.co/common-pile/comma_v0.1_training_dataset/5afc546db324e7f39f297ba757c9a60547151e7c) | Excludes depbank, jvj, nordjyllandnews, synne for Dynaword; <br> uses subsets and weighting from [Comma-v0.1-2T](https://huggingface.co/common-pile/comma-v0.1-2t) cooldown phase for Common-Pile ; LR schedule with 1000 steps warmup, constant 1e-5, 1000 steps cooldown |
| stage 2 | 524,288 | 18,926 | [revision="stage2"](https://huggingface.co/danish-foundation-models/munin-7b-open-pt/tree/stage2)  | 2/3 [Dynaword](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword/tree/9e230b35e31a510e5ab909112ad5bfc9463b2c23); <br> 1/3 [Common-Pile](https://huggingface.co/common-pile/comma_v0.1_training_dataset/5afc546db324e7f39f297ba757c9a60547151e7c) | Excludes depbank, jvj, nordjyllandnews, synne for Dynaword; <br> uses subsets and weighting from [Comma-v0.1-2T](https://huggingface.co/common-pile/comma-v0.1-2t) cooldown phase for Common-Pile; LR schedule with 500 steps warmup, constant 1e-5, 500 steps cooldown |
| stage 3 | 524,288 | 18,926 | [revision="stage3"](https://huggingface.co/danish-foundation-models/munin-7b-open-pt/tree/stage3)  | 2/3 [Dynaword](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword/tree/9e230b35e31a510e5ab909112ad5bfc9463b2c23); <br> 1/3 [Common-Pile](https://huggingface.co/common-pile/comma_v0.1_training_dataset/5afc546db324e7f39f297ba757c9a60547151e7c) | Excludes depbank, jvj, nordjyllandnews, synne for Dynaword; <br> uses subsets and weighting from [Comma-v0.1-2T](https://huggingface.co/common-pile/comma-v0.1-2t) cooldown phase for Common-Pile; LR schedule with 500 steps warmup, square root decay from 1e-5 |

## Limitations

DFM-Decoder-open-v0-7b-pt was trained only on Danish and English-language data and code from the 15 programming languages covered by the [stack-edu classifiers](https://huggingface.co/collections/HuggingFaceTB/the-ultimate-collection-of-code-classifiers-67b5aa3eb8994a4b71453005).
It will likely have poor performance on other languages or programming languages.

As a base model, DFM-Decoder-open-v0-7b-pt has not been aligned for safety and may, for example, reflect social biases present in its training data or potentially provide toxic or harmful information.

## License

The model is made available under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) open source license. It may therefore be used, modified, distributed, and sublicensed for any purpose, including commercial use, without the licensee having to release their own derivative works under the same permissive terms, provided that users retain copyright and license notices and document any modifications they make.

## Project partners & funding

The development of DFM-Decoder-open-v0-7b-pt was performed in a close collaboration between [Aarhus University](https://chc.au.dk/), the [Alexandra Institute](https://alexandra.dk/), and the [University of Southern Denmark](https://www.sdu.dk/en/forskning/machine-learning) as part of [Danish Foundation Models](https://foundationmodels.dk/).

Funding was provided by the [Danish Ministry of Digital Affairs](https://www.english.digmin.dk/) and the [Danish Ministry of Higher Education and Science](https://ufm.dk/en).

## How to cite

Coming soon.
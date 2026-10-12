---
license: odc-by
task_categories:
- text-generation
language:
- en

configs:
  - config_name: default
    data_files:
      - split: train
        path: data/**/*
---

⚠️ **IMPORTANT NOTICE** ⚠️

This is the Dolma 3 *pool*, pre–quality upsampling and mixing. 
If you are interested in *the data used* to train [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) and [Olmo 3 32B](https://huggingface.co/allenai/Olmo-3-1025-32B), visit [**allenai/dolma3_mix-6T-1025**](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025). 


-----

<img alt="Logo for Dolma Pool" src="dolma-pool.png" width="301px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Dolma 3 Pool
The Dolma 3 pool is a dataset of over 9 trillion tokens from a diverse mix of web content, academic publications, code, and more. For detailed documenation on Dolma 3 processing and data, please see our [Dolma 3 Github repository](https://github.com/allenai/dolma3/blob/main/datasets/dolma3_mix/pools/9T/README.md#9t-pretraining-pool). For more information on Dolma in general, please see our original release [here](https://huggingface.co/datasets/allenai/dolma).


### A Note on the Dolma 3 Pool: Source Links
The dolma 3 pool contains documents for Common Crawl (web) and olmOCR Science PDFs **only**. To access the documents from the remaining sources in this pool, follow the source links below:

- **Common Crawl**: Current repository
- **olmOCR Science PDFs**: Current repository
- **StackEdu**: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- **arXiv**: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
- **FineMath 3+**: https://huggingface.co/datasets/HuggingFaceTB/finemath
- **Wikipedia & Wikibooks**: https://huggingface.co/datasets/allenai/dolma (dolma v1.7)

## Dataset Sources

This dataset contains the full pool of documents considered to train the first stage of Olmo 3 7B.

| Source | Type | 9T Pool Tokens | 9T Pool Docs |
|--------|------|----------------|--------------|
| Common Crawl | Web pages | 8.14T | 9.67B |
| olmOCR Science PDFs | Academic documents | 972B | 101M |
| StackEdu (Rebalanced) | GitHub code | 137B | 167M |
| arXiv | Papers with LaTeX | 21.4B | 3.95M |
| FineMath 3+ | Math web pages | 34.1B | 21.4M |
| Wikipedia & Wikibooks | Encyclopedic | 3.69B | 6.67M |
| **Total** | | **9.31T** | **9.97B** |


## Downloading Dolma 3

You can download and load this data using HuggingFace's `datasets` library with the following code:

```python
from datasets import load_dataset
dataset = load_dataset("allenai/dolma3_pool", split="train",)
```

You can further specify a specific split of the dataset to load. In this repository, Common Crawl data folders are foramtted as `common_crawl-topic-vigintile`. Similarly, olmOCR PDF data folders are formatted as `olmocr_science_pdfs-topic`. For example:

```python
from datasets import load_dataset
dataset = load_dataset("allenai/dolma3_pool", 
                        data_files="data/olmocr_science_pdfs-*/*.jsonl.zst",
                        split="train")
```

Note: You can iterate over over the dataset directly without having to download the entire dataset. Simply set `streaming=True` in the command above.

### Licensing Information

Dolma 3 is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

### Citation

```
@misc{olmo2025olmo3,
title={Olmo 3},
author={Team Olmo and Allyson Ettinger and Amanda Bertsch and Bailey Kuehl and David Graham and David Heineman and Dirk Groeneveld and Faeze Brahman and Finbarr Timbers and Hamish Ivison and Jacob Morrison and Jake Poznanski and Kyle Lo and Luca Soldaini and Matt Jordan and Mayee Chen and Michael Noukhovitch and Nathan Lambert and Pete Walsh and Pradeep Dasigi and Robert Berry and Saumya Malik and Saurabh Shah and Scott Geng and Shane Arora and Shashank Gupta and Taira Anderson and Teng Xiao and Tyler Murray and Tyler Romero and Victoria Graf and Akari Asai and Akshita Bhagia and Alexander Wettig and Alisa Liu and Aman Rangapur and Chloe Anastasiades and Costa Huang and Dustin Schwenk and Harsh Trivedi and Ian Magnusson and Jaron Lochner and Jiacheng Liu and Lester James V. Miranda and Maarten Sap and Malia Morgan and Michael Schmitz and Michal Guerquin and Michael Wilson and Regan Huff and Ronan Le Bras and Rui Xin and Rulin Shao and Sam Skjonsberg and Shannon Zejiang Shen and Shuyue Stella Li and Tucker Wilde and Valentina Pyatkin and Will Merrill and Yapei Chang and Yuling Gu and Zhiyuan Zeng and Ashish Sabharwal and Luke Zettlemoyer and Pang Wei Koh and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
year={2025},
eprint={2512.13961},
archivePrefix={arXiv},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2512.13961},
}
```

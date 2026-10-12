---
license: odc-by
language:
- en

configs:
  - config_name: default
    data_files:
      - split: train
        path: data/**/*
    features:
      - name: id
        dtype: string
      - name: text
        dtype: string
      - name: metadata
        dtype: string
      - name: source
        dtype: string
      - name: version
        dtype: string
      - name: created
        dtype: string
      - name: added
        dtype: string
      - name: doc
        dtype: string
      - name: attributes
        dtype: string
---


<img alt="Logo for Dolmino Mix" src="dolmino-mix.png" width="289px" style="margin-left:'auto' margin-right:'auto' display:'block'">

# Dolma 3 Dolmino dataset pool for Olmo 3 stage 2 annealing training 

This dataset contains the high-quality pool of data considered for the second stage of Olmo 3 32B. 

## Dataset Sources

| Source | Category |
|--------|----------|
| TinyMATH Mind | Math (synth) |
| TinyMATH PoT | Math (synth) |
| CraneMath | Math (synth) |
| MegaMatt | Math (synth) |
| Dolmino Math | Math (synth) |
| StackEdu (FIM) | Code |
| CraneCode | Python (synth) |
| Reddit To Flashcards | QA (synth) |
| Wiki To RCQA | QA (synth) |
| Nemotron Synth QA | QA (synth) |
| Math Meta-Reasoning | Thinking (synth) |
| Code Meta-Reasoning | Thinking (synth) |
| Program-Verifiable | Thinking (synth) |
| OMR Rewrite FullThoughts | Thinking (synth) |
| QWQ Reasoning Traces | Thinking (synth) |
| General Reasoning Mix | Thinking (synth) |
| Gemini Reasoning Traces | Thinking (synth) |
| Llama Nemotron Reasoning Traces | Thinking (synth) |
| OpenThoughts2 Reasoning Traces | Thinking (synth) |
| Tulu 3 SFT | Instruction (synth) |
| Dolmino 1 Flan | Instruction (synth) |
| OLMOCR Science PDFs (High Q.) | PDFs |
| STEM-Heavy Crawl | Web pages |
| Common Crawl (High Q.) | Web pages |

---

## Ingredients
There were two ingredients used during stage 2 midtraining annealling of Olmo 3 32B. There were 2 versions of a 100B mix:
- Ingredient 1
  - 100B tokens
  - Mix composition: web pages, code, math/QA/thinking/instruction/PDFs
- Ingredient 2
  - 100B tokens
  - Mix composition: web pages, code, math/QA/thinking/instruction/PDFs


## Licensing Information

Dolma 3 Dolmino is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

## Citation

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
---
license: odc-by
language:
- en

configs:
  - config_name: default
    data_files:
      - split: train
        path: data/**/*
---

<img alt="Logo for Longmino Mix" src="longmino-mix.png" width="326px" style="margin-left:'auto' margin-right:'auto' display:'block'">


# Dolma 3 Longmino Mix (100B)

The Dolma 3 Longmino Mix (100B) is the mixture of data used for the third stage of training for Olmo 3 32B model. 

## Dataset Sources

| Source | Type |
|--------|------|
| LC-s2pdf-REX 32k-64k | Synth PDFs |
| LC-s2pdf-CWE 32k-64k | Synth PDFs |
| LC-s2pdf 32k-64k | PDFs |
| LC-s2pdf 8k-32k (8-16k) | PDFs |
| LC-s2pdf 8k-32k (16-32k) | PDFs |
| Midtraining Data | Mix |



## Licensing Information

Dolma 3 Longmino is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

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
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
        path: data/**/*.jsonl.zst
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

---

<img src="https://cdn-uploads.huggingface.co/production/uploads/65316953791d5a2611426c20/JopP0oxXQlhiB7YHQGZhY.png" width="300" alt="dolma-mix">


# Dolma 3 Mix (6T)
The Dolma 3 Mix (6T) is the collection of data used during the pretraining stage to train the Olmo-3-1125-32B model. This dataset is made up of ~6 trillion tokens from a diverse mix of web content, academic publications, code, and more. The majority of this dataset comes from Common Crawl.  

For more information on Dolma, please see our original release [here](https://huggingface.co/datasets/allenai/dolma).

## Smaller Sample for Analysis Available!
If you would like a smaller sample of this mix to examine and experiment with, which uses the same upsampling strategy, please see our 150B mix: https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025.

## Licensing Information
Dolma 3 mix is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

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

Find the paper at: https://allenai.org/papers/olmo3
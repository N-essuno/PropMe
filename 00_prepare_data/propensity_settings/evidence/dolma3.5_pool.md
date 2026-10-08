---
license: odc-by
task_categories:
- text-generation
language:
- en

configs:
  - config_name: cc_all_dressed
    data_files:
      - split: train
        path: cc_all_dressed/**/*
  - config_name: cc_all_dressed_addons
    data_files:
      - split: train
        path: cc_all_dressed_addons/**/*
  - config_name: common_pile_code
    data_files:
      - split: train
        path: common_pile_code/**/*
  - config_name: common_pile_textbook
    data_files:
      - split: train
        path: common_pile_textbook/**/*
  - config_name: common_pile_wiki
    data_files:
      - split: train
        path: common_pile_wiki/**/*
  - config_name: dolma4pdfs
    data_files:
      - split: train
        path: dolma4pdfs/**/*
  - config_name: dolma_code
    data_files:
      - split: train
        path: dolma_code/**/*
  - config_name: dolma_code_prose
    data_files:
      - split: train
        path: dolma_code_prose/**/*
  - config_name: finemath
    data_files:
      - split: train
        path: finemath/**/*
  - config_name: hplt
    data_files:
      - split: train
        path: hplt/**/*
  - config_name: proofpile2
    data_files:
      - split: train
        path: proofpile2/**/*
  - config_name: sponge211unified_code_prose
    data_files:
      - split: train
        path: sponge211unified_code_prose/**/*
  - config_name: swallow-code
    data_files:
      - split: train
        path: swallow-code/**/*
  - config_name: swallow-math
    data_files:
      - split: train
        path: swallow-math/**/*
  - config_name: the-stack-v2
    data_files:
      - split: train
        path: the-stack-v2/**/*
---

⚠️ **IMPORTANT NOTICE** ⚠️

This is the Dolma 3.5 *pool*. It contains no quality upsampling or mixing. This is an updated version of the Dolma 3 pool with additional quality filtering and more data sources.
If you are interested in *the data used* to train [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) and [Olmo 3 32B](https://huggingface.co/allenai/Olmo-3-1025-32B), visit [**allenai/dolma3_mix-6T-1025**](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025). 


-----


# Dolma 3.5 Pool
The Dolma 3.5 pool is a dataset of nearly 10 trillion tokens from a diverse mix of web content, academic publications, code, and more. For detailed documenation on the previous version of Dolma 3 and its processing and data, please see our [Dolma 3 Github repository](https://github.com/allenai/dolma3/blob/main/datasets/dolma3_mix/pools/9T/README.md#9t-pretraining-pool). For more information on Dolma in general, please see our original release [here](https://huggingface.co/datasets/allenai/dolma).

# Changes from Dolma 3

## (1) PDFs
We improve on the olmOCR science PDFs Data Pool from Dolma 3. We split this data source into three sub components, and make improvements to each one before running deduplication globally across all three
components to produce a final pool focusing on high-quality mostly academically-oriented content.

#### a. olmo-crawled-pdfs 
This pool mirrors the crawled PDFs from Dolma 3. We updated our crawling cutoff date to November 2025, and reprocess all of the selected PDFs using olmOCR-2-7B-1025-FP8. This version of the olmOCR model has higher quality, especially on math and table data, and scores 82.4 points on
olmOCR-bench compared to 68.4 for the release that was used in Dolma 3.

#### b. s2orcforolmo 
The pool explicitly selects for academically oriented PDFs with logic similar to the original peS2o/S2ORC set. These PDFs are also processed using olmOCR-2-7B-1025-FP8. After OCR, we annotated each PDF with a Field of Study (FOS) fasttext classifer. To prune out large blocks
of references, a second fasttext classifier was used, and lines detected as references were removed (available in datamap-rs). No additional filtering was done, as this subset was already pre-selected to include high quality academic
papers.

#### c. FinePDFs 
We include 138 million English language documents from FinePDFs ([Kydlíček et al., 2025](https://github.com/huggingface/finepdfs)) and tag them with both [WebOrganizer (Wettig et al. (2025))](https://huggingface.co/WebOrganizer) category, and a quality score. This pool was seeded using the documents in the Fine PDFs eng_Latn subset. A
minhash deduplication was applied first to just this subset. We then tag this set with WebOrganizer categories,
but otherwise do very minimal processing.

## (2) Code
For Dolma 3.5, we expand the code pool by over 9 times, going from 137 billion tokens in Dolma 3 to over 1,351 billion tokens. Further, we increase coverage to more programming languages, going from 15 in
Dolma 3 to 50 in Dolma 3.5. We achieve this by re-processing [The Stack v2 dataset (Lozhkov et al., 2024)](https://huggingface.co/datasets/bigcode/the-stack-v2) from scratch rather than relying on [Stack-Edu (Allal et al., 2025)](https://huggingface.co/datasets/HuggingFaceTB/stack-edu). Beside code repositories, we also consider code data from web-crawled prose, which is a new source of code data for Dolma 3.5. 

### Code Sources

#### a. Swallow Code v2
We include the Python subset of Swallow Code v2, an LLM-rewritten Python corpus derived from The Stack v2 using Qwen3-235B-A22B-Instruct. Swallow Code v2 contains 49.8
billion tokens of Python code and is released under the Apache 2.0 license. Compared to its predecessor, it
removes the restrictive Llama community license and substantially increases scale. We use the medium-quality
filtered subset, which has additionally been decontaminated and n-gram filtered.

#### b. Web Code Prose
We describe web code prose as a document containing both natural prose and code. We
identify code and prose at line granularity, such that any given line is classified as either code, prose, or other.
We track the composition of documents by counting observed line classes. We define a code-prose document
as a document containing at least 5% code lines, 30% prose lines, with a minimum of 10 lines of code present.

We use fastText to classify each line of web documents as either code, prose, or other. The
classifier is trained on a collection of code, prose, and code-prose heavy documents.
Code documents were collected for each programming language by manually selecting some open source
popular projects. Prose documents were collected manually from public domain books. And other documents
were generated for common web-text patterns such as dates, numeric sequences, short word sequences, and
isolated punctuation

#### c. Common Pile Code Text 
We include several code-adjacent text sources from Common Pile, collectively referred to as Common Pile Code Text. These sources contain natural language discussions of 9 programming topics rather than raw source code, complementing the Stack v2 pool with conversational and
explanatory content:

- **GitHub Archive** Pull request discussions and code review comments from public GitHub repositories.

- **Stack Exchange** Programming-related questions and answers covering a broad range of languages and topics.

- **Ubuntu IRC** Chat logs from Ubuntu IRC channels, providing informal technical discussion data.

All three subsets have been filtered for license compatibility, decontaminated, and n-gram filtered prior to
inclusion

# Previous Version: Dolma 3
For more details on Dolma 3, please see:
- Pool: https://huggingface.co/datasets/allenai/dolma3_pool
- Mix: https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B

## Licensing Information

Dolma 3.5 is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our [Responsible Use Guidelines](https://allenai.org/responsible-use).

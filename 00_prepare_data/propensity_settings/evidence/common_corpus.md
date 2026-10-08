---
language:
- en
- fr
- de
- zh
- it
- es
- ja
- pl
- la
- nl
- ru
- ar
- ko

configs:
  - config_name: default
    data_files:
      - split: train
        path: "common_corpus_1/subset_100_1.parquet"
---


# Common Corpus

<p align="center">
  <a href="https://iclr.cc/virtual/2026/poster/10011885"><b>Full paper - ICLR 2026 oral</b></a>
</p>

Common Corpus is the largest open licensed text dataset, comprising 2.27 trillion tokens (2,267,302,720,836 tokens). It is a diverse dataset, consisting of books, newspapers, scientific articles, government and legal documents, code, and more. Common Corpus has been created by Pleias in association with several partners.

Common Corpus differs from existing open datasets in that it is:
*  **Truly Open**: contains only data that is either uncopyrighted or freely licensed
*  **Traceable**: each individual document is associated with documented contextual information, including licensed use or lack of copyright.
*  **Multilingual**: mostly representing English and French data, but contains data for 8 languages with more than 10 billion tokens (German, Spanish, Italian, Polish, Greek, Latin) and 33 languages with more than 1 billion tokens.
*  **Diverse**: consisting of scientific articles, government and legal documents, code, and cultural heritage data, including books and newspapers
*  **Extensively Curated**: spelling and formatting has been corrected from digitized texts, harmful and toxic content has been removed, and content with low educational content has also been removed.

The dataset in its entirety meets the requirements of the Code of Conduct of the AI Act and goes further than the current requirements for data transparency. It aims to set a new standard of openness in AI, showing that detailed provenance at a granular document level is a realistic objective, even at the scale of 2.3 trillion tokens.

Common Corpus makes it possible to train model compatible with [the Open Source Initiative’s definition](https://opensource.org/ai/open-source-ai-definition#:~:text=An%20Open%20Source%20AI%20is,including%20to%20change%20its%20output.) of open-source AI, which includes openness of use, meaning use is permitted for “any purpose and without having to ask for permission". Based on the available licensing information Common Corpus can be filtered to only include public domain works or a subset of free licenses (like attribution only).


# About Common Corpus

Common Corpus is made of six carefully curated collections:
*  **OpenCulture**: our largest collection at 967,018,390,906 tokens, featuring public domain books, newspapers from cultural heritage repositories and open projets like Wikisource ad Gutenberg. We're developing innovative tools of OCR correction based on Pleias Models to correct historical digitization errors, while implementing advanced toxicity filtering to ensure content meets modern ethical standards.
*  **OpenGovernment**: 579,150,518,908 tokens of financial and legal documents, including Finance Commons (from sources like SEC and WTO) and Legal Commons (including Europarl, Caselaw Access Project, Chinese Case Law), providing enterprise-grade training data from regulatory bodies and administrative sources.
*  **OpenSource**: 283,227,402,898 tokens of high-quality code in open source from GitHub, filtered using ArmoRM to ensure only the top 80% of submissions by quality rating are included.
*  **OpenScience**: 281,193,563,789 tokens of academic content from Open Alex and other open science reposiories, processed using vision-language models to preserve crucial document structure and formatting.
*  **OpenWeb**: 88,517,032,065 tokens from Wikipedia (official releases from the [Wikimedia Foundation](https://huggingface.co/datasets/wikimedia/wikipedia) on Huggingface), YouTube Commons and Stack-Exchange.
*  **Open Semantic**: 67,958,671,827 tokens from Wikidata (official releases from the [Wikimedia Foundation](https://huggingface.co/datasets/wikimedia/wikipedia) on Huggingface). The data has been reprocessed thanks to support and help of Wikidata and Wikimedia Germany. It includes the transcriptions of all the semantic triplets into natural language statements in over 300 languages.

| Collection 	| Domain               	| Sources                                                                               	|
|----------------|--------------------------|-------------------------------------------------------------------------------------------|
| OpenGovernment | legal and administrative | [Finance Commons](https://huggingface.co/collections/PleIAs/finance-commons-66925e1095c7fa6e6828e26c) (e.g. SEC, WTO) and Legal Commons (e.g. Europarl, Caselaw Access Project, Chinese CaseLaw) |
| OpenCulture	| cultural heritage    	| public domain books and newspapers, Wikisource                                                    	|
| OpenScience	| academic             	| OpenAlex                                                             	|
| OpenWeb    	| web text             	| [YouTube Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons), MOSEL, Stack Exchange, CCCC                                                       	|
| OpenSource 	| code                 	| GitHub                                                                                	|
| OpenSemantic 	| Semantic data                 	| Wikidata                                                                                	|

The first version of [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) was released in November of 2024.  The second version added Wikidata and detailed document-level information, including licensing and other core metadata whenever available. The third ongoing version dramatically expand the language coverage of Common Corpus beyond the US and Europe with the integration of large collection of documents in Chinese, Japanese, Arabic, Korean and Hindi.

The dataset release is accompanied by a comprehensive technical report (ICRL 2026 - oral) detailing our methodologies and data sources will accompany the release, ensuring full transparency and reproducibility.

## Dataset Structure

<details >
  <summary>Data Fields</summary>

*  `identifier`: unique text identifier. In many cases, this is also the link to the original resources.
*  `collection`: name of one of the XX sub-collections curated for Common corpus.
*  `open type`: one of the six leading collection groupings:
*  `license`: sharing rights for the content either uncopyrighted (public domain, US federal public domain, CC0 on Wikidata) or various free licenses (Creative Commons, MIT, French Licence ouverte, etc.)
*  `date`: date of creation of the resource where known. Due to the significance of public domain and other cultural heritage content, more than half of Common Corpus predates the 21st century.
*  `title`: title of the resource when known or alternatively the filename.
*  `creator`: institution publishing/collecting/curating the resource.
*  `language`: automatically identified language.
*  `word_count`: number of space delimited words.
*  `token_count`: number of tokens as calculated by Pleias official tokenizer and Gemma-3 tokenizer for Chinese, Japanese, Arabic, Korean and few additional non-Western languages.
*  `text`: full text, without formatting.

</details > 
<br />


## Provenance

The provenance of the datasets that make up Refined Common Corpus is detailed in the technical report [link]. Additionally, the original source URL is available in the metadata for each document for most of the dataset. 


## How to Use

### Considerations for Using the Data

All data in Common Corpus are either uncopyrighted or freely licensed and may be used for both commercial and non-commercial purposes. 

The dataset is multilingual. The language text is included in the metadata, so data can be filtered by language. Additionally, some of the text data are historical. The year each text is written is included in the metadata, therefore it is possible to construct a dataset with a custom date cutoff if desired.

### Discussion of Bias

Some of the dataset sources contain biased and toxic content, such as stereotypes about certain minoritized groups. We have removed texts which had high toxicity scores according to our toxicity classifier, [Celadon](https://huggingface.co/PleIAs/celadon), or which contain offensive terms and slurs. See our [preprint](https://arxiv.org/pdf/2410.22587) for more details.

### Personal and Sensitive Information

We have attempted to remove personally identifiable information (PII). We primarily use [Microsoft Presidio](https://microsoft.github.io/presidio/), but make additional modifications to account for language- and country-specific considerations, such as European phone number formats. 

Some small parts of the French administrative common crawl have been entirely dropped using our unreleased small reasoning model for GDPR-filtering, due to the heightened risk of transmitting identifiable indirect personal information.

## Using Common Corpus
from datasets import load_dataset
data = load_dataset('PleIAs/common_corpus')


# Acknowledgements

The Corpus was built up with the support and concerted efforts of the AI Alliance, the French Ministry of Culture as part of the prefiguration of the service offering of the Alliance for Language technologies EDIC (ALT-EDIC). 

This dataset was also made in partnership with Wikimedia Enterprise and Wikidata/Wikimedia Germany. We're also thankful to our partner Libraries Without Borders for continuous assistance on extending low resource language support.

The corpus was stored and processed with the generous support of the AI Alliance, Jean Zay (Eviden, Idris), Tracto AI, Mozilla. Generation of OCR correction at scale were performed using HPC resources from two GENCI–IDRIS grants: 2023-AD011014736 and GC011015451.

Some parts of the corpus have been built on top of other similar open science LLM community initiatives such as German-Commons, MOSEL, kl3m, AI4Bharat, Creative Commons Common Crawl. We included a new curator field to properly acknowledge this data work.

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/ai_alliance.png" style="width: 33%; margin: 0 auto; display: inline-block;"/>
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/logo-genci-header.svg" style="width: 33%; margin: 0 auto; display: inline-block;"/>
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/Nvidia_(logo).svg.png" style="width: 33%; margin: 0 auto; display: inline-block;"/>   
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/tractoAI.png" style="width: 33%; margin: 0 auto; display: inline-block;"/>
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/mozilla.png" style="width: 33%; margin: 0 auto; display: inline-block;"/>
  <img src="https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/logo/wikimedia_logo.png" style="width: 33%; margin: 0 auto; display: inline-block;"/>
</div>
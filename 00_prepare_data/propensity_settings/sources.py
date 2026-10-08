"""Which data counts as unseen for each tested model, and where it comes from.

Eligibility rule (applies to every corpus and every training stage of its models):
a source is used only if

  (a) its content date (creation, publication, addition to the dataset; never a
      crawl date) is after the model's training data, or
  (b) the training-data documentation says the source was not included.

The verbatim filter against each training index runs on top of this
(run_trace.py), so a sentence that also occurs in the training data is dropped
even when its source is eligible.

Training data of the tested models:

  DFM (dfm-decoder-open-v0-7b-pt, stage1/stage2/main): Danish Dynaword
      @ 9e230b35 (v1.2.12, 2025-09-19) minus depbank, jvj, nordjyllandnews,
      synne (model card), plus common-pile/comma_v0.1_training_dataset @ 5afc546
      (2025-06-06).
  Comma (comma-v0.1-2t): comma_v0.1_training_dataset (all 31 Common Pile
      sources, listed in COMMON_PILE_SOURCES).
  OLMo 3 (Olmo-3-1125-32B): data cutoff Dec 2024. Stage 1 dolma3_mix-5.5T-1125
      (from the Dolma 3 pool: Common Crawl, olmOCR science PDFs, StackEdu,
      arXiv, FineMath, Wikipedia/Wikibooks from dolma v1.7), stage 2
      dolma3_dolmino_mix-100B-1125 (CC high quality, STEM-heavy crawl, olmOCR
      PDFs, synthetic data), stage 3 dolma3_longmino_mix-100B-1125 (s2pdf
      PDFs, midtraining data). The cards are saved in evidence/.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
# Root of the large data (indexes/, raw/), outside the repository by default (as in run_tracing.py).
DATA_ROOT = Path(os.environ.get("PROPME_DATA_ROOT", DATA_DIR.parents[1] / "propme_data"))
RAW_DIR = DATA_ROOT / "raw" / "unseen_specific"
EVIDENCE_DIR = DATA_DIR / "evidence"

FINAL_SIZE = 1000
# Candidates per source, as a multiple of its final share: room for the
# verbatim filter and for the similarity selection (select_prompts.py).
OVERSAMPLE = 4
SEED = 42

DYNAWORD_REPO = "danish-foundation-models/danish-dynaword"
DYNAWORD_1212_REVISION = "9e230b35e31a510e5ab909112ad5bfc9463b2c23"
# Dynaword 1.2.12 subsets left out of DFM training (model card).
DFM_EXCLUDED = ("depbank", "jvj", "nordjyllandnews", "synne")
# Subsets added to Dynaword after 1.2.12 (CHANGELOG v1.2.13-v1.2.25). adl was
# regenerated in v1.2.24 but is in 1.2.12, so it is not here.
DYNAWORD_ADDED_AFTER_1212 = (
    "kb_administrative_publication",
    "kb_historical_letters",
    "municipality_meetings",
    "hvadvilduhelst",
    "tidsskrift-dk",
    "dakultur",
    "mosel_voxpopuli",
    "mosel_youtubecommons",
    "folketingets-dokumenter",
    "kalliope",
    "logir",
)

COMMON_PILE_SOURCES = (
    "arxiv_abstracts", "arxiv_papers", "biodiversity_heritage_library", "caselaw_access_project", "cccc",
    "data_provenance_initiative", "doab", "foodista", "github_archive", "library_of_congress", "libretexts",
    "news", "oercommons", "peS2o", "pre_1929_books", "pressbooks", "project_gutenberg", "public_domain_review",
    "pubmed", "python_enhancement_proposals", "regulations", "stackexchange", "stackv2_edu", "stackv2_html",
    "ubuntu_irc", "uk_hansard", "usgpo", "uspto", "wikimedia", "wikiteam", "youtube",
)
COMMA_DATASET_DATE = "2025-06-06"  # comma_v0.1_training_dataset @ 5afc546
# Common Corpus (PleIAs/common_corpus) English collections whose upstream source
# is not one of the Common Pile sources (rule b). Common Pile's
# library_of_congress is LoC "Selected Digitized Books", not the Chronicling
# America newspapers behind US-PD-Newspapers.
COMMON_CORPUS_ELIGIBLE = (
    "SEC",
    "WTO",
    "GATT_library",
    "Eurlex",
    "Eurovoc",
    "TEDEUTenders",
    "VoxPopuli",
    "dotgov",
    "US-PD-Newspapers",
    "NewZealand-PD-Newspapers",
)
# Science collections overlap peS2o / arXiv / PubMed at the paper level, so
# they count only by content date (rule a): publication date after the Comma
# training dataset.
COMMON_CORPUS_DATED = ("OpenAlex", "Open-Science-Pile", "French-Science-Pile", "Spanish-Science-Pile",
                       "German-Science-Pile")
# Same upstream as a Common Pile source; listed for the README.
COMMON_CORPUS_EXCLUDED = {
    "Wikipedia": "wikimedia",
    "Wiki Discussions": "wikimedia / wikiteam",
    "Wikidata": "wikimedia",
    "StackExchange": "stackexchange",
    "Youtube-Commons": "youtube",
    "USPTO": "uspto",
    "Caselaw Access Project": "caselaw_access_project",
    "Court Listener": "caselaw_access_project",
    "UK Hansard (all)": "uk_hansard",
    "reg_docs": "regulations",
    "dockets": "regulations",
    "govinfo": "usgpo",
    "Creative Commons Common Crawl (CCC)": "cccc",
    "English-PD": "pre_1929_books / project_gutenberg",
    "US-PD-Books": "pre_1929_books",
    "LoC-PD-Books": "library_of_congress",
    "LibriLight": "project_gutenberg (LibriVox reads PD books)",
}

DOLMA35_REPO = "allenai/dolma3.5_pool"
HPLT_TOPICS = (
    "adult_content", "art_and_design", "crime_and_law", "education_and_jobs", "electronics_and_hardare",
    "entertainment", "fashion_and_beauty", "finance_and_business", "food_and_dining", "games", "health",
    "history_and_geography", "home_and_hobbies", "industrial", "literature", "politics", "religion",
    "science_math_and_technology", "social_life", "software", "software_development", "sports_and_fitness",
    "transportation", "travel_and_tourism",
)
# Top quality bins (higher = better) of the Dolma 3.5 HPLT pool.
HPLT_VIGINTILES = tuple(f"vigintile_{i:04d}" for i in range(15, 20))
FINEPDFS_REPO = "HuggingFaceFW/finepdfs"

# New Wikipedia articles (rule a): created on or after the release of the
# training dataset that could hold them, which is later than any snapshot in it.
WIKI_CREATED_AFTER = {
    "dw": ("da", "2025-09-19"),  # Dynaword 1.2.12 commit
    "cp": ("en", COMMA_DATASET_DATE),
    "d3": ("en", "2025-11-20"),  # Olmo 3 1125 release (data cutoff Dec 2024)
}

# corpus -> sources sampled uniformly. Common Corpus science collections are
# added at build time only if enough dated documents exist.
CORPUS_SOURCES = {
    "dw": (*DYNAWORD_ADDED_AFTER_1212, *DFM_EXCLUDED, "wikipedia_new"),
    "cp": (*COMMON_CORPUS_ELIGIBLE, "wikipedia_new"),
    "d3": ("hplt", "finepdfs", "wikipedia_new"),
}
CORPUS_LANG = {"dw": "dan", "cp": "eng", "d3": "eng"}
# Index each corpus's candidates are traced against (run_tracing.INDEXES), plus
# extra indexes checked for verbatim occurrences only (DFM also saw Comma).
CORPUS_INDEX = {"dw": "dynaword1212", "cp": "commonpile", "d3": "dolma3_split"}
CORPUS_EXTRA_INDEXES = {"dw": ("commonpile",), "cp": (), "d3": ()}
# Prompt-set corpus suffix -> dataset folder in memorization_experiment/data (as run_tracing.PROMPT_SET_DATASETS).
CORPUS_DATASET = {"cp": "commonpile", "d3": "dolma3", "dw": "dynaword2"}
# Corpus name of each corpus's training data in scripts/embedding_similarity.py.
CORPUS_EMBEDDING = {"cp": "commonpile", "d3": "dolma3", "dw": "dynaword1212"}
# Minimum probability of the corpus language (lang_id.py) for a selected prompt.
LANGID_MIN_P_TARGET = 0.2

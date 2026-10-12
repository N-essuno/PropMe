# Propensity settings: generic and specific prompt sets

The two ordinary-use (propensity) settings of PropMe, built per training corpus (`cp` Common Pile, `d3` Dolma 3, `dw` Dynaword):

| setting | prompts | built by | file in `memorization_experiment/data/` |
|---|---|---|---|
| `generic_<x>` | Tatoeba sentences that do not occur verbatim in the training index | `build_samples.py`, `run_tracing.py`, `group_by_overlap.py`, `build_prompt_sets.py` | `<dataset>/generic/generic_prompts.jsonl` |
| `specific_<x>` | sentences from sources the tested models have not seen, of Tatoeba length and closer to the training data than Tatoeba | `sources.py`, `download_sources.py`, `wikipedia_new.py`, `build_candidates.py`, `trace_candidates.py`, `build_unseen_prompt_sets.py` | `<dataset>/specific/specific_prompts.jsonl` |

`<dataset>` is `commonpile`, `dolma3` or `dynaword2`. Both settings use the same overlap logic (`run_tracing.py`, `group_by_overlap.py`), so a prompt is kept only if it is not a full match in its training index (overlap groups 1-5, see [Overlap groups](#overlap-groups)).

All commands run from the repository root. The large data (indexes and raw downloads) lives under `$PROPME_DATA_ROOT` (default: `propme_data/` in the repository root): indexes in `$PROPME_DATA_ROOT/indexes/`, raw downloads for the specific sets in `$PROPME_DATA_ROOT/raw/unseen_specific/`. Steps marked GPU need `torch` (and vLLM for the embedding steps); the others need `infini-gram`.

## Generic prompt sets (Tatoeba)

2k-sentence samples of Tatoeba, with their Llama-2 token lengths and their overlap with the English and Danish training indexes.

| set | source | sampling |
|---|---|---|
| `tatoeba_eng_2k.jsonl` | [Tatoeba eng export](https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2) | 2,000 of 2,038,137 unique sentences |
| `tatoeba_dan_2k.jsonl` | [Tatoeba dan export](https://downloads.tatoeba.org/exports/per_language/dan/dan_sentences_detailed.tsv.bz2) | 2,000 of 66,540 unique sentences |

Sampling uses seed 42. Texts are deduplicated before sampling. Each line has `id`, `text`, `source`, `lang` and `n_tokens` (Llama-2 tokens without BOS).

### Reproduce

```bash
# Download the raw files into 00_prepare_data/propensity_settings/raw/
for l in eng dan; do
  curl -fL -o 00_prepare_data/propensity_settings/raw/${l}_sentences_detailed.tsv.bz2 \
    https://downloads.tatoeba.org/exports/per_language/$l/${l}_sentences_detailed.tsv.bz2
done

# Sample the sets and compute token lengths -> tatoeba_<lang>_2k.jsonl, length_stats.json
python 00_prepare_data/propensity_settings/build_samples.py

# Trace against the indexes -> traces/, tracing_overview.{json,md}
python 00_prepare_data/propensity_settings/run_tracing.py --num-workers 8

# Overlap groups -> groups/groups_<set>_<index>.jsonl, groups/groups_overview.{json,md}
python 00_prepare_data/propensity_settings/group_by_overlap.py --threads 64

# Prompt sets -> memorization_experiment/data/<dataset>/generic/generic_prompts.jsonl, prompt_sets/README.md
python 00_prepare_data/propensity_settings/build_prompt_sets.py

# Overlap metrics of the prompt texts -> prompt_sets/prompt_metrics.{json,md}
python 00_prepare_data/propensity_settings/prompt_set_metrics.py
```

`run_tracing.py` traces English sets against Common Pile and Dolma3 and Danish sets against Dynaword. Dolma3 defaults to the combined symlinked index (`dolma3_index_link`); `--indexes dolma3_split` loads the 27 separate split indexes instead (the same list as `scripts/test_simpletrace_dolma3.py`) and writes its outputs under the `dolma3_split` name. Use `--indexes` to trace every selected set against other indexes (e.g. a cross-language check), `--sets` to pick sets, and `--summary-only` to rebuild the overview. Pairs that already have outputs are skipped unless `--overwrite` is passed. `--num-workers` and `--find-threads` set the SimpleTrace processes and their lookup threads; `--verbatim-threads` (default 16) sets how many whole-sentence lookups the verbatim pass runs at once.

### Overlap metrics

Each (set, index) pair gets two passes.

**Verbatim lookup** (`traces/verbatim_<set>_<index>.jsonl`): an exact count of how often each whole sentence occurs in the index. The sentence is looked up under two tokenizations, because Llama-2 marks the leading space on the first token (`▁Jeg`), while the same sentence after a newline, quote or dash starts without it (`J`, `eg`). Looking up only the first form, as SimpleTrace's k-eidetic evaluation does, misses every sentence that sits on its own line, such as subtitle lines. On 100 Tatoeba-dan sentences against Dynaword it found 8 instead of 17.

- `verbatim_found_ratio`: fraction of sentences occurring at least once.
- `verbatim_found_ge_10x_ratio`: fraction occurring at least 10 times.
- `verbatim_found_spaced_only_ratio`: the same as `verbatim_found_ratio` but with only the leading-space form, to show the effect.
- `verbatim_median_count_when_found`: median occurrence count among found sentences.

**SimpleTrace** (`traces/st_<set>_<index>_{results.jsonl,summary.json}`), in `mixed` mode with 10 docs per span (the same settings as the memorization experiments) and SimpleTrace's default `--seed 42`, so reruns give identical results:

- `generations_full_matches_ratio`: fraction of sentences where a retrieved document contains the full sentence as a substring. This is a lower bound on verbatim presence, because it only checks the up to 10 retrieved docs per span.
- `generations_with_spans_ratio`, `average_longest_span_length`: how much of each sentence is covered by matched spans.
- `avg_longest_span_coverage`: mean of (longest span / sentence length). `longest_span_coverage_ge_0.5_ratio` is the fraction of sentences where the longest span covers at least half the sentence.
- `generations_with_10_token_span_ratio`: fraction of sentences with a matched span of at least 10 tokens. The threshold is lower than the 50-70 used for generations, because these sentences are short.
- `avg_nv_recall`, `generations_with_nv_recall_ratio`, `generations_above_nv_recall_0.5_ratio`: NV recall, as in the main README.

### Overlap groups

`group_by_overlap.py` assigns every sentence of each (set, index) pair that has a verbatim lookup to one group:

| group | rule |
|---|---|
| 1 | longest matched span < 25% of the sentence |
| 3 | longest matched span 25-50% (50 excluded) |
| 4 | longest matched span 50-75% (75 excluded) |
| 5 | longest matched span >= 75%, but the full sentence does not occur |
| 6 | full sentence occurs, in more than k distinct documents |
| 7 | full sentence occurs, in at most k distinct documents |

- **Full sentence** (a full match) means the whole text occurs in the index: found by the verbatim lookup (either tokenization), or as a document or substring of a document that SimpleTrace retrieved for it. The second catches occurrences tokenized differently at the edges, e.g. a final `?` merged with a following quote into `?"`. `full_match_source` says which found it; the SimpleTrace check reads `traces/st_<set>_<index>_results.jsonl`.
- **Longest matched span** is recomputed exactly instead of read from the SimpleTrace results, which keep only spans of at least 4 tokens and the rarest ceil(5% of length) of them. It is the longest token span, starting at any position (SimpleTrace's unfiltered step 1), that occurs verbatim in the index, taken over both tokenizations, each divided by its own token count. Spans can start or end mid-word.
- **Coverage is in Llama-2 tokens.** `coverage_chars`, the character share of the same span, is stored as well; Danish splits into more tokens than English, so it compares better across languages.
- **Distinct documents** are counted only for full matches occurring more than k times (occurrences bound documents from above), stopping at k + 1, so `distinct_docs` is null for the others and at most k + 1. `--exact-doc-counts` counts them for every full match. `--k` sets k (default 100).

### Prompt sets

`build_prompt_sets.py` turns the groups into the generic prompt sets, stored with the other experiment data as `memorization_experiment/data/<dataset>/generic/generic_prompts.jsonl`: for each index, the Tatoeba sentences in groups 1-5 (not full matches). It only needs `groups/`, not the index. `prompt_sets/README.md` lists sizes, lengths and group breakdowns, and `prompt_sets/prompt_metrics.md` the SimpleTrace metrics of the prompt texts themselves.

## Specific prompt sets (unseen sources)

1000 prompts per training corpus (Common Pile, Dolma 3, Dynaword): whole sentences of Tatoeba length, from text the tested models have not seen and closer to their training data than Tatoeba. Each prompt is

1. **from an unseen source:** a source eligible under the rule below;
2. **not verbatim in the training data:** overlap groups 1-5 against the model's training index, using the same logic as the generic sets, and no occurrence in any other index the model was trained on;
3. **of Tatoeba length:** one sentence per document, matched to the token lengths of `generic_<x>`.

Each set as a whole is **more similar to the training data than Tatoeba**. This holds both on Qwen3-Embedding-8B and on bge-m3. The prompts were selected for similarity against an independent sample of the training data, not the one the evaluation uses.

| prompt set | file in `memorization_experiment/data/` | for |
|---|---|---|
| `specific_dw` | `dynaword2/specific/specific_prompts.jsonl` | DFM (stage1, stage2, main) |
| `specific_cp` | `commonpile/specific/specific_prompts.jsonl` | DFM, Comma |
| `specific_d3` | `dolma3/specific/specific_prompts.jsonl` | OLMo 3 |

In the embedding evaluation and in `results.md` these sets are named `unseen_<x>`. Full results are in `results.md`, written by `report.py`. It compares the sets with the 1000-prompt Tatoeba sets used in the experiments (`generic/generic_prompts_1000.jsonl`). `results_full_generic.md` (`report.py --generic full`) does the same against the full generic sets.

### Results

Mean max_sim (Qwen3-Embedding-8B; cosine to the nearest of 10k random training documents, seed 42), difference to Tatoeba with 95% bootstrap CI:

| set | prompts | sources | avg tokens (Tatoeba) | max_sim (Tatoeba) | difference [95% CI] | bge-m3 difference [95% CI] | in-distribution sentences |
|---|---|---|---|---|---|---|---|
| `specific_dw` | 1000 | 16 | 15.3 (12.5) | 0.499 (0.453) | +0.046 [+0.042, +0.050] | +0.009 [+0.006, +0.012] | 0.573 |
| `specific_cp` | 1000 | 11 | 12.9 (10.7) | 0.485 (0.461) | +0.024 [+0.020, +0.027] | +0.005 [+0.002, +0.008] | 0.517 |
| `specific_d3` | 1000 | 3 | 12.8 (10.7) | 0.485 (0.463) | +0.022 [+0.019, +0.026] | +0.013 [+0.009, +0.016] | 0.489 |

- **Other comparisons:** the sets also beat the 1000-prompt Tatoeba sets (`generic_prompts_1000.jsonl`) on max_sim, topk_mean and mean_sim with both models. The max_sim differences are +0.043 (dw), +0.022 (cp) and +0.021 (d3) with Qwen3, and +0.012, +0.009 and +0.024 with bge-m3, all with 95% CIs above 0 (`results.md`). The same holds against the full generic sets (`results_full_generic.md`).
- **Length matching:** the candidates were length-matched to the full `generic_prompts.jsonl`. Its 1000-prompt random sample has the same length distribution (means within 0.3 tokens, same quartiles), so the sets need no rebuilding.
- **Length:** the specific sets are 1-2 tokens longer on average, because short whole sentences are rarer than short Tatoeba ones. Within the same length bins (1-10 and 11-20 tokens, ~90% of the prompts) they still score higher than Tatoeba with both models.
- **Verbatim check:** `build_unseen_prompt_sets.py verify` found 0 of the 3000 prompts in their training indexes (and for Dynaword also in Common Pile).
- **Similarity selection:** before selection, the Common Pile and Dolma 3 candidates were no more similar than Tatoeba (mean max_sim 0.460 and 0.459, against 0.461 and 0.463 for Tatoeba). The eligible sources are by construction ones the training data lacks, which is what makes the selection necessary. The Dynaword candidates were already a little closer (0.468 vs 0.453). The selection ranks prompts within each source, so the source shares stay uniform.
- **Small sources:** jvj (11), synne (18), GATT_library (16) and TEDEUTenders (2) have too few usable documents and are fully included. Their unused share went to the other sources of their corpus.

### Eligibility rule

A source is used only if

- **(a)** its **content date** is after the model's training data: the date the content was created, published or added to the dataset. A crawl date never counts, because a 2025 crawl can repeat an older page that was in training. Or:
- **(b)** the training-data documentation says the source was **not included**.

The rule covers every training stage of each model. For OLMo 3 that is pretraining (`dolma3_mix-5.5T-1125`), midtraining (Dolmino) and long context (Longmino). The model and dataset cards it relies on are saved in `evidence/`.

| model | training data |
|---|---|
| DFM (`dfm-decoder-open-v0-7b-pt`, stage1/stage2/main) | Danish Dynaword @ `9e230b35` (v1.2.12, 2025-09-19) minus depbank, jvj, nordjyllandnews and synne, plus `comma_v0.1_training_dataset` @ `5afc546` (2025-06-06) |
| Comma (`comma-v0.1-2t`) | `comma_v0.1_training_dataset` (31 Common Pile sources) |
| OLMo 3 (`Olmo-3-1125-32B`) | data cutoff Dec 2024. Dolma 3 pool: Common Crawl, olmOCR science PDFs, StackEdu, arXiv, FineMath, Wikipedia/Wikibooks from dolma v1.7. Dolmino: CC high quality, STEM-heavy crawl, olmOCR PDFs, synthetic data. Longmino: s2pdf PDFs, midtraining data. |

### Sources

All sources of a corpus are sampled uniformly. A source with too few usable documents gives all of them, and its unused share is spread evenly over the others (water-filling). The proposed per-source document counts were ignored.

| corpus | source | rule | notes |
|---|---|---|---|
| Dynaword (`dw`) | kb_administrative_publication, kb_historical_letters, municipality_meetings, hvadvilduhelst, tidsskrift-dk, dakultur, mosel_voxpopuli, mosel_youtubecommons, folketingets-dokumenter, kalliope, logir | (b) | added to Dynaword after 1.2.12 (CHANGELOG v1.2.13-v1.2.25), so not in DFM's training data. Taken from Dynaword @ main. adl was regenerated in v1.2.24 but is in 1.2.12, so it is not used. |
| | depbank, jvj, nordjyllandnews, synne | (b) | in 1.2.12, but the DFM model card says they were excluded from training |
| | wikipedia_new | (a) | da.wikipedia articles created after 2025-09-19 (the Dynaword 1.2.12 commit) |
| Common Pile (`cp`) | Common Corpus (`PleIAs/common_corpus`, `language == English`) collections SEC, WTO, GATT_library, Eurlex, Eurovoc, TEDEUTenders, VoxPopuli, dotgov, US-PD-Newspapers, NewZealand-PD-Newspapers | (b) | their upstream source is none of the 31 Common Pile sources. Common Pile's `library_of_congress` is LoC "Selected Digitized Books", not the Chronicling America newspapers. |
| | wikipedia_new | (a) | en.wikipedia articles created after 2025-06-06 (the `comma_v0.1_training_dataset` commit) |
| Dolma 3 (`d3`) | HPLT (`allenai/dolma3.5_pool/hplt`, quality vigintiles 15-19) | (b) | the Dolma 3.5 card lists HPLT as a source added after Dolma 3. It is in none of the OLMo 3 stages. |
| | FinePDFs (`HuggingFaceFW/finepdfs`, `eng_Latn`) | (b) | OLMo 3's PDFs are Ai2's own olmOCR crawl. FinePDFs is in no stage. |
| | wikipedia_new | (a) | en.wikipedia articles created after 2025-11-20 (OLMo 3 1125 release; the documented cutoff is Dec 2024) |

Excluded:

- **Common Corpus collections with a Common Pile upstream:** Wikipedia, Wiki Discussions, Wikidata, StackExchange, Youtube-Commons, USPTO, Caselaw Access Project, Court Listener, UK Hansard, reg_docs, dockets, govinfo, CCCC, English-PD, US-PD-Books, LoC-PD-Books and LibriLight.
- **Common Corpus science collections** (OpenAlex, *-Science-Pile): papers overlap peS2o, arXiv and PubMed, so only content dated after 2025-06-06 would count, and the downloaded shards have none (the newest is 2023).
- **Dolma 3.5 olmo-crawled-pdfs and s2orcforolmo:** the same kind of content is in Dolma 3, Dolmino and Longmino, and neither carries a content date.
- **FineWeb-Edu 2025 dumps:** they carry only a crawl date, and their upstream (Common Crawl) is in Dolma 3.

**New Wikipedia articles.** Page ids are assigned at creation, in increasing order. The id threshold for a date is the median page id of the first 50 page creations on or after it, from the creation log (one API request). Articles with a higher id, in the main namespace, and not redirects or disambiguation pages, are taken from the `pages-articles-multistream` dumps (dawiki 20261001, enwiki 20260901). Only the dump parts above the threshold are downloaded. Wikitext is converted with mwparserfromhell. See `wikipedia_new.py`.

### Pipeline

1. **Download** (`download_sources.py`, `wikipedia_new.py`):
   - Dynaword @ main (the unseen subsets) and @ 1.2.12.
   - 40 random Common Corpus shards (shards mix all collections).
   - 24 random HPLT top-vigintile shards.
   - 8 random FinePDFs `eng_Latn` files.
   - New Wikipedia articles (dw 2500, cp 1000, d3 6000).
   - The model and dataset cards, into `evidence/`.
2. **Dynaword 1.2.12 index** (`build_dynaword1212.py`): Dynaword @ `9e230b35` minus the 4 excluded subsets (5,536,139 docs), indexed with infini-gram (`--tokenizer llama --add_metadata`) at `$PROPME_DATA_ROOT/indexes/dynaword1212_index`, plus `02_unigram_probs/unigram_probs_dynaword1212.json`. It is registered as `dynaword1212` in `run_tracing.INDEXES`.
   - The older `dynaword` index is a later, unpinned download that contains the post-1.2.12 subsets, so it would mark every unseen Dynaword sentence as seen.
   - Spot check: sentences from wikipedia, retsinformationdk and tv2r are found in it (15/15 each), while sentences from jvj, nordjyllandnews and kb_administrative_publication are not (0/15 each).
3. **Document pools** (`build_candidates.py pool`): up to 4000 random documents per source (8000 for HPLT and FinePDFs), stored as `work/pool_<corpus>.jsonl`. Common Corpus is filtered to English and the eligible collections; HPLT to records whose main language is English.
4. **Quality** (`quality_score.py`, GPU): the FineWeb-Edu classifier scores every English document (first 512 tokens). Documents with an int score ≥ 2 are kept, which is roughly the top 30-45% of each source.
   - The suggested bar of ≥ 3 keeps only 0-16% per source. It would have removed VoxPopuli, GATT, TEDEUTenders, WTO and SEC entirely, because speeches, filings and law are not "educational" text.
   - Danish sources are curated and not scored.
5. **Candidates** (`build_candidates.py sentences`): one whole sentence per document, of Tatoeba length.
   - Target lengths are drawn from the token lengths of the corpus's `generic_<x>` (Tatoeba) prompt set.
   - Each target goes to the next unused document (random order) that has a prose sentence within 1 token of it (10% above 10 tokens). The sentence must also be in the corpus language, judged by a function-word share ≥ 0.15; Danish sentences with þ/ð are rejected.
   - Duplicate documents and texts are dropped.
   - Up to 4× each source's final share is kept.
   - Because the sentences already have Tatoeba lengths, they are not cut.
6. **Verbatim filter** (`trace_candidates.py`): the same logic as the generic sets.
   - SimpleTrace runs in mixed mode with 10 docs per span, followed by `group_by_overlap` groups: occurrences over both Llama-2 tokenizations, SimpleTrace substring matches and the longest matched span. Groups 1-5 are kept.
   - Indexes: cp → `commonpile` (all Comma training sources, including the ones weighted 0.25; that over-excludes, which is safe), d3 → `dolma3_split`, dw → `dynaword1212`.
   - Dynaword candidates must also not occur in `commonpile`, which DFM saw too.
7. **Language check** (`lang_id.py`, GPU): `qanastek/51-languages-classifier` (XLM-R trained on short MASSIVE utterances) scores every candidate.
   - Candidates whose probability of the corpus language is below 0.2 are not selected. For Danish, that probability includes Norwegian Bokmål.
   - This removes sentences in another language that came from mixed-language documents, such as English abstracts in tidsskrift-dk or Portuguese and Dutch passages in Eurlex.
   - Sentences ending in a title abbreviation (`Mr.`, `hr.`, ...), which sentence splitting cut short, are not selected either.
8. **Similarity selection** (`scripts/embedding_similarity.py` + `build_unseen_prompt_sets.py select`):
   - The candidates are embedded (Qwen3-Embedding-8B) against a **selection reference**: 10k training documents per corpus drawn with seed 43 (`embedding_selection/`). This sample is independent of the seed-42 sample used for the evaluation.
   - Within each source, the candidates with the highest length-neutral score are kept. The score is max_sim minus the mean max_sim of candidates of similar length, so the selection doesn't shift the lengths.
   - Shares are water-filled to 1000 per corpus.
9. **Checks:**
   - `build_unseen_prompt_sets.py verify` recounts every prompt's verbatim occurrences (0 required, in all indexes) and checks sizes, unique ids and texts, and per-source counts.
   - `scripts/embedding_similarity.py` compares the specific sets with `tatoeba_<lang>` and `generic_<x>` on the seed-42 evaluation sample (`--summary-name unseen_eval`, outputs in `embedding_similarity/`), with bge-m3 as a robustness check.
   - `report.py` writes `results.md`.

```bash
python 00_prepare_data/propensity_settings/download_sources.py evidence dynaword_new dynaword1212
python 00_prepare_data/propensity_settings/download_sources.py common_corpus --num-files 40
python 00_prepare_data/propensity_settings/download_sources.py hplt --num-files 24
python 00_prepare_data/propensity_settings/download_sources.py finepdfs --num-files 8
python 00_prepare_data/propensity_settings/wikipedia_new.py --corpora dw --num-docs 2500
python 00_prepare_data/propensity_settings/wikipedia_new.py --corpora cp --num-docs 1000
python 00_prepare_data/propensity_settings/wikipedia_new.py --corpora d3 --num-docs 6000
python 00_prepare_data/propensity_settings/build_dynaword1212.py export index unigram --cpus 40 --mem 220
python 00_prepare_data/propensity_settings/build_candidates.py pool --corpora dw cp
python 00_prepare_data/propensity_settings/build_candidates.py pool --corpora d3 --pool-size 8000
python 00_prepare_data/propensity_settings/quality_score.py --corpora cp d3                          # GPU
python 00_prepare_data/propensity_settings/build_candidates.py sentences --corpora dw cp d3
python 00_prepare_data/propensity_settings/trace_candidates.py --corpora dw cp d3 --num-workers 12 --find-threads 4

# Selection reference (seed 43): sample the training docs, then embed the candidates (GPU)
python scripts/embedding_similarity.py --sample-only --seed 43 \
    --output-dir 00_prepare_data/propensity_settings/embedding_selection --corpora commonpile dolma3 dynaword1212 --sets tatoeba_eng tatoeba_dan
for x in "cp commonpile eng" "d3 dolma3 eng" "dw dynaword1212 dan"; do set -- $x
  python scripts/embedding_similarity.py --seed 43 --output-dir 00_prepare_data/propensity_settings/embedding_selection \
      --corpora $2 --extra-sets candidates_$1=00_prepare_data/propensity_settings/candidates_$1.jsonl:$3:candidates \
      --sets candidates_$1 --summary-name selection_$1
done

python 00_prepare_data/propensity_settings/lang_id.py --corpora dw cp d3                                # GPU
python 00_prepare_data/propensity_settings/build_unseen_prompt_sets.py select verify

# Evaluation on the seed-42 sample (GPU; the dynaword1212 docs are sampled first with --sample-only
# --corpora dynaword1212), then the report
M=memorization_experiment/data
python scripts/embedding_similarity.py --corpora commonpile dolma3 dynaword1212 \
    --extra-sets unseen_cp=$M/commonpile/specific/specific_prompts.jsonl:eng:unseen generic_cp=$M/commonpile/generic/generic_prompts.jsonl:eng:tatoeba \
                 unseen_d3=$M/dolma3/specific/specific_prompts.jsonl:eng:unseen generic_d3=$M/dolma3/generic/generic_prompts.jsonl:eng:tatoeba \
                 unseen_dw=$M/dynaword2/specific/specific_prompts.jsonl:dan:unseen generic_dw=$M/dynaword2/generic/generic_prompts.jsonl:dan:tatoeba \
    --sets tatoeba_eng tatoeba_dan generic_cp generic_d3 generic_dw unseen_cp unseen_d3 unseen_dw \
    --compare unseen_cp:tatoeba_eng unseen_cp:generic_cp unseen_d3:tatoeba_eng unseen_d3:generic_d3 unseen_dw:tatoeba_dan unseen_dw:generic_dw \
    --summary-name unseen_eval
# The same with --model BAAI/bge-m3 --instruction '' as a robustness check.
# For results.md, run both again with the 1000-prompt generic sets: generic_<x>_1000=$M/<dataset>/generic/generic_prompts_1000.jsonl:<lang>:tatoeba
# in --extra-sets, --sets and --compare, and --summary-name unseen_eval_1000.
python 00_prepare_data/propensity_settings/report.py                  # results.md (1000-prompt generic sets)
python 00_prepare_data/propensity_settings/report.py --generic full   # results_full_generic.md
```

### Prompt fields

`text`, `domain` (= source), `id` (`<x>-<source>-<doc id>`), `source`, `lang`, `n_tokens` (Llama-2, no BOS), `target_tokens` (the Tatoeba length it was matched to), overlap with the training index (`group`, `coverage_tokens`, `coverage_chars`, as in the generic sets), `lang_p` (probability of the corpus language, `lang_id.py`), `selection_max_sim` and `selection_score` (similarity to the selection reference), and provenance: `subsource` (HPLT topic, FinePDFs dump, Wikipedia language), `doc_id`, `date` (content date where known; `>YYYY-MM-DD` for new Wikipedia articles), `url`, `quality` and `quality_int` (FineWeb-Edu, English).

## Generations

The generations for the experiments are produced by `memorization_experiment/generation/run_generations_all.sh` (see `memorization_experiment/README.md`). `run_generations.py` here is an alternative driver that generates completions for the prompt sets with vLLM, one model at a time: it starts `vllm serve` (the `vllm` executable on `PATH`, or `--vllm-bin`), runs `memorization_experiment/generation/generate_vllm.py` on each prompt set, and stops the server.

| model | Hugging Face id @ revision | prompt sets | temperature / top_p |
|---|---|---|---|
| `olmo3-32b` | `allenai/Olmo-3-1125-32B` @ main | `*_d3` | 0.7 / 0.95 |
| `dfm-stage1`, `dfm-stage2`, `dfm-main` | `danish-foundation-models/dfm-decoder-open-v0-7b-pt` @ stage1, stage2, main | `*_dw`, `*_cp` | 0.7 / 0.95 |
| `comma-2t` | `common-pile/comma-v0.1-2t` @ main | `*_cp` | 0.7 / 0.95 |

All models use the same sampling settings. None sets them in `generation_config.json`; the Olmo 3 model card's usage example uses temperature 1.0 and top_p 0.7, which are deliberately not used here. For each corpus `x` the prompt sets are `generic_x` and `specific_x` (`--set-kinds` to choose).

Besides the prompt sets (setting `prompted`), two prompt-free settings run through `memorization_experiment/generation/generate_vllm_free.py`:

| setting | prompt | runs per model |
|---|---|---|
| `unconditional` | start-of-document token only: samples from the model's own distribution, with no bias from a prompt | one (`unconditional`) |
| `minimal_cue` | start-of-document token + one very common word, which fixes the language but not the content | one per language of its corpora: `minimal_cue_en` (cp, d3), `minimal_cue_da` (dw) |

- **Number of generations:** these settings have no prompts to repeat, so `--free-generations N` (the total per run) is required whenever they are selected.
- **Start-of-document token:** the tokenizer's BOS (`<|begin_of_text|>` for DFM and Comma). Olmo 3 has no BOS and separates training documents with `<|endoftext|>`, so it gets that. No tokenizer adds it to text prompts by itself, so these prompts are sent as token ids.
- **Cue words:** each generation gets its own word, drawn from the `--cue-top-n` (default 100) most frequent alphabetic words of the language in `wordfreq`, weighted by frequency (seed 42), with the first letter capitalized because it starts a document (`The`, `Det`). A cue is one word, which can be more than one token (e.g. `Så`). The drawn cues are counted in each output's `config.cue_counts`.
- **Seeds:** generations sharing a prompt are requested together with vLLM's `n`; generation g of a prompt is always sampled with seed 42 + g.

```bash
python 00_prepare_data/propensity_settings/run_generations.py --dry-run --free-generations 1000          # print the commands
python 00_prepare_data/propensity_settings/run_generations.py --num-generations 5 --free-generations 1000  # all models and settings
python 00_prepare_data/propensity_settings/run_generations.py --settings prompted --num-generations 5      # prompt sets only
python 00_prepare_data/propensity_settings/run_generations.py --models olmo3-32b --tensor-parallel-size 2 --free-generations 1000
```

Outputs go next to the prompts in `memorization_experiment/data`: `<dataset>/<setting>/generations/<model>/<setting>_generations.json` for the prompt sets (e.g. `dynaword2/generic/generations/dfm-main/generic_generations.json`) and `prompt_free/<run>/generations/<model>/<run>_generations.json` for `unconditional` and `minimal_cue_<lang>` (one record per completion, with its `prompt_id` and `sample_idx`; trace them with `simple_trace.py --is-generation-json`, which keys its results by these two, so identical completions are traced separately), with `run_config.json`, `vllm_server.log` and the run logs in `logs/generation/<model>/`. Defaults follow the earlier experiments: 256 new tokens, seed 42. Olmo 3 32B needs about 65 GB for its bf16 weights, so one 80 GB GPU or `--tensor-parallel-size 2`. `--no-serve` uses a server you started yourself; finished runs are skipped unless `--overwrite`.

`filter_generations.py` moves the generations of prompts that are no longer in a rebuilt prompt set out of the generation files.

## Files

- `tatoeba_<lang>_2k.jsonl`, `length_stats.json`, `tracing_overview.{json,md}`: the Tatoeba samples and their overlap with the indexes.
- `candidates_<x>.jsonl`: candidate sentences for the specific sets; `work/` (regenerable, not tracked) holds the document pools and intermediate files.
- `traces/`, `groups/`: SimpleTrace results, verbatim lookups and overlap groups of the Tatoeba samples (`*_tatoeba_*`) and the specific-set candidates (`*_unseen_*`).
- `prompt_sets/`: summaries of the generic prompt sets.
- `embedding_similarity/`: embedding evaluation of the prompt sets (seed-42 sample); `embedding_selection/` (not tracked): the seed-43 selection reference.
- `evidence/`: model and dataset cards behind the eligibility rule.
- `results.md`, `results_full_generic.md`: the report of the specific sets.
- `raw/`: the Tatoeba downloads.

Logs of these steps are in `logs/propensity_settings/`.

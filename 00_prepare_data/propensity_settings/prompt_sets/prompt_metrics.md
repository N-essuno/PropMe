# Prompt-set overlap metrics

SimpleTrace metrics of the **prompt texts themselves** against the index of each prompt set
(not of model generations). Built with `../prompt_set_metrics.py` from the per-text results of the traced
2k Tatoeba samples in `../traces/` (`mixed` mode, 10 docs per span).

- **FMR** (`generations_full_matches_ratio`): share of prompts for which a retrieved document contains the whole prompt.
- **ALS** (`average_longest_span_length`): mean length, in Llama-2 tokens, of each prompt's longest traced span.
- **NVR** (`avg_nv_recall`): mean near-verbatim recall over all retrieved documents; NVR on hits averages only documents with recall > 0.

| prompt set | index | prompts | FMR | ALS (tokens) | NVR | NVR on hits |
|---|---|---|---|---|---|---|
| generic_cp | Common Pile | 1642 | 0.0000 | 6.59 | 0.4481 | 0.6608 |
| generic_d3 | Dolma3 | 1174 | 0.0000 | 7.35 | 0.5312 | 0.7030 |
| generic_dw | Dynaword | 1664 | 0.0000 | 7.60 | 0.2597 | 0.6471 |

Notes:

- `generic_<x>` excludes full matches: prompts whose whole text occurs in the index, found by the token-level
  verbatim lookup or as a document or substring of a document SimpleTrace retrieved (`../group_by_overlap.py`),
  so its FMR is 0.
- ALS here comes from SimpleTrace's traced spans (at least 4 tokens, rarest spans kept), not from the exact longest match
  used for the overlap groups.

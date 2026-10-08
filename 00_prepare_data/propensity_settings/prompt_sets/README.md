# Prompt sets

Sentences from the 2k Tatoeba samples that do not occur verbatim in the given index:
overlap groups 1, 3, 4 and 5 from `../group_by_overlap.py` (groups 6 and 7, full-sentence matches, are excluded).
These are the generic sets; the specific sets (unseen-source sentences) are described in `../README.md`.
Built with `../build_prompt_sets.py`. The `file` column is relative to the repository root.

| prompt set | file | source sample | index | prompts | excluded full matches | avg / median Llama-2 tokens |
|---|---|---|---|---|---|---|
| generic_cp | `memorization_experiment/data/commonpile/generic/generic_prompts.jsonl` | `tatoeba_eng_2k` | Common Pile | 1642 | 358 | 11.35 / 10 |
| generic_d3 | `memorization_experiment/data/dolma3/generic/generic_prompts.jsonl` | `tatoeba_eng_2k` | Dolma3 | 1174 | 826 | 11.54 / 10 |
| generic_dw | `memorization_experiment/data/dynaword2/generic/generic_prompts.jsonl` | `tatoeba_dan_2k` | Dynaword | 1664 | 336 | 13.33 / 12 |

## By overlap group

Group = share of the prompt (in Llama-2 tokens) covered by its longest span that occurs verbatim in the index;
6 / 7 = the whole prompt occurs verbatim, in more than / at most 100 documents (excluded from the sets).

| prompt set | 1: < 25% | 3: 25-50% | 4: 50-75% | 5: >= 75%, not full | 6: full, > k docs | 7: full, <= k docs |
|---|---|---|---|---|---|---|
| generic_cp | 19 (1.2%) | 241 (14.7%) | 877 (53.4%) | 505 (30.8%) | 0 (0.0%) | 0 (0.0%) |
| generic_d3 | 7 (0.6%) | 109 (9.3%) | 589 (50.2%) | 469 (39.9%) | 0 (0.0%) | 0 (0.0%) |
| generic_dw | 17 (1.0%) | 307 (18.4%) | 872 (52.4%) | 468 (28.1%) | 0 (0.0%) | 0 (0.0%) |

## Fields

- `text`, `domain`: as in the `memorization_experiment` prompt files (`domain` is `tatoeba`).
- `id`, `source`, `lang`: the sentence in the 2k sample (`../<set>_2k.jsonl`).
- `group`, `coverage_tokens`, `coverage_chars`, `n_tokens`: its overlap with the index (`../groups/`).

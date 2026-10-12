# OLMoTrace reimplementation vs. SimpleTrace: features and benchmark results

This report compares the OLMoTrace reimplementation in `olmotrace/` (Liu et al., 2025,
[arXiv:2504.07096](https://arxiv.org/abs/2504.07096)) with SimpleTrace (`03_tracing/simple_trace.py`).
It has two parts:

1. A code-level comparison of the two pipelines: which features they share and where they differ.
2. Tests of the shared features on the three production indexes (Dynaword, Common Pile, Dolma3), with
   both tools run under the same conditions.

All scripts, inputs and raw outputs are in `olmotrace/comparison/` (see §6).

## TL;DR

- **Same core primitive, same answers.** Both tools find, for each start position, the longest
  prefix of the remaining text that occurs in the index, using one FIND plus the two neighbouring
  suffix-array entries. On 19,080 suffixes from all three indexes the two tools return the same
  length every time, and both equal the definition checked directly against the index (§4.1). The
  SimpleTrace "one-neighbour" bug described in the older `OLMOTRACE_REPORT.md` is fixed in the
  current code.
- **The final spans differ because of the rules around that primitive.** These include start
  positions, delimiters, word boundaries, merging and how many spans are kept. On short single
  sentences the tools agree closely (token Jaccard 0.93–0.99 vs. OLMoTrace). On 256-token verbatim
  training text and on real 7B generations they agree less (0.74–0.94). OLMoTrace highlights more
  tokens (Dolma3 verbatim coverage 0.90 vs. 0.76 for SimpleTrace `text`), mostly because it can start
  a span at the beginning of a line or at punctuation (§4.2).
- **Both find the source document of verbatim text.** OLMoTrace retrieves it for 192/192 inputs,
  SimpleTrace `mixed` for 192/192 and SimpleTrace `text` for 191/192 (§4.3).
- **SimpleTrace `mixed`, the mode the PropMe experiments use, behaves as intended for measuring
  memorization** (§4.6):
  - **Whole-text detection.** It flags every input whose whole text occurs in the index (all 216
    across the six input sets, none missed). It also flags 4 Danish continuations that are copied
    verbatim but begin mid-word, which a token-level lookup misses.
  - **Agreement with OLMoTrace.** On short text its highlighting is the closest to OLMoTrace's
    (Jaccard 0.98–0.99). Per generation, it ranks coverage more like OLMoTrace than `text` mode does
    (Spearman 0.85 vs. 0.79).
  - **Settings.** It orders the generic/specific/prefix settings the same way OLMoTrace does.
  - **Source of the differences.** Where it differs from OLMoTrace, 88% of the extra tokens come
    from spans that run across a sentence end, which is by design.
- **Performance depends on the input.**
  - On novel text and model generations, the two tools take about the same time and CPU.
  - On verbatim text over the 65-shard Dolma3 index, OLMoTrace is 15–30× slower (warm: 141 s vs.
    6.6 s for 64 texts) and uses about 30× more CPU, about 98% of it in the kernel.
  - In that run OLMoTrace's page tables reached **78.8 GB in a 96 GB container**. SimpleTrace stays
    under 8 GB because it caps page tables per worker and reuses each start's match for the next
    start.
  - **OLMoTrace with 32 worker processes on Dolma3 ran the container out of memory** (§4.4).

## 1. Setup

| | |
|---|---|
| Code tested | branch `iter2` at `ac41a5b`, **including uncommitted working-tree changes** (`03_tracing/simple_trace.py` md5 `e23586f5…`, `olmotrace/olmo_trace.py` md5 `67dca1cb…`) |
| Environment | Python 3.11.14, infini-gram 2.6.0, transformers 4.57.5; `rank_bm25` is not installed, so OLMoTrace uses its bundled port |
| Resources | cgroup limit of **32 CPUs** (`cpu.max` 3200000/100000) and **96 GB RAM** (`memory.max`). The host shows 256 cores and 754 GB, but the container cannot use them. The indexes are on a network filesystem (wekafs). The host is shared, with load average 5–58 during the runs. |
| Indexes | Dynaword: `dynaword_index` (1 shard, 52 GB). Common Pile: `commonpile_index/common_pile_train_index` (6 shards, 3.3 TB). Dolma3: `dolma3_index_link` (65 shards, 52 TB), the symlinked split index used by `04_validation/validation_full_dolma3.py` |
| Unigram tables | `02_unigram_probs/unigram_probs_{dynaword,common_pile_train,dolma3_link}.json` (the same table for both tools) |

Dolma3 (52 TB) and Common Pile (3.3 TB) are far larger than the memory available for page cache, so
on these indexes timings depend on which parts of the index are already in the page cache (§3).

## 2. Feature comparison (code analysis)

Both tools follow the same five-step structure from the paper. Each step is described below.

### 2.1 Overlapping features

| Feature | OLMoTrace (`olmotrace/olmo_trace.py`) | SimpleTrace (`03_tracing/simple_trace.py`) |
|---|---|---|
| Index backend | infini-gram `InfiniGramEngine`, several `--index-dir` values loaded as one multi-shard index | same |
| Engine flags | prefetching off (`*_prefetch_depth=0`), no unigram precompute | same (`load_engine`) |
| Tokenizer | `meta-llama/Llama-2-7b-hf`, no BOS/EOS | same, so token offsets can be compared directly |
| Step 1 primitive | `get_longest_prefix_len`: FIND on the whole suffix; if count = 0, compute the LCP with SA ranks `l-1` and `l` in every shard, max over shards | `longest_indexed_prefix_len`: FIND on the whole suffix; if count = 0, fetch the documents of ranks `l-1` and `l` in every shard and search them for the prefix. **Same results** (§4.1) |
| Step 1 parallelism | suffix queries on a thread pool (`--find-threads`) | same (`--find-threads`), in blocks of 4 consecutive starts |
| Non-maximal suppression | keep a span only if it extends past every span that starts earlier | same algorithm |
| Step 2 | keep `K = ceil(0.05·L)` spans with the lowest unigram probability | same rule |
| Step 3 | second FIND per kept span; up to `--docs-per-span` (10) occurrences, sampled at random if there are more, without building the full occurrence list; one batched document fetch | same |
| Step 4 | merge overlapping spans | merges, but with a different rule (§2.2) |
| Inputs | `--dataset dummy/generic`, `--is-jsonl --text-field`, `--is-generation-json --generation-text-field`, `--limit` | same flags |
| Outputs | results JSONL plus summary JSON; span-length histogram with `--length-buckets` | same |
| Runtime | `ProcessPoolExecutor` (one generation per task, `--num-workers`) with threads inside each worker | same |
| Reproducibility | seeded sampling per generation | seeded sampling per generation (seeded with `(seed, text)` rather than `(seed, index)`) |

### 2.2 Differences in the shared steps

| Aspect | OLMoTrace | SimpleTrace `text` | SimpleTrace `mixed` |
|---|---|---|---|
| Start positions | every begin-of-word token: `▁` pieces, punctuation, non-continuation bytes, **and the first token of a line** | only `▁` tokens, never the last token. A span cannot start at the beginning of a line, because Llama-2 drops the `▁` after `\n`, or at punctuation such as `"` and `(` | every position except the last |
| Lookups per start | one FIND per start | reuses the previous start's match as a lower bound and skips or shortens most lookups (`longest_prefix_lens`). Same results, far fewer queries on verbatim text | same as `text` |
| Sentence delimiters | the token-level set `.`, `▁.`, `<0x0A>`. **198 Llama-2 tokens that contain `.`, `!` or `?` (such as `."`, `).`, `...`) are not delimiters**, and neither are `!` and `?`. A span is trimmed back to its first delimiter. | character-level `! . ? \n` inside any token. A span is trimmed back to the boundary | none |
| End of span | the next token must begin a word, and punctuation counts as a word start, so a span can end before `,` | the next token must start with `▁`, so a span can never end just before `,` or `.` | no rule; minimum 4 tokens |
| Lone newline spans | allowed: a `\n` after a blank line is a valid 1-token span and can take a step-2 slot | impossible | impossible (minimum 4 tokens) |
| Step 2 score | sum of log-probabilities; `K` computed exactly with `Fraction` | product of probabilities, which underflows to 0 for long spans and then ties; `ceil(0.05·L)` in floating point, so `L = 60` gives `K = 4` | same as `text` |
| Step 3 context | snippet of 80 tokens plus 500 tokens of context on each side of the match | a 5·L-token window around each match (L = generation length) | same, plus evenly spaced sampling for full-match lookups |
| Step 4 merging | overlapping spans are **always** merged into one highlight; each keeps its documents; snippets from the same document are merged into one document | overlapping spans are merged **only if their union occurs in the index**; documents are then re-retrieved for the union | same as `text` |
| Memory | no limit on the growth of page tables from the memory-mapped index | `PageTableBoundedEngine` reopens the index once a worker's page tables exceed 1 GB (`--max-page-table-gb`); results are unchanged | same |

### 2.3 Features only one tool has

| OLMoTrace only (from the paper) | SimpleTrace only (PropMe) |
|---|---|
| Step 5: BM25 reranking with the user prompt plus the response as the query, normalized by 0.18·#chars, with high/medium/low relevance for documents and spans | `mixed` mode: FIND on the whole generation, full-document matching after normalization, and `match_tier` labels (`exact_full_raw`, `exact_full_normalized`, `partial`) |
| Uses the prompt (`--prompt-field`, `--generation-prompt-field`) | Adaptive NV-recall for every retrieved document |
| Per-step latency in every result line; links from spans to documents and from documents to spans | Memorization summary: full and normalized exact matches, NV-recall thresholds, n-token span ratio, optional k-eidetic counts |
| The paper's hyperparameters are exposed as flags (`--span-fraction`, `--snippet-context-tokens`, `--document-context-tokens`) | Feeds `05_propensity_metrics` directly; results keyed by `prompt_id:sample_idx` |
| 21 unit tests, including a brute-force check of Algorithm 1 | 25 validation tests (`04_validation/validation.py`) plus large randomized validators per corpus |

Both test suites pass in the environment above: OLMoTrace 21/21 in 34 s, SimpleTrace 25/25 in 0.8 s.

### 2.4 Statements in `olmotrace/OLMOTRACE_REPORT.md` that are out of date

That report describes an earlier SimpleTrace. With the current code:

- *"SimpleTrace's longest-match step misses the longer match on 26% of test suffixes (one
  neighbour)"*: no longer true. It now checks both neighbours and agrees with OLMoTrace and with the
  definition on 19,080/19,080 suffixes (§4.1).
- *"Results keyed by generation text, so duplicate generations collapse"*: results are now keyed by
  position, or by `prompt_id:sample_idx`.
- It does not mention SimpleTrace's lower-bound reuse in step 1 or its page-table cap. These two
  features explain most of the performance differences in §4.4.
- *"The two tools take about the same time"*: this holds for novel text and generations, but not for
  verbatim text on Dolma3 (§4.4).

## 3. Test design and fairness controls

**Inputs** (`olmotrace/comparison/prepare_inputs.py`). Both tools read the same JSONL file, in the
same order.

| Set | N | Avg tokens | Content | Used on |
|---|---|---|---|---|
| `dynaword_generations` | 300 | 160 | real greedy generations of `dfm-decoder-open-v0-7b-pt` for the generic/specific/prefix prompt sets (with prompts) | Dynaword |
| `*_verbatim` | 64 per corpus | 256 | the first 256 tokens of training documents sampled from that index (`memorization_experiment/data/*/…_sample_docs.jsonl`), with their source `doc_ix` and id | own corpus |
| `english_novel` | 100 | 32 | the 100 LLM-written English sentences of `load_generic_dataset()` | Common Pile, Dolma3 |

There are no English model generations in the repository, and no GPU was available to produce
them, so `english_novel` stands in for non-memorized English output.

**Fairness controls**

- Same index, unigram table, tokenizer and engine flags. Both tools use `--docs-per-span 10`, the
  same `--length-buckets`, **`--num-workers 8 --find-threads 4`** (32 concurrent queries, equal to the
  CPU quota) and their real CLIs.
- Only one run at a time. Each run's process tree was polled every 0.2–0.25 s for private memory
  (`RssAnon`) and page tables (`VmPTE`); CPU time comes from `getrusage`.
- **Cold cache:** each set was split into 3 interleaved slices, and the three variants (OLMoTrace,
  SimpleTrace `text`, SimpleTrace `mixed`) ran on each slice in Latin-square order. Each variant
  therefore ran first, on cold pages, on exactly one slice.
- **Warm cache:** two repetitions on the full set in mirrored order
  (ot → text → mixed → mixed → text → ot).
- The tools' sampling seeds differ (OLMoTrace 0 with the input index; SimpleTrace 42 with the text).
  This changes *which* 10 documents are sampled for frequent spans, never the spans themselves.

**What is not identical.** End-to-end times include work that only one tool does: OLMoTrace's BM25,
and SimpleTrace's NV-recall and `mixed` full-match search. `trace_elapsed` also includes about
1–2 s of worker start-up, which dominates the small sets. The step-1 benchmark (§4.1) isolates the
shared part.

## 4. Results

### 4.1 Test 1: the step-1 primitive (`comparison/step1_compare.py`)

For every SimpleTrace start position of each text, both tools' longest-prefix functions were called
on **the same engine object**, in alternating order. Each answer was checked against the definition
with two extra index queries:
- valid: `count(suffix[:n]) > 0`
- maximal: `n = len` or `count(suffix[:n+1]) = 0`

The script also timed each tool's full step-1 lookup loop over a text, single-threaded, running each
loop twice and alternating which tool went first. 8 processes in parallel.

| Index / inputs | Suffixes | No full match | OT = ST | OT correct | ST correct | Mean ms per call OT / ST | Step-1 loop, warm, s: OT / ST / OT on ST's starts |
|---|---|---|---|---|---|---|---|
| Dynaword / 100 generations | 8,402 | 8,061 | 100% | 100% | 100% | 0.19 / 0.23 | 1.82 / 1.55 / 1.33 |
| Dynaword / 16 verbatim | 1,687 | 0 | 100% | 100% | 100% | 0.11 / 0.10 | 0.147 / 0.029 / 0.112 |
| Common Pile / 100 novel | 2,392 | 1,990 | 100% | 100% | 100% | 1.18 / 1.35 | 2.27 / 2.27 / 2.35 |
| Common Pile / 16 verbatim | 2,039 | 0 | 100% | 100% | 100% | 0.37 / 0.38 | 0.92 / 0.16 / 0.61 |
| Dolma3 / 100 novel | 2,392 | 1,878 | 100% | 100% | 100% | 26.0 / 26.2 | 27.2 / 23.0 / 26.3 |
| Dolma3 / 16 verbatim | 2,168 | 0 | 100% | 100% | 100% | 17.1 / 17.2 | 13.1 / 2.2 / 10.9 |

Loop times are summed over texts. On every text, SimpleTrace's block loop returned exactly the same
lengths as its single calls (`st_loop_equals_single_calls_all = true`).

What this shows:

- The two implementations of the shared primitive are **equivalent and correct**, and **cost the same
  per call** (within about 15%).
- The cost difference comes from **how many calls are made**:
  - On verbatim text every suffix matches to the end. SimpleTrace then skips the next three starts in
    its block, and its loop is **4.9–6.0× faster**: OT/ST = 13.1/2.2 s on Dolma3, 0.92/0.16 s on
    Common Pile.
  - Comparing OLMoTrace on its own starts with OLMoTrace on SimpleTrace's starts shows that
    OLMoTrace's larger start set (it also starts at punctuation and at line starts; 3,064 vs. 2,168
    starts on Dolma3 verbatim) adds another 1.2–1.6×.
  - On novel text most suffixes do not match to the end, the lower bound rarely helps, and both
    tools cost the same.

### 4.2 Test 2: span agreement (warm1 runs, full sets)

Definitions:
- Coverage: the fraction of response tokens inside a final span.
- Jaccard: the token-level Jaccard index of the highlighted token sets, compared with OLMoTrace's
  merged spans (averaged over texts).
- Identical: the share of texts with exactly the same final spans as OLMoTrace.
- Inner boundary: the share of final spans with `! . ? \n` before their last character.

| Inputs (index) | Tool | Coverage | Spans/gen | Span len mean / median / max | Docs/gen | Jaccard vs OT | Identical | Inner boundary |
|---|---|---|---|---|---|---|---|---|
| Dolma3 verbatim (64×256) | OLMoTrace | 0.90 | 12.6 | 18.3 / 13 / 163 | 63.2 | – | – | 6.0% |
| | ST text | 0.76 | 10.6 | 18.3 / 14 / 160 | 38.8 | 0.80 | 0% | 0% |
| | ST mixed | 1.00 | 1.0 | 256 (whole text) | 1.5 | 0.90 | 0% | 100% |
| Common Pile verbatim (64×256) | OLMoTrace | 0.81 | 12.3 | 16.9 / 12 / 161 | 44.3 | – | – | 7.3% |
| | ST text | 0.67 | 10.7 | 16.1 / 11 / 156 | 37.8 | 0.74 | 0% | 0% |
| | ST mixed | 1.00 | 1.0 | 256 (whole text) | 1.1 | 0.81 | 0% | 100% |
| Dynaword verbatim (64×256) | OLMoTrace | 0.97 | 8.5 | 29.3 / 17 / 256 | 27.5 | – | – | 5.3% |
| | ST text | 0.92 | 7.8 | 30.2 / 18 / 256 | 22.1 | 0.94 | 50% | 0% |
| | ST mixed | 1.00 | 1.0 | 256 (whole text) | 1.0 | 0.97 | 2% | 98% |
| Dynaword 7B generations (300×160) | OLMoTrace | 0.53 | 5.6 | 14.6 / 12 / 111 | 24.2 | – | – | 1.2% |
| | ST text | 0.49 | 8.5 | 11.7 / 11 / 91 | 29.4 | 0.80 | 6% | 0% |
| | ST mixed | 0.55 | 8.5 | 15.9 / 14 / 123 | 15.3 | 0.75 | 5% | 48% |
| Dolma3 novel English (100×32) | OLMoTrace | 0.59 | 1.3 | 12.3 / 11 / 27 | 14.2 | – | – | 0% |
| | ST text | 0.59 | 2.2 | 10.1 / 9 / 26 | 7.7 | 0.94 | 22% | 0% |
| | ST mixed | 0.59 | 2.2 | 10.3 / 10 / 26 | 8.0 | 0.99 | 29% | 0% |
| Common Pile novel English (100×32) | OLMoTrace | 0.52 | 1.6 | 9.2 / 8 / 26 | 7.5 | – | – | 0% |
| | ST text | 0.52 | 2.2 | 8.0 / 7 / 24 | 8.1 | 0.93 | 35% | 0% |
| | ST mixed | 0.53 | 2.2 | 8.1 / 7.5 / 25 | 7.9 | 0.98 | 41% | 0% |

Every retrieved document or context contains its span text, for all tools and all sets (100%).

Why the spans differ (code analysis checked against these outputs):

- **Line-start and punctuation starts.** 60–71% of the tokens that only OLMoTrace highlights belong to
  OLMoTrace spans starting at a token without `▁`: the first word of a line, or `"`, `(` and similar.
  SimpleTrace `text` cannot start a span there. This is the main reason for the 0.76 vs. 0.90 coverage
  on Dolma3.
- **Merging.**
  - OLMoTrace always merges overlapping spans, so it has fewer, longer final spans on generations:
    5.6 vs. 8.5 per generation, with the same `K` in both tools.
  - SimpleTrace merges only when the union occurs in the index, so it often keeps two overlapping
    spans, for example `…the smell of fresh` and `smell of fresh rain…`.
  - This is also why spans are rarely *identical* even when coverage is equal.
- **Delimiter gaps in OLMoTrace.** 5–7% of OLMoTrace's spans on verbatim text cross a sentence end.
  The delimiter set misses compound tokens and `!`/`?`, for example
  `“Hi, I’m Marcel.” The boy whispered sitting down next to me.` (crossing at `.”`) and
  `Another masterpiece from SIE Santa Monica Studios...`. SimpleTrace `text` never does this.
  `mixed` has no delimiter rule at all (48% of its spans on generations).
- **Lone newline spans.** 101 of OLMoTrace's 804 kept spans on Dolma3 verbatim (12.6%), and 56/790 on
  Common Pile, are a single `\n` token after a blank line, each occurring about 2.8·10¹¹ times in
  Dolma3. They use up step-2 slots without carrying any information.
- **Code.** Both tools' delimiter rules break source code into small pieces. On a Java file
  (Common Pile verbatim #58), OLMoTrace keeps `package fruitymod` plus eleven `;\n` spans (735M
  occurrences each), and SimpleTrace `text` keeps only `package`. SimpleTrace `mixed` matches the whole
  256-token window.
- **`mixed` on verbatim text** reports one whole-text span, by design: the full-generation FIND
  succeeds and the result subsumes every partial span. Its 1.0–1.5 documents per text are the
  full-match documents (`exact_full_raw` for 64/64 texts on every corpus, mean NV-recall 0.99).

### 4.3 Test 3: source attribution on verbatim text

Does the tool retrieve the document the input was cut from? A hit means the source `doc_ix` or id is
among the generation's retrieved documents.

| Corpus | OLMoTrace | ST text | ST mixed |
|---|---|---|---|
| Dolma3 | 64/64 | 64/64 | 64/64 |
| Common Pile | 64/64 | 63/64 | 64/64 |
| Dynaword | 64/64 | 64/64 | 64/64 |

The only miss is the Java file above: SimpleTrace `text`'s single span `package` (1 token) has
millions of occurrences, so a sample of 10 does not contain the source.

OLMoTrace's BM25 rates 34% of its Dolma3 verbatim documents "high" relevance, 13% on Common Pile and
12% on Dynaword. On Dynaword generations it rates 55% "high", partly because the prompt is in the
query. The paper reports 14% high-relevance documents on OLMo chat responses.

### 4.4 Test 4: performance (8 workers × 4 find-threads, same inputs)

`trace` is the tool's own reported `Elapsed Time`. It covers starting the worker pool and tracing,
but not loading inputs or writing the summary. Cold is the tool's first-position slice in the Latin
square (about 1/3 of the set). Warm is the mean of the two full-set repetitions. CPU is user + system
time over the process tree for the warm runs. Page tables (PT) is the peak sum of `VmPTE` over all
runs of the group.

| Inputs (index) | Tool | Cold, 1/3 of set (s) | Warm, full set (s) | Warm CPU (s) | Peak PT (GB) |
|---|---|---|---|---|---|
| Dolma3 verbatim (64) | OLMoTrace | 106.0 | **141.3** | **4,354** | **78.8** |
| | ST text | 13.0 | 6.6 | 129 | 7.3 |
| | ST mixed | 16.3 | 7.4 | 158 | 7.6 |
| Dolma3 novel (100) | OLMoTrace | 12.3 | 7.5 | 153 | 26.2 |
| | ST text | 11.1 | 5.9 | 117 | 7.6 |
| | ST mixed | 12.8 | 7.4 | 145 | 7.6 |
| Common Pile verbatim (64) | OLMoTrace | 18.4 | 2.0 | 30 | 6.5 |
| | ST text | 3.8 | 1.2 | 15 | 1.8 |
| | ST mixed | 3.2 | 1.0 | 17 | 2.0 |
| Common Pile novel (100) | OLMoTrace | 1.5 | 1.2 | 19 | 2.2 |
| | ST text | 1.3 | 1.0 | 19 | 2.1 |
| | ST mixed | 1.4 | 1.1 | 20 | 2.5 |
| Dynaword generations (300) | OLMoTrace | 2.1 | 1.8 | 25 | 0.4 |
| | ST text | 1.8 | 2.2 | 25 | 0.4 |
| | ST mixed | 2.6 | 2.6 | 33 | 0.5 |
| Dynaword verbatim (64) | OLMoTrace | 1.2 | 1.1 | 12 | 0.3 |
| | ST text | 1.0 | 1.1 | 12 | 0.1 |
| | ST mixed | 1.0 | 0.9 | 11 | 0.2 |

Private memory (`RssAnon`) was 1.3–1.7 GB for every run, mostly 8 copies of the tokenizer and the
unigram table.

**Reading the cold column.** The slices differ, so compare times within a slice too. On Dolma3
verbatim, OLMoTrace was slower than both SimpleTrace variants on every slice, even when it ran third
on an already warm slice (47.9 s and 30.1 s, against 13.0 s and 3.9 s for SimpleTrace `text` on
those slices). On the other sets, running first roughly doubles to quadruples the time for any tool,
and the cold-cache effect is larger than the difference between tools.

**Why OLMoTrace is slow on verbatim Dolma3:**

1. **About 5× more index lookups** (§4.1). No lookups are skipped on fully matching suffixes, and
   there are more start positions. Each lookup is a FIND across 65 shards.
2. **Unbounded page tables.**
   - Every lookup touches suffix-array and token pages in all 65 memory-mapped shards. Each worker
     builds its own page tables for those pages, and nothing frees them.
   - The warm 64-text run reached 78.8 GB of page tables. Three 21-text runs reached about 32 GB each.
   - In the 8 × 1 run, which recorded user and system time separately, OLMoTrace used **37 s user vs.
     2,017 s system time**. Almost all of its CPU goes to the kernel populating page tables and
     handling page faults.
   - The full 64-text run used about twice the CPU of the three cold 21-text slices together, which
     is consistent with page tables pushing the page cache out of the 96 GB container.
   - SimpleTrace reopens a worker's index once its page tables pass 1 GB, so its total stays at about
     7 GB.
3. Step 1 is 85% of OLMoTrace's time on this set, step 3 is 15% and BM25 is under 1%. This matches the
   paper's finding that span computation dominates.

### 4.5 Concurrency sensitivity and the out-of-memory event

To check that the ranking does not depend on the 8 × 4 layout, the worst case (Dolma3 verbatim, 64
texts, warm) was re-run with other layouts:

| Layout (workers × threads) | Tool | Trace (s) | CPU (s) | user / system (s) | Peak PT (GB) |
|---|---|---|---|---|---|
| 8 × 4 (main) | OLMoTrace | 147.8 / 134.8 | 4,544 / 4,164 | – | 78.7 / 78.8 |
| | ST text | 8.8 / 4.5 | 180 / 78 | – | 7.0 / 7.3 |
| 8 × 1 | OLMoTrace | 93.7 / 83.2 | 2,460 / 2,054 | – / 37 / 2,017 | 78.8 / 78.7 |
| | ST text | 11.0 / 4.6 | 140 / 63 | 23 / 118, 18 / 44 | 7.4 / 6.7 |
| 16 × 2 | OLMoTrace | **stopped by the memory guard** (60 GB cap) after 60 s at 28/64 texts | n/a (killed) | – | 54.8 |
| | ST text | 4.6 / 4.4 | 96 / 94 | 22 / 75, 21 / 74 | 14.3 / 14.7 |
| 32 × 1 | OLMoTrace | **killed the container (out of memory) at 33/64 texts** (unguarded run) | | | > 96 (cgroup limit) |
| | ST text | 5.4 / 5.3 | 128 / 127 | 22 / 106, 22 / 105 | 27.6 / 27.3 |

The 16 × 2 and 32 × 1 runs were made after the container restarted. They were preceded by one
unmeasured SimpleTrace warm-up run, and the guard was set to 60 GB. SimpleTrace's page tables grow
with the number of workers but stay below its bound (workers × 1 GB). Its system time also exceeds
its user time, but the absolute amounts are about 20× smaller than OLMoTrace's. At every layout
SimpleTrace finished the 64 Dolma3 verbatim texts in 4–11 s, while OLMoTrace needed 83–148 s or did
not fit in memory.

**Out-of-memory event.** The first 32 × 1 attempt for OLMoTrace exceeded the 96 GB cgroup limit.
32 processes each built page tables for the 65 Dolma3 shards, and the whole job was killed. The
benchmark runner (`comparison/run_bench.py`) now watches the container's unreclaimable memory
(anon + page tables + unreclaimable slab) and kills the traced process tree above
`--max-unreclaimable-gb`. The default is 70 GB; 60 GB was used for the runs in this section.
OLMoTrace's per-worker page tables (about 4 GB per worker for about 3 texts, about 10 GB for 8 texts)
mean that **8 workers is about the maximum for OLMoTrace on Dolma3 in this 96 GB container**, and
even that is within about 17 GB of the limit on 64 texts. SimpleTrace's total is bounded by
`--num-workers × --max-page-table-gb` (32 GB at 32 workers).

### 4.6 SimpleTrace `mixed` mode in detail (`comparison/mixed_analysis.py`)

`mixed` is the mode used by the experiment presets in `memorization_experiment/`. It works like
this:
1. It first runs one FIND on the whole generation. If the generation occurs, it fetches those
   documents as full matches.
2. It searches for spans from **every** start position, with no sentence or word-boundary rules,
   keeping only pieces of at least 4 tokens.
3. Steps 2–4 are the same as in `text` mode: `K = ceil(0.05·L)`, rarest first, and overlapping
   spans are merged only if their union is in the index.
4. If no full match was found, it takes up to 8 long "anchor" spans and checks whether their
   documents contain the whole generation after light normalization (`exact_full_normalized`).
5. Every retrieved document gets a `match_tier` and an NV-recall score.

All numbers below come from the warm1 runs of §4.2, plus one FIND per input of its full token
sequence as a ground-truth check (single process).

**Full-text detection.** "In index" means the input's full token sequence has a FIND count > 0.
"Flagged" means at least one retrieved document has `match_tier = exact_full_raw`.

| Inputs (index) | N | In index | Flagged by `mixed` | Flagged and in index | Flagged, not in index | In index but missed | Distinct docs with the full text: 1 / 2–9 / ≥10 |
|---|---|---|---|---|---|---|---|
| Dynaword generations | 300 | 6 | 10 | 6 | 4 | 0 | 2 / 4 / 4 |
| Dynaword verbatim | 64 | 64 | 64 | 64 | 0 | 0 | 63 / 1 / 0 |
| Common Pile verbatim | 64 | 64 | 64 | 64 | 0 | 0 | 59 / 5 / 0 |
| Dolma3 verbatim | 64 | 64 | 64 | 64 | 0 | 0 | 51 / 13 / 0 |
| Common Pile novel English | 100 | 8 | 8 | 8 | 0 | 0 | 0 / 1 / 7 |
| Dolma3 novel English | 100 | 10 | 10 | 10 | 0 | 0 | 1 / 1 / 8 |

- **No misses.** Every input whose full token sequence is in the index was flagged.
- **The 4 "flagged, not in index" cases are real copies.** All 4 are prefix-setting continuations
  that start mid-word, because the prompt ended inside a word: `…Varsel, at mø` +
  `de i Hof= og Stadsrettens Skiftecommission…` (88 tokens), and `…Meubler tilk` +
  `jøbs og til leie…` (123 tokens).
  - On its own, the continuation tokenizes as `▁de…`; inside the document it is `mø`+`de`. So the
    full-sequence FIND returns 0.
  - `mixed` can start a span inside a word (here at token 1, matching to the end). Its string-level
    `exact_full_raw` check then finds the whole continuation verbatim in the retrieved document.
  - OLMoTrace, and a whole-text token check, would not report these generations as full copies.
- **Normalized matching never fired.** `exact_full_normalized` occurred 0 times in these sets, so
  every full match here is a raw verbatim match.
- **The "novel" English sentences that are in the corpus.**
  - 9 of the 10 Dolma3 full matches are short stock phrases (`Run the script.`, `Access denied.`,
    `Believe in yourself.`), each found in 10 or more documents.
  - The 10th is a whole encyclopedic sentence found verbatim:
    `Bioluminescence is the production and emission of light by a living organism as the result of a
    chemical reaction.`
- **Duplication.** The number of distinct documents containing the full text (capped at
  `--docs-per-span` = 10) shows how much of the copied material is duplicated in each corpus. Of the
  verbatim windows, 13/64 occur in 2–9 Dolma3 documents, against 5/64 for Common Pile and 1/64 for
  Dynaword.

**Memorization signals by prompt setting (Dynaword generations, 100 per setting).** The prefix
setting is the capability (extraction) attack; generic and specific are ordinary use.

| Signal | Tool | generic | specific | prefix |
|---|---|---|---|---|
| Coverage | OLMoTrace | 0.47 | 0.50 | 0.61 |
| | ST mixed | 0.50 | 0.53 | 0.64 |
| | ST text | 0.41 | 0.45 | 0.60 |
| Longest span per generation, mean (tokens) | OLMoTrace | 19.7 | 20.3 | 29.5 |
| | ST mixed | 17.0 | 16.9 | 27.9 |
| Generations with a span ≥ 60 tokens | OLMoTrace | 3 | 0 | 14 |
| | ST mixed | 2 | 1 | 11 |
| Generations whose full text is in the corpus | ST mixed | 0 | 1 | 9 |
| Generations with max NV-recall > 0.5 / ≥ 0.9 | ST mixed | 0 / 0 | 1 / 0 | 18 / 6 |
| Token Jaccard, mixed vs. OLMoTrace | | 0.71 | 0.73 | 0.82 |
| Spearman of per-generation coverage, mixed vs. OLMoTrace | | 0.74 | 0.80 | 0.93 |

- **Both tools tell the same story.** The prefix attack draws out clearly more training text than
  ordinary prompts: about 0.6 coverage, 11–14 generations with a span of 60 or more tokens, and 9
  full copies, against 0–3 such generations for generic and specific prompts. Specific prompts are
  close to generic ones: slightly higher coverage, but no more long spans.
- **Only `mixed` provides the full-copy and NV-recall signals** that `05_propensity_metrics` uses.
- **Agreement with OLMoTrace rises with the amount of memorized text:** Jaccard 0.71 → 0.82, and
  coverage rank correlation 0.74 → 0.93.
- **OLMoTrace's longest highlight is longer** (23.2 vs. 20.6 tokens on average). OLMoTrace merges
  every overlapping span into one highlight, while `mixed` merges only if the union is in the index.
  So OLMoTrace's merged "spans" can be longer than any run of text that actually occurs in the
  corpus.

**Per-generation agreement with OLMoTrace (Spearman rank correlation over generations).**

| Inputs | Coverage: mixed / text | Longest span: mixed / text |
|---|---|---|
| Dynaword generations (300) | **0.85** / 0.79 | 0.74 / **0.88** |
| Common Pile novel English (100) | **0.99** / 0.93 | 0.80 / 0.79 |
| Dolma3 novel English (100) | **0.99** / 0.95 | 0.84 / 0.84 |

On the verbatim sets `mixed` reports the same 256-token span for every input, so a rank correlation
is undefined there. Its agreement with OLMoTrace is 0.81–0.97 Jaccard (§4.2).

- `mixed` ranks generations by **how much** text is memorized more like OLMoTrace than `text` mode
  does.
- It ranks them by **longest span** less like OLMoTrace on generations. Its spans continue across
  sentence ends, while OLMoTrace's stop at each period and are then merged.

**Where the tokens that only one tool highlights come from** (Dynaword generations, each token
counted once):

| Tokens highlighted by `mixed` only (4,848) | Share | Tokens highlighted by OLMoTrace only (3,497) | Share |
|---|---|---|---|
| in a `mixed` span that crosses a sentence end | 87.9% | in an OLMoTrace span shorter than 4 tokens | 0% |
| in a `mixed` span that starts inside a word | 6.1% | step 2 kept different spans, or merging | 100% |
| other (step 2 / merging) | 6.0% | | |

- Before step 2, `mixed`'s candidate spans cover every token of OLMoTrace's candidate spans. It
  queries every start position and never trims a match, so the only possible exceptions are spans
  under 4 tokens, and none occurred here.
- OLMoTrace-only tokens therefore come **only from step 2**. Both tools keep the same number `K` of
  rarest spans, `mixed` spends some of those slots on longer spans that cross sentence ends, and
  some shorter matches that OLMoTrace keeps are left out.
- On the verbatim sets, 100% of `mixed`-only tokens come from its single whole-text span, and no
  token is highlighted by OLMoTrace alone.

**SimpleTrace's own `mixed` summary metrics** (from `*__warm1__st_mixed.summary.json`; NV-recall
threshold 0.5; n-token span threshold 60):

| Inputs (index) | Generations with a full match | Generations with NV-recall > 0.5 | Generations with a span ≥ 60 tokens | Mean longest span (tokens) | Mean NV-recall on hits |
|---|---|---|---|---|---|
| Dynaword generations | 3.3% | 6.3% | 4.7% | 20.6 | 0.60 |
| Dynaword / Common Pile / Dolma3 verbatim | 100% | 100% | 100% | 256 | 0.99–1.00 |
| Common Pile novel English | 8.0% | 9.0% | 0% | 8.6 | 0.44 |
| Dolma3 novel English | 10.0% | 20.0% | 0% | 11.1 | 0.48 |

**Cost.** `mixed` costs about the same as `text`. On Dynaword generations it is about 20% slower
(warm 2.6 vs. 2.2 s, CPU 33 vs. 25 s) because of the extra whole-generation FIND, more start
positions and the anchor search. On Dolma3 verbatim it takes 7.4 s against 6.6 s, still about
**19× faster than OLMoTrace** (141 s) with about 10× smaller page tables (7.6 vs. 78.8 GB, §4.4).

## 5. Conclusions and recommendations

1. **Use the tool that matches the question.**
   - OLMoTrace reproduces the paper: highlights for a reader, relevance ranking and prompt-aware BM25.
   - SimpleTrace is built for memorization metrics: full-match tiers, NV-recall and propensity inputs.
   - Both rest on the same primitive, and that primitive is correct and equivalent in both (§4.1).
     Their headline numbers differ because of span-shaping rules, not because of retrieval errors.
2. **For PropMe experiments** (SimpleTrace `mixed`, the mode used by the experiment presets), the
   results above support SimpleTrace as it is:
   - It finds the source of every verbatim text.
   - It flags every input whose whole text is in the index, with no misses. Its string-level check
     also catches copied continuations that begin mid-word, which token lookups miss (§4.6).
   - It agrees with OLMoTrace's coverage within 0.01 on short novel text. Per generation, it ranks
     coverage more like OLMoTrace than `text` mode does.
   - It reproduces the same ordering of the prompt settings as OLMoTrace (generic ≈ specific < prefix).
   - It is the only one of the two that runs safely at scale on Dolma3.
   - Its spans differ from OLMoTrace's mainly because they continue across sentence ends. For
     measuring memorization this is a feature, but `mixed` spans should not be presented as
     OLMoTrace-style highlights.
3. **If OLMoTrace is to be used on Dolma3**, port two changes from SimpleTrace that do not change
   results:
   - the lower-bound reuse in step 1 (shown equivalent in §4.1);
   - a page-table cap (reopen the engine).
   Without them, run it with at most 8 workers on this container, and on more texts only with
   a memory guard.
4. **Possible fixes to the span rules** (these change results, so they are design decisions):
   - SimpleTrace `text`: allow starts at the beginning of a line and at punctuation, using
     OLMoTrace's begin-of-word rule. This would close most of its coverage gap.
   - OLMoTrace: add compound period tokens (`."`, `).`, `...`) and `!`/`?` to the delimiters, and drop
     1-token whitespace spans before step 2. This departs from the letter of the paper.
5. **Refresh `olmotrace/OLMOTRACE_REPORT.md`**: several of its SimpleTrace statements are out of date
   (§2.4).

## 6. Reproducing

```bash
# from the repository root
python -m unittest discover -s olmotrace/tests          # 21 OLMoTrace tests
python 04_validation/validation.py                       # 25 SimpleTrace tests
python olmotrace/comparison/prepare_inputs.py            # -> olmotrace/comparison/inputs/
python olmotrace/comparison/schedule.py                  # end-to-end runs, 8x4, Latin square + warm reps
python olmotrace/comparison/step1_compare.py --index-dir <index> --inputs <set> --output <json>   # §4.1
olmotrace/comparison/sensitivity.sh                      # §4.5, memory-guarded
python olmotrace/comparison/analyze.py > olmotrace/comparison/outputs/analysis.json
PROPME_DATA_ROOT=<data root> python olmotrace/comparison/mixed_analysis.py --check-index \
    > olmotrace/comparison/outputs/mixed_analysis.json   # §4.6
```

| Path | Content |
|---|---|
| `comparison/inputs/` | the shared input sets and Latin-square slices |
| `comparison/outputs/e2e/runs.jsonl` | one record per run: timing, CPU, memory, page tables |
| `comparison/outputs/e2e/<group>__<phase>__<tool>.{results.jsonl,summary.json}` | raw tool outputs (their logs are in `logs/olmotrace_comparison/e2e/`) |
| `comparison/outputs/step1_*.json` | per-suffix test 1 results |
| `comparison/outputs/analysis.json` | the aggregates used in §4.2–4.4 |
| `comparison/outputs/mixed_analysis.json` | the `mixed`-mode aggregates used in §4.6 |

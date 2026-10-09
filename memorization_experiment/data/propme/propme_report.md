# PropMe results report

Generated 2026-10-09 by `05_propensity_metrics/make_propme_report.py` from the SimpleTrace traces in `memorization_experiment/data/propme/`.

## Setup and changes since the paper

| | Paper (ARR, 12 Oct) | This report |
|---|---|---|
| Models / corpora | Comma (CP); DFM Decoder stages 1, 2, final (CP, DW) | same, plus **Olmo 3 32B on Dolma 3 (D3)** |
| Decoding | greedy (temperature 0), 1 generation per prompt | sampling, temperature 0.7, top-p 0.95, 256 new tokens, **10 generations per prompt** |
| Prompts per setting | 100 | 1000 prompts × 10 = 9984 generations (prompt-free settings: 9999 generations) |
| Generic | GPT-generated plausible prompts | Tatoeba sentences not verbatim in the index (random 1000-prompt sample) |
| Specific | GPT-generated, dataset-inspired | sentences from unseen sources close to the training data (00_prepare_data/propensity_settings, 1000 per corpus) |
| Prefix (capability) | first 50 tokens of training documents | same construction, random 1000-prompt sample |
| New propensity settings | — | **Unconditional** (start-of-document token only) and **Minimal cue** (start-of-document token + one frequent word drawn per generation) |
| Confidence intervals | Wilson 95% for NVR and FMR | percentile bootstrap 95% (B = 10,000) for NVR, FMR, ALS and PM; resampling prompts (each with its 10 generations) for generic, specific and prefix, generations for unconditional and minimal cue |

Settings are listed from the least to the most targeted prompting: Unconditional, Minimal cue, Generic, Specific (propensity settings) and Prefix (capability setting). A full-match ratio of 0 is shown with its rule-of-three upper bound 3/n (n = independent units). ↑ / ↓ mark differences whose 95% CI lies above / below 0, ≈ those whose CI includes 0. † PM is 0 by convention when both values are 0 (paper, Eq. 1).

## Degenerate outputs (excluded from the main results)

Some generations are repetition loops of symbols: `|` on every line up to the 256-token limit (most cases, sometimes with an occasional word such as `Ja |` / `Nej |`), repeated numbers or percentages (`120 | 120`, `0.0% 0.0%`), dot leaders (`. . . .`), rule lines (`---`) and nested braces. They match training text verbatim — empty table columns and tables of contents exist in every corpus — so they count as full matches and long spans without any memorized content. A generation is **degenerate** when both hold:

- its zlib compression ratio (compressed / raw UTF-8 size) is below 0.1, i.e. it is highly repetitive (zlib entropy is used to discard trivial repetitive matches by Carlini et al., 2021);
- under 50% of its non-space characters are letters.

Both conditions are needed: repetition alone also flags generations that repeat a natural-language sentence (common under generic prompts), and a low letter share alone also flags code, tables and logs, which are kept because their verbatim reproduction is memorization (their share is examined separately in Section 5.3). Sentence loops therefore stay in the main tables; most match only short spans, but a minority match repetitive training text over 50–360 tokens. Section 5 discounts them with a span-level loop filter (5.2).

Degenerate generations are removed from numerators and denominators of every metric, in every setting including prefix, so each PM compares like with like. Bootstrap units are kept (a prompt whose 10 generations are all degenerate contributes nothing), so paired comparisons are unchanged. **All tables and sections of this report use the filtered results**; the unfiltered Tables A and B are in the appendix, and Section 6 gives each claim's unfiltered status. The rule was set after the loops were found while inspecting DFM's specific-prompt full matches, not chosen by its effect on PM. Every excluded generation was checked to be such a loop; non-repetitive symbol sequences are kept, e.g. one counting sequence `1. 2. 3. … 77.` among DFM's specific full matches on Common Pile.

Excluded generations per run, with the full matches among them in parentheses (of 10,000 generations each):

| Model | Data | Unconditional | Minimal Cue | Generic | Specific | Prefix |
|---|---|---|---|---|---|---|
| Comma | CP | 13 (1) | 1 (0) | 3 (0) | 5 (2) | 12 (0) |
| DFM | CP | 1 (0) | 3 (0) | 3 (2) | 75 (38) | 18 (0) |
| DFM | DW | 1 (0) | 1 (0) | 16 (8) | 74 (34) | 4 (0) |
| Olmo 3 32B | D3 | 8 (0) | 0 | 0 | 0 | 12 (0) |
| DFM Stage 1 | CP | 9 (5) | 2 (0) | 12 (5) | 68 (40) | 21 (0) |
| DFM Stage 2 | CP | 4 (0) | 1 (0) | 5 (2) | 60 (28) | 15 (0) |
| DFM Stage 1 | DW | 9 (4) | 2 (0) | 13 (6) | 68 (28) | 4 (0) |
| DFM Stage 2 | DW | 4 (0) | 1 (0) | 16 (3) | 51 (16) | 3 (0) |

## Table A — Memorization metrics across model–corpus pairs (cf. paper Tables 1 and 3)

| Model | Data | Prompt | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | Excluded |
|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 0.0353 [0.0327, 0.0379] | 0.0018 [0.0010, 0.0027] | 35.76 [35.08, 36.42] | 13 |
| Comma | CP | Minimal Cue | 0.0338 [0.0313, 0.0365] | 0.0035 [0.0024, 0.0047] | 29.10 [28.47, 29.76] | 1 |
| Comma | CP | Generic | 0.0023 [0.0018, 0.0028] | 0.0014 [0.0007, 0.0022] | 13.88 [13.53, 14.27] | 3 |
| Comma | CP | Specific | 0.0034 [0.0027, 0.0042] | 0.0017 [0.0009, 0.0027] | 16.23 [15.78, 16.71] | 5 |
| Comma | CP | Prefix | 0.0291 [0.0226, 0.0363] | 0.0056 [0.0028, 0.0089] | 32.46 [30.60, 34.42] | 12 |
| DFM | CP | Unconditional | 0.0072 [0.0058, 0.0087] | 0.0015 [0.0008, 0.0023] | 10.31 [10.03, 10.60] | 1 |
| DFM | CP | Minimal Cue | 0.0286 [0.0260, 0.0313] | 0.0071 [0.0055, 0.0087] | 18.58 [18.04, 19.14] | 3 |
| DFM | CP | Generic | 0.0061 [0.0052, 0.0071] | 0.0057 [0.0042, 0.0073] | 13.38 [13.10, 13.67] | 3 |
| DFM | CP | Specific | 0.0029 [0.0023, 0.0035] | 0.0029 [0.0018, 0.0041] | 15.19 [14.70, 15.70] | 75 |
| DFM | CP | Prefix | 0.0260 [0.0202, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.24 [28.55, 32.01] | 18 |
| DFM | DW | Unconditional | 0.0475 [0.0450, 0.0501] | 0.0287 [0.0255, 0.0320] | 26.25 [25.82, 26.69] | 1 |
| DFM | DW | Minimal Cue | 0.0273 [0.0253, 0.0294] | 0.0208 [0.0180, 0.0237] | 20.25 [19.98, 20.53] | 1 |
| DFM | DW | Generic | 0.0008 [0.0005, 0.0012] | 0.0011 [0.0004, 0.0020] | 15.53 [15.34, 15.72] | 16 |
| DFM | DW | Specific | 0.0034 [0.0026, 0.0043] | 0.0050 [0.0032, 0.0072] | 17.29 [16.62, 18.00] | 74 |
| DFM | DW | Prefix | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.44 [24.05, 26.89] | 4 |
| Olmo 3 32B | D3 | Unconditional | 0.0249 [0.0236, 0.0263] | 0.0007 [0.0002, 0.0013] | 53.42 [52.62, 54.24] | 8 |
| Olmo 3 32B | D3 | Minimal Cue | 0.0121 [0.0109, 0.0133] | 0.0040 [0.0028, 0.0053] | 26.54 [25.90, 27.19] | 0 |
| Olmo 3 32B | D3 | Generic | 0.0021 [0.0015, 0.0031] | 0.0002 [0.0000, 0.0005] | 15.10 [14.83, 15.41] | 0 |
| Olmo 3 32B | D3 | Specific | 0.0012 [0.0005, 0.0023] | 0.0006 [0.0001, 0.0014] | 17.68 [17.19, 18.23] | 0 |
| Olmo 3 32B | D3 | Prefix | 0.0085 [0.0058, 0.0120] | 0.0029 [0.0009, 0.0058] | 26.90 [25.48, 28.39] | 12 |

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3. Excluded: degenerate generations left out. FMR counts full matches of any length, as in the paper; under generic and specific prompts most of them are short outputs (often under 10 tokens) that occur verbatim in the training data. FMR20 / FMR50, counting only full matches of ≥ 20 / ≥ 50 tokens, are in Section 5.2.

## Table B — Propensity memorization scores against prefix (cf. paper Tables 2 and 4)

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_NVR | 0.548 [0.490, 0.614] | 0.538 [0.480, 0.602] | 0.073 [0.054, 0.100] | 0.105 [0.078, 0.141] |
| Comma | CP | PM_FMR | 0.243 [0.134, 0.413] | 0.384 [0.255, 0.560] | 0.200 [0.101, 0.353] | 0.233 [0.118, 0.400] |
| DFM | CP | PM_NVR | 0.217 [0.169, 0.277] | 0.523 [0.464, 0.591] | 0.190 [0.151, 0.240] | 0.100 [0.075, 0.134] |
| DFM | CP | PM_FMR | 0.288 [0.158, 0.459] | 0.657 [0.538, 0.787] | 0.606 [0.481, 0.747] | 0.441 [0.294, 0.615] |
| DFM | DW | PM_NVR | 0.508 [0.473, 0.543] | 0.372 [0.338, 0.410] | 0.017 [0.010, 0.026] | 0.068 [0.052, 0.087] |
| DFM | DW | PM_FMR | 0.439 [0.382, 0.505] | 0.362 [0.306, 0.428] | 0.029 [0.011, 0.055] | 0.121 [0.078, 0.175] |
| Olmo 3 32B | D3 | PM_NVR | 0.744 [0.674, 0.811] | 0.586 [0.497, 0.679] | 0.201 [0.130, 0.298] | 0.124 [0.054, 0.229] |
| Olmo 3 32B | D3 | PM_FMR | 0.194 [0.064, 0.467] | 0.579 [0.396, 0.820] | 0.064 [0.000, 0.235] | 0.171 [0.023, 0.454] |

PM = f_setting / (f_setting + f_prefix); 0.5 is neutral, lower values mean lower propensity relative to capability. Setting and prefix sets are resampled independently. PM_FMR inherits FMR's short full matches (see Table A); PM_FMR20 / PM_FMR50 and the long-span scores PM_R50 / PM_NVR50 are in Section 5.2.

## 1. How memorization varies across settings, per model and corpus

### Comma on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0062 [-0.0016, +0.0132] ≈ (prefix ×0.8) | -0.0038 [-0.0072, -0.0010] ↓ (prefix ×3.1) | +3.29 [+1.20, +5.29] ↑ (prefix ×0.9) |
| Minimal Cue | +0.0048 [-0.0030, +0.0118] ≈ (prefix ×0.9) | -0.0021 [-0.0056, +0.0009] ≈ (prefix ×1.6) | -3.36 [-5.43, -1.37] ↓ (prefix ×1.1) |
| Generic | -0.0268 [-0.0341, -0.0202] ↓ (prefix ×12.6) | -0.0042 [-0.0077, -0.0014] ↓ (prefix ×4.0) | -18.58 [-20.60, -16.67] ↓ (prefix ×2.3) |
| Specific | -0.0257 [-0.0330, -0.0191] ↓ (prefix ×8.5) | -0.0039 [-0.0074, -0.0011] ↓ (prefix ×3.3) | -16.23 [-18.25, -14.31] ↓ (prefix ×2.0) |

- **NVR ranking:** Unconditional (0.0353) > Minimal Cue (0.0338) > Prefix (0.0291) > Specific (0.0034) > Generic (0.0023).
- **FMR ranking:** Prefix (0.0056) > Minimal Cue (0.0035) > Unconditional (0.0018) > Specific (0.0017) > Generic (0.0014).
- **ALS ranking:** Unconditional (35.76) > Prefix (32.46) > Minimal Cue (29.10) > Specific (16.23) > Generic (13.88).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.548 [0.490, 0.614]; Minimal Cue PM_NVR = 0.538 [0.480, 0.602].

### DFM on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | -0.0188 [-0.0255, -0.0127] ↓ (prefix ×3.6) | -0.0022 [-0.0043, -0.0003] ↓ (prefix ×2.5) | -19.93 [-21.72, -18.23] ↓ (prefix ×2.9) |
| Minimal Cue | +0.0025 [-0.0044, +0.0090] ≈ (prefix ×0.9) | +0.0034 [+0.0008, +0.0058] ↑ (prefix ×0.5) | -11.66 [-13.51, -9.88] ↓ (prefix ×1.6) |
| Generic | -0.0199 [-0.0267, -0.0139] ↓ (prefix ×4.3) | +0.0020 [-0.0004, +0.0043] ≈ (prefix ×0.7) | -16.86 [-18.68, -15.16] ↓ (prefix ×2.3) |
| Specific | -0.0231 [-0.0298, -0.0171] ↓ (prefix ×9.0) | -0.0008 [-0.0030, +0.0012] ≈ (prefix ×1.3) | -15.05 [-16.89, -13.27] ↓ (prefix ×2.0) |

- **NVR ranking:** Minimal Cue (0.0286) > Prefix (0.0260) > Unconditional (0.0072) > Generic (0.0061) > Specific (0.0029).
- **FMR ranking:** Minimal Cue (0.0071) > Generic (0.0057) > Prefix (0.0037) > Specific (0.0029) > Unconditional (0.0015).
- **ALS ranking:** Prefix (30.24) > Minimal Cue (18.58) > Specific (15.19) > Generic (13.38) > Unconditional (10.31).
- **Propensity at or above neutral (PM ≥ 0.5):** Minimal Cue PM_NVR = 0.523 [0.464, 0.591]; Minimal Cue PM_FMR = 0.657 [0.538, 0.787]; Generic PM_FMR = 0.606 [0.481, 0.747].

### DFM on Dynaword

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0015 [-0.0052, +0.0077] ≈ (prefix ×1.0) | -0.0080 [-0.0170, +0.0006] ≈ (prefix ×1.3) | +0.81 [-0.68, +2.26] ≈ (prefix ×1.0) |
| Minimal Cue | -0.0187 [-0.0252, -0.0126] ↓ (prefix ×1.7) | -0.0159 [-0.0248, -0.0076] ↓ (prefix ×1.8) | -5.18 [-6.65, -3.78] ↓ (prefix ×1.3) |
| Generic | -0.0452 [-0.0514, -0.0394] ↓ (prefix ×57.3) | -0.0356 [-0.0440, -0.0278] ↓ (prefix ×33.3) | -9.91 [-11.38, -8.51] ↓ (prefix ×1.6) |
| Specific | -0.0426 [-0.0489, -0.0368] ↓ (prefix ×13.7) | -0.0317 [-0.0403, -0.0237] ↓ (prefix ×7.3) | -8.15 [-9.75, -6.59] ↓ (prefix ×1.5) |

- **NVR ranking:** Unconditional (0.0475) > Prefix (0.0460) > Minimal Cue (0.0273) > Specific (0.0034) > Generic (0.0008).
- **FMR ranking:** Prefix (0.0367) > Unconditional (0.0287) > Minimal Cue (0.0208) > Specific (0.0050) > Generic (0.0011).
- **ALS ranking:** Unconditional (26.25) > Prefix (25.44) > Minimal Cue (20.25) > Specific (17.29) > Generic (15.53).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.508 [0.473, 0.543].

### Olmo 3 32B on Dolma 3

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0163 [+0.0127, +0.0194] ↑ (prefix ×0.3) | -0.0022 [-0.0050, -0.0001] ↓ (prefix ×4.1) | +26.53 [+24.81, +28.15] ↑ (prefix ×0.5) |
| Minimal Cue | +0.0035 [-0.0001, +0.0065] ≈ (prefix ×0.7) | +0.0011 [-0.0019, +0.0035] ≈ (prefix ×0.7) | -0.36 [-2.02, +1.21] ≈ (prefix ×1.0) |
| Generic | -0.0064 [-0.0099, -0.0035] ↓ (prefix ×4.0) | -0.0027 [-0.0055, -0.0007] ↓ (prefix ×14.5) | -11.80 [-13.36, -10.38] ↓ (prefix ×1.8) |
| Specific | -0.0073 [-0.0108, -0.0044] ↓ (prefix ×7.1) | -0.0023 [-0.0052, -0.0001] ↓ (prefix ×4.8) | -9.22 [-10.83, -7.75] ↓ (prefix ×1.5) |

- **NVR ranking:** Unconditional (0.0249) > Minimal Cue (0.0121) > Prefix (0.0085) > Generic (0.0021) > Specific (0.0012).
- **FMR ranking:** Minimal Cue (0.0040) > Prefix (0.0029) > Unconditional (0.0007) > Specific (0.0006) > Generic (0.0002).
- **ALS ranking:** Unconditional (53.42) > Prefix (26.90) > Minimal Cue (26.54) > Specific (17.68) > Generic (15.10).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.744 [0.674, 0.811]; Minimal Cue PM_NVR = 0.586 [0.497, 0.679]; Minimal Cue PM_FMR = 0.579 [0.396, 0.820].

## 2. Continual pre-training: DFM against Comma on Common Pile (cf. paper Section 5.2)

DFM Decoder is Comma continually pre-trained on two-thirds Dynaword and one-third Common Pile. Paired differences DFM − Comma on the same prompts (unconditional: independent samples):

### DFM Stage 1 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0256 [-0.0288, -0.0225] ↓ | +0.0034 [+0.0018, +0.0051] ↑ | -25.27 [-25.99, -24.57] ↓ | independent |
| Minimal Cue | -0.0063 [-0.0089, -0.0037] ↓ | +0.0027 [+0.0009, +0.0045] ↑ | -10.25 [-10.85, -9.65] ↓ | paired |
| Generic | +0.0034 [+0.0025, +0.0044] ↑ | +0.0045 [+0.0029, +0.0061] ↑ | -0.60 [-1.03, -0.18] ↓ | paired |
| Specific | -0.0002 [-0.0011, +0.0006] ≈ | +0.0018 [+0.0005, +0.0031] ↑ | -1.01 [-1.61, -0.38] ↓ | paired |
| Prefix | -0.0032 [-0.0060, -0.0007] ↓ | -0.0014 [-0.0041, +0.0008] ≈ | -2.67 [-3.57, -1.79] ↓ | paired |

### DFM Stage 2 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0301 [-0.0330, -0.0272] ↓ | +0.0010 [-0.0003, +0.0024] ≈ | -25.70 [-26.40, -25.00] ↓ | independent |
| Minimal Cue | -0.0067 [-0.0093, -0.0040] ↓ | +0.0026 [+0.0008, +0.0044] ↑ | -10.06 [-10.67, -9.45] ↓ | paired |
| Generic | +0.0028 [+0.0019, +0.0036] ↑ | +0.0031 [+0.0018, +0.0044] ↑ | -0.61 [-1.02, -0.20] ↓ | paired |
| Specific | -0.0002 [-0.0010, +0.0006] ≈ | +0.0012 [+0.0000, +0.0024] ↑ | -0.87 [-1.44, -0.28] ↓ | paired |
| Prefix | -0.0022 [-0.0050, +0.0004] ≈ | -0.0009 [-0.0028, +0.0007] ≈ | -2.23 [-3.09, -1.38] ↓ | paired |

### DFM − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0280 [-0.0311, -0.0251] ↓ | -0.0003 [-0.0014, +0.0009] ≈ | -25.45 [-26.15, -24.74] ↓ | independent |
| Minimal Cue | -0.0053 [-0.0080, -0.0026] ↓ | +0.0036 [+0.0017, +0.0055] ↑ | -10.52 [-11.12, -9.92] ↓ | paired |
| Generic | +0.0038 [+0.0029, +0.0048] ↑ | +0.0043 [+0.0028, +0.0058] ↑ | -0.50 [-0.94, -0.08] ↓ | paired |
| Specific | -0.0005 [-0.0013, +0.0003] ≈ | +0.0012 [+0.0000, +0.0024] ↑ | -1.04 [-1.58, -0.50] ↓ | paired |
| Prefix | -0.0030 [-0.0056, -0.0005] ↓ | -0.0019 [-0.0044, +0.0001] ≈ | -2.22 [-3.06, -1.42] ↓ | paired |

### Propensity shift, Comma → DFM (final)

| Prompt | PM_NVR | PM_FMR |
|---|---|---|
| Unconditional | 0.548 [0.490, 0.614] → 0.217 [0.169, 0.277]; Δ -0.331 [-0.378, -0.284] ↓ | 0.243 [0.134, 0.413] → 0.288 [0.158, 0.459]; Δ +0.045 [-0.123, +0.206] ≈ |
| Minimal Cue | 0.538 [0.480, 0.602] → 0.523 [0.464, 0.591]; Δ -0.015 [-0.047, +0.017] ≈ | 0.384 [0.255, 0.560] → 0.657 [0.538, 0.787]; Δ +0.273 [+0.133, +0.402] ↑ |
| Generic | 0.073 [0.054, 0.100] → 0.190 [0.151, 0.240]; Δ +0.117 [+0.087, +0.154] ↑ | 0.200 [0.101, 0.353] → 0.606 [0.481, 0.747]; Δ +0.406 [+0.274, +0.533] ↑ |
| Specific | 0.105 [0.078, 0.141] → 0.100 [0.075, 0.134]; Δ -0.005 [-0.031, +0.020] ≈ | 0.233 [0.118, 0.400] → 0.441 [0.294, 0.615]; Δ +0.208 [+0.054, +0.356] ↑ |

## 3. Common Pile and Dynaword memorization profiles of DFM (cf. paper Section 5.2)

CP / DW values of DFM (final) and their difference CP − DW (different prompt sets: independent resampling):

| Prompt | ALS CP / DW | FMR CP / DW | NVR CP / DW |
|---|---|---|---|
| Unconditional | 10.31 / 26.25; Δ -15.94 [-16.46, -15.42] ↓ | 0.0015 / 0.0287; Δ -0.0272 [-0.0306, -0.0239] ↓ | 0.0072 / 0.0475; Δ -0.0402 [-0.0432, -0.0373] ↓ |
| Minimal Cue | 18.58 / 20.25; Δ -1.67 [-2.28, -1.05] ↓ | 0.0071 / 0.0208; Δ -0.0137 [-0.0170, -0.0105] ↓ | 0.0286 / 0.0273; Δ +0.0013 [-0.0020, +0.0046] ≈ |
| Generic | 13.38 / 15.53; Δ -2.15 [-2.49, -1.79] ↓ | 0.0057 / 0.0011; Δ +0.0046 [+0.0029, +0.0063] ↑ | 0.0061 / 0.0008; Δ +0.0053 [+0.0043, +0.0064] ↑ |
| Specific | 15.19 / 17.29; Δ -2.10 [-2.97, -1.25] ↓ | 0.0029 / 0.0050; Δ -0.0021 [-0.0044, +0.0001] ≈ | 0.0029 / 0.0034; Δ -0.0005 [-0.0015, +0.0005] ≈ |
| Prefix | 30.24 / 25.44; Δ +4.80 [+2.53, +7.07] ↑ | 0.0037 / 0.0367; Δ -0.0330 [-0.0417, -0.0249] ↓ | 0.0260 / 0.0460; Δ -0.0200 [-0.0285, -0.0113] ↓ |

## 4. Memorization across DFM training stages (cf. paper Section 5.3 and Appendix C)

### Dynaword

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0430 [0.0406, 0.0455] | 0.0363 [0.0327, 0.0399] | 24.73 [24.34, 25.13] | 0.529 [0.493, 0.566] | 0.520 [0.462, 0.585] |
| Unconditional | Stage 2 | 0.0451 [0.0426, 0.0476] | 0.0280 [0.0248, 0.0313] | 25.72 [25.31, 26.14] | 0.511 [0.476, 0.548] | 0.447 [0.388, 0.513] |
| Unconditional | Final | 0.0475 [0.0450, 0.0501] | 0.0287 [0.0255, 0.0320] | 26.25 [25.82, 26.69] | 0.508 [0.473, 0.543] | 0.439 [0.382, 0.505] |
| Minimal Cue | Stage 1 | 0.0245 [0.0225, 0.0265] | 0.0206 [0.0178, 0.0234] | 19.43 [19.17, 19.71] | 0.390 [0.353, 0.429] | 0.381 [0.322, 0.448] |
| Minimal Cue | Stage 2 | 0.0244 [0.0225, 0.0265] | 0.0192 [0.0166, 0.0220] | 19.73 [19.46, 20.01] | 0.362 [0.326, 0.400] | 0.356 [0.301, 0.423] |
| Minimal Cue | Final | 0.0273 [0.0253, 0.0294] | 0.0208 [0.0180, 0.0237] | 20.25 [19.98, 20.53] | 0.372 [0.338, 0.410] | 0.362 [0.306, 0.428] |
| Generic | Stage 1 | 0.0005 [0.0003, 0.0007] | 0.0009 [0.0002, 0.0019] | 15.52 [15.29, 15.76] | 0.012 [0.008, 0.018] | 0.026 [0.006, 0.057] |
| Generic | Stage 2 | 0.0006 [0.0004, 0.0010] | 0.0012 [0.0005, 0.0022] | 15.55 [15.33, 15.78] | 0.015 [0.009, 0.022] | 0.033 [0.012, 0.062] |
| Generic | Final | 0.0008 [0.0005, 0.0012] | 0.0011 [0.0004, 0.0020] | 15.53 [15.34, 15.72] | 0.017 [0.010, 0.026] | 0.029 [0.011, 0.055] |
| Specific | Stage 1 | 0.0030 [0.0023, 0.0039] | 0.0037 [0.0022, 0.0054] | 16.68 [16.07, 17.30] | 0.074 [0.056, 0.095] | 0.100 [0.061, 0.149] |
| Specific | Stage 2 | 0.0036 [0.0026, 0.0047] | 0.0049 [0.0029, 0.0073] | 16.75 [16.15, 17.35] | 0.077 [0.056, 0.102] | 0.124 [0.074, 0.186] |
| Specific | Final | 0.0034 [0.0026, 0.0043] | 0.0050 [0.0032, 0.0072] | 17.29 [16.62, 18.00] | 0.068 [0.052, 0.087] | 0.121 [0.078, 0.175] |
| Prefix | Stage 1 | 0.0383 [0.0333, 0.0436] | 0.0335 [0.0260, 0.0412] | 22.81 [21.73, 23.97] | — | — |
| Prefix | Stage 2 | 0.0431 [0.0375, 0.0489] | 0.0347 [0.0269, 0.0427] | 24.10 [22.87, 25.37] | — | — |
| Prefix | Final | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.44 [24.05, 26.89] | — | — |

Significant paired differences from the final checkpoint (14 of 30 stage–setting–metric comparisons):

- Stage 1 − Final, Unconditional, NVR: -0.0045 [-0.0080, -0.0009] ↓
- Stage 1 − Final, Unconditional, FMR: +0.0076 [+0.0027, +0.0125] ↑
- Stage 1 − Final, Unconditional, ALS: -1.52 [-2.11, -0.93] ↓
- Stage 1 − Final, Minimal Cue, NVR: -0.0028 [-0.0047, -0.0009] ↓
- Stage 1 − Final, Minimal Cue, ALS: -0.82 [-1.08, -0.55] ↓
- Stage 2 − Final, Minimal Cue, NVR: -0.0029 [-0.0047, -0.0010] ↓
- Stage 2 − Final, Minimal Cue, ALS: -0.52 [-0.78, -0.26] ↓
- Stage 1 − Final, Specific, FMR: -0.0013 [-0.0026, -0.0001] ↓
- Stage 1 − Final, Specific, ALS: -0.61 [-1.15, -0.07] ↓
- Stage 2 − Final, Specific, ALS: -0.55 [-1.04, -0.09] ↓
- Stage 1 − Final, Prefix, NVR: -0.0078 [-0.0114, -0.0044] ↓
- Stage 1 − Final, Prefix, ALS: -2.62 [-3.37, -1.94] ↓
- Stage 2 − Final, Prefix, NVR: -0.0029 [-0.0053, -0.0006] ↓
- Stage 2 − Final, Prefix, ALS: -1.34 [-1.87, -0.86] ↓

### Common Pile

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0096 [0.0079, 0.0115] | 0.0052 [0.0038, 0.0067] | 10.48 [10.21, 10.77] | 0.271 [0.219, 0.336] | 0.553 [0.420, 0.709] |
| Unconditional | Stage 2 | 0.0052 [0.0040, 0.0063] | 0.0028 [0.0018, 0.0039] | 10.06 [9.79, 10.34] | 0.161 [0.122, 0.211] | 0.373 [0.247, 0.542] |
| Unconditional | Final | 0.0072 [0.0058, 0.0087] | 0.0015 [0.0008, 0.0023] | 10.31 [10.03, 10.60] | 0.217 [0.169, 0.277] | 0.288 [0.158, 0.459] |
| Minimal Cue | Stage 1 | 0.0275 [0.0250, 0.0301] | 0.0062 [0.0047, 0.0078] | 18.85 [18.32, 19.40] | 0.515 [0.456, 0.581] | 0.596 [0.464, 0.742] |
| Minimal Cue | Stage 2 | 0.0272 [0.0247, 0.0298] | 0.0061 [0.0046, 0.0077] | 19.05 [18.52, 19.61] | 0.503 [0.444, 0.571] | 0.564 [0.438, 0.713] |
| Minimal Cue | Final | 0.0286 [0.0260, 0.0313] | 0.0071 [0.0055, 0.0087] | 18.58 [18.04, 19.14] | 0.523 [0.464, 0.591] | 0.657 [0.538, 0.787] |
| Generic | Stage 1 | 0.0057 [0.0048, 0.0068] | 0.0059 [0.0043, 0.0075] | 13.28 [12.99, 13.59] | 0.181 [0.143, 0.229] | 0.584 [0.452, 0.737] |
| Generic | Stage 2 | 0.0051 [0.0043, 0.0059] | 0.0045 [0.0032, 0.0059] | 13.27 [12.98, 13.57] | 0.159 [0.125, 0.202] | 0.489 [0.356, 0.652] |
| Generic | Final | 0.0061 [0.0052, 0.0071] | 0.0057 [0.0042, 0.0073] | 13.38 [13.10, 13.67] | 0.190 [0.151, 0.240] | 0.606 [0.481, 0.747] |
| Specific | Stage 1 | 0.0032 [0.0025, 0.0040] | 0.0035 [0.0024, 0.0047] | 15.22 [14.65, 15.83] | 0.110 [0.081, 0.147] | 0.456 [0.316, 0.630] |
| Specific | Stage 2 | 0.0032 [0.0026, 0.0039] | 0.0029 [0.0019, 0.0040] | 15.36 [14.82, 15.93] | 0.108 [0.081, 0.144] | 0.383 [0.254, 0.552] |
| Specific | Final | 0.0029 [0.0023, 0.0035] | 0.0029 [0.0018, 0.0041] | 15.19 [14.70, 15.70] | 0.100 [0.075, 0.134] | 0.441 [0.294, 0.615] |
| Prefix | Stage 1 | 0.0259 [0.0201, 0.0320] | 0.0042 [0.0022, 0.0066] | 29.79 [28.12, 31.55] | — | — |
| Prefix | Stage 2 | 0.0268 [0.0208, 0.0333] | 0.0047 [0.0026, 0.0072] | 30.23 [28.50, 32.06] | — | — |
| Prefix | Final | 0.0260 [0.0202, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.24 [28.55, 32.01] | — | — |

Significant paired differences from the final checkpoint (6 of 30 stage–setting–metric comparisons):

- Stage 1 − Final, Unconditional, NVR: +0.0024 [+0.0001, +0.0047] ↑
- Stage 1 − Final, Unconditional, FMR: +0.0037 [+0.0021, +0.0054] ↑
- Stage 2 − Final, Unconditional, NVR: -0.0021 [-0.0040, -0.0003] ↓
- Stage 2 − Final, Unconditional, FMR: +0.0013 [+0.0000, +0.0026] ↑
- Stage 2 − Final, Minimal Cue, ALS: +0.46 [+0.10, +0.83] ↑
- Stage 2 − Final, Generic, NVR: -0.0011 [-0.0019, -0.0002] ↓

## 5. Robustness: predictability, boilerplate and repeated sampling

Analyses of the existing traces that address two objections: that matches reflect predictable or duplicated text rather than memorization (Cooper et al., 2026; Huang et al., 2024), and that single samples misstate extraction (Hayes et al., 2025). Same bootstrap as above (prompts or generations resampled, B = 10,000). Degenerate outputs are excluded as in the main tables.

### 5.1 Predictability floor: non-member text

The prompt texts themselves are not in the training data (verified), so the spans they match there show what chance alone produces for short natural sentences.

| Index | ALS of generic prompts (n) | ALS of specific prompts (n) | Specific prompts with a ≥ 20-token span | Longest span |
|---|---|---|---|---|
| Common Pile | 6.59 (1642) | 6.54 (1000) | 0.0% | 19 |
| Dolma 3 | 7.35 (1174) | 7.32 (1000) | 0.8% | 49 |
| Dynaword | 7.60 (1664) | 8.03 (1000) | 0.8% | 30 |

Spans of about 6–8 tokens arise without memorization; prompts are short (~12 tokens), so the floor for longer spans is not measured here.

### 5.2 Long spans only

R20 / R50: share of generations whose longest training span has ≥ 20 / ≥ 50 tokens. NVR50: NVR counting only documents retrieved through ≥ 50-token spans (over all retrieved documents). FMR20 / FMR50: full matches whose generation is ≥ 20 / ≥ 50 tokens long. Cooper et al. (2026) find that, under targeted extraction, text not in the training data has its true 50-token continuation reproduced 0.02% of the time, against 0.74% for training text (OLMo 2 32B, Wikipedia, greedy decoding); at 10 tokens the non-training rate is about 24% of the training rate. 50 tokens is therefore used as a conservative threshold motivated by their result, not as a calibrated false-positive rate: these metrics count matches with any training document, and no matched control was run here. Our own baseline (5.1) only shows that non-training prompts match spans of about 6.5–8 tokens on average.

Spans that are repetition loops do not count here or in the rest of Section 5: after collapsing whitespace, one unit of ≤ 200 characters repeated at least 3 times with ≥ 90% of characters equal one period later (e.g. "I'm sorry. I'm sorry. …", "Hvad er der i vejen?" on every line, `| | |`, `rrrr`). Such loops match repetitive training text over 50–360 tokens without reproducing any document's content. They are tested on spans of ≥ 20 tokens; code or prose whose repeated parts vary (names, numbers) is not periodic. The last column counts the generations whose ≥ 50-token spans were all loops, among all generations with a ≥ 50-token span.

| Model | Data | Prompt | R20 [95% CI] | R50 [95% CI] | NVR50 [95% CI] | Loop-only ≥ 50-token matches |
|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 67.2% [66.3, 68.1] | 17.7% [16.9, 18.4] | 0.0116 [0.0106, 0.0126] | 10 of 1777 |
| Comma | CP | Minimal Cue | 44.2% [43.3, 45.2] | 10.9% [10.3, 11.5] | 0.0110 [0.0100, 0.0121] | 16 of 1103 |
| Comma | CP | Generic | 7.7% [7.0, 8.5] | 0.9% [0.7, 1.2] | 0.0006 [0.0004, 0.0009] | 36 of 130 |
| Comma | CP | Specific | 15.5% [14.4, 16.7] | 2.1% [1.8, 2.5] | 0.0011 [0.0008, 0.0014] | 15 of 225 |
| Comma | CP | Prefix | 53.5% [51.1, 55.9] | 13.6% [12.0, 15.2] | 0.0099 [0.0074, 0.0125] | 52 of 1410 |
| DFM | CP | Unconditional | 6.7% [6.2, 7.2] | 1.9% [1.7, 2.2] | 0.0023 [0.0018, 0.0029] | 4 of 198 |
| DFM | CP | Minimal Cue | 19.4% [18.6, 20.1] | 6.5% [6.0, 7.0] | 0.0086 [0.0075, 0.0096] | 8 of 655 |
| DFM | CP | Generic | 9.4% [8.6, 10.2] | 0.6% [0.4, 0.8] | 0.0005 [0.0003, 0.0007] | 23 of 82 |
| DFM | CP | Specific | 12.6% [11.6, 13.7] | 1.4% [1.1, 1.7] | 0.0007 [0.0005, 0.0010] | 24 of 163 |
| DFM | CP | Prefix | 51.0% [48.6, 53.4] | 11.4% [9.9, 13.0] | 0.0088 [0.0063, 0.0114] | 45 of 1184 |
| DFM | DW | Unconditional | 49.4% [48.4, 50.4] | 11.0% [10.4, 11.6] | 0.0093 [0.0084, 0.0101] | 2 of 1102 |
| DFM | DW | Minimal Cue | 33.9% [33.0, 34.9] | 3.3% [2.9, 3.6] | 0.0039 [0.0033, 0.0047] | 3 of 328 |
| DFM | DW | Generic | 15.1% [14.1, 16.0] | 0.4% [0.3, 0.5] | 0.0001 [0.0000, 0.0002] | 8 of 45 |
| DFM | DW | Specific | 21.4% [19.9, 23.0] | 2.8% [2.3, 3.4] | 0.0012 [0.0008, 0.0017] | 19 of 301 |
| DFM | DW | Prefix | 41.6% [39.0, 44.1] | 10.1% [8.6, 11.7] | 0.0108 [0.0082, 0.0135] | 2 of 1010 |
| Olmo 3 32B | D3 | Unconditional | 84.3% [83.6, 85.0] | 41.4% [40.4, 42.4] | 0.0111 [0.0103, 0.0118] | 34 of 4170 |
| Olmo 3 32B | D3 | Minimal Cue | 39.4% [38.4, 40.3] | 8.6% [8.1, 9.2] | 0.0048 [0.0042, 0.0053] | 5 of 868 |
| Olmo 3 32B | D3 | Generic | 8.9% [8.2, 9.7] | 0.8% [0.5, 1.1] | 0.0005 [0.0002, 0.0010] | 17 of 94 |
| Olmo 3 32B | D3 | Specific | 20.4% [19.2, 21.7] | 1.7% [1.3, 2.1] | 0.0004 [0.0002, 0.0007] | 2 of 172 |
| Olmo 3 32B | D3 | Prefix | 46.0% [43.8, 48.3] | 9.0% [7.6, 10.5] | 0.0036 [0.0024, 0.0050] | 8 of 909 |

Full matches by length. Most full matches under generic and specific prompts are short outputs that end early and occur verbatim in the training data (e.g. "He is a gentleman.", "Havde I det sjovt?"); FMR20 and FMR50 count only full matches of ≥ 20 / ≥ 50 tokens (loops excluded).

| Model | Data | Prompt | FMR (all) [95% CI] | FMR20 [95% CI] | FMR50 [95% CI] |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 0.0018 [0.0010, 0.0027] | 0.0014 [0.0007, 0.0022] | 0.0012 [0.0006, 0.0019] |
| Comma | CP | Minimal Cue | 0.0035 [0.0024, 0.0047] | 0.0032 [0.0021, 0.0044] | 0.0030 [0.0020, 0.0041] |
| Comma | CP | Generic | 0.0014 [0.0007, 0.0022] | 0.0001 [0.0000, 0.0003] | 0.0001 [0.0000, 0.0003] |
| Comma | CP | Specific | 0.0017 [0.0009, 0.0026] | 0.0001 [0.0000, 0.0003] | 0.0001 [0.0000, 0.0003] |
| Comma | CP | Prefix | 0.0056 [0.0029, 0.0090] | 0.0033 [0.0008, 0.0064] | 0.0032 [0.0008, 0.0063] |
| DFM | CP | Unconditional | 0.0015 [0.0008, 0.0023] | 0.0002 [0.0000, 0.0005] | 0.0002 [0.0000, 0.0005] |
| DFM | CP | Minimal Cue | 0.0071 [0.0055, 0.0088] | 0.0015 [0.0008, 0.0023] | 0.0015 [0.0008, 0.0023] |
| DFM | CP | Generic | 0.0057 [0.0042, 0.0073] | 0.0002 [0.0000, 0.0005] | 0.0000 [0.0000, 0.0000] |
| DFM | CP | Specific | 0.0029 [0.0018, 0.0041] | 0.0002 [0.0000, 0.0005] | 0.0001 [0.0000, 0.0003] |
| DFM | CP | Prefix | 0.0037 [0.0020, 0.0056] | 0.0014 [0.0003, 0.0029] | 0.0014 [0.0003, 0.0029] |
| DFM | DW | Unconditional | 0.0287 [0.0254, 0.0321] | 0.0070 [0.0054, 0.0087] | 0.0018 [0.0010, 0.0027] |
| DFM | DW | Minimal Cue | 0.0208 [0.0181, 0.0236] | 0.0026 [0.0017, 0.0037] | 0.0009 [0.0004, 0.0016] |
| DFM | DW | Generic | 0.0011 [0.0004, 0.0020] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| DFM | DW | Specific | 0.0050 [0.0032, 0.0072] | 0.0001 [0.0000, 0.0003] | 0.0000 [0.0000, 0.0000] |
| DFM | DW | Prefix | 0.0367 [0.0290, 0.0451] | 0.0085 [0.0047, 0.0130] | 0.0052 [0.0021, 0.0090] |
| Olmo 3 32B | D3 | Unconditional | 0.0007 [0.0002, 0.0013] | 0.0007 [0.0002, 0.0013] | 0.0007 [0.0002, 0.0013] |
| Olmo 3 32B | D3 | Minimal Cue | 0.0040 [0.0028, 0.0052] | 0.0037 [0.0025, 0.0049] | 0.0036 [0.0025, 0.0048] |
| Olmo 3 32B | D3 | Generic | 0.0002 [0.0000, 0.0005] | 0.0001 [0.0000, 0.0003] | 0.0001 [0.0000, 0.0003] |
| Olmo 3 32B | D3 | Specific | 0.0006 [0.0001, 0.0014] | 0.0003 [0.0000, 0.0009] | 0.0003 [0.0000, 0.0009] |
| Olmo 3 32B | D3 | Prefix | 0.0029 [0.0009, 0.0057] | 0.0019 [0.0001, 0.0045] | 0.0019 [0.0001, 0.0045] |

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_R50 | 0.565 [0.536, 0.598] | 0.444 [0.413, 0.479] | 0.065 [0.048, 0.085] | 0.134 [0.112, 0.159] |
| Comma | CP | PM_NVR50 | 0.540 [0.476, 0.612] | 0.528 [0.463, 0.601] | 0.057 [0.035, 0.088] | 0.101 [0.071, 0.140] |
| Comma | CP | PM_FMR (all) | 0.243 [0.133, 0.411] | 0.384 [0.256, 0.565] | 0.200 [0.102, 0.355] | 0.233 [0.121, 0.397] |
| Comma | CP | PM_FMR20 | 0.298 [0.140, 0.636] | 0.492 [0.309, 0.792] | 0.029 [0.000, 0.150] | 0.029 [0.000, 0.154] |
| Comma | CP | PM_FMR50 | 0.273 [0.120, 0.625] | 0.484 [0.297, 0.800] | 0.030 [0.000, 0.166] | 0.030 [0.000, 0.167] |
| DFM | CP | PM_R50 | 0.145 [0.123, 0.170] | 0.362 [0.329, 0.399] | 0.049 [0.036, 0.064] | 0.109 [0.087, 0.135] |
| DFM | CP | PM_NVR50 | 0.208 [0.155, 0.278] | 0.494 [0.421, 0.579] | 0.054 [0.032, 0.085] | 0.077 [0.052, 0.113] |
| DFM | CP | PM_FMR (all) | 0.288 [0.159, 0.454] | 0.657 [0.542, 0.784] | 0.606 [0.481, 0.746] | 0.441 [0.295, 0.613] |
| DFM | CP | PM_FMR20 | 0.125 [0.000, 0.499] | 0.517 [0.289, 0.852] | 0.125 [0.000, 0.500] | 0.126 [0.000, 0.501] |
| DFM | CP | PM_FMR50 | 0.125 [0.000, 0.499] | 0.517 [0.289, 0.852] | 0.000 [0.000, 0.000] | 0.067 [0.000, 0.335] |
| DFM | DW | PM_R50 | 0.522 [0.482, 0.564] | 0.244 [0.211, 0.280] | 0.035 [0.023, 0.050] | 0.220 [0.180, 0.264] |
| DFM | DW | PM_NVR50 | 0.462 [0.400, 0.532] | 0.268 [0.214, 0.334] | 0.011 [0.002, 0.024] | 0.103 [0.068, 0.146] |
| DFM | DW | PM_FMR (all) | 0.439 [0.381, 0.504] | 0.362 [0.306, 0.426] | 0.029 [0.011, 0.055] | 0.121 [0.077, 0.174] |
| DFM | DW | PM_FMR20 | 0.452 [0.333, 0.607] | 0.234 [0.142, 0.380] | 0.000 [0.000, 0.000] | 0.012 [0.000, 0.045] |
| DFM | DW | PM_FMR50 | 0.257 [0.136, 0.476] | 0.148 [0.057, 0.333] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Olmo 3 32B | D3 | PM_R50 | 0.821 [0.798, 0.844] | 0.489 [0.448, 0.532] | 0.079 [0.054, 0.108] | 0.159 [0.122, 0.199] |
| Olmo 3 32B | D3 | PM_NVR50 | 0.756 [0.686, 0.824] | 0.571 [0.481, 0.670] | 0.119 [0.045, 0.239] | 0.107 [0.053, 0.183] |
| Olmo 3 32B | D3 | PM_FMR (all) | 0.194 [0.063, 0.467] | 0.579 [0.393, 0.816] | 0.064 [0.000, 0.235] | 0.171 [0.020, 0.454] |
| Olmo 3 32B | D3 | PM_FMR20 | 0.269 [0.083, 0.889] | 0.660 [0.428, 0.976] | 0.050 [0.000, 0.500] | 0.136 [0.000, 0.750] |
| Olmo 3 32B | D3 | PM_FMR50 | 0.269 [0.083, 0.889] | 0.654 [0.422, 0.975] | 0.050 [0.000, 0.500] | 0.136 [0.000, 0.750] |

PM_FMR (all) differs slightly from Table B: Table B resamples setting and prefix with separate random streams of bootstrap_ci.py; the estimate is the same.

### 5.3 Natural-language generations only (code and symbol-heavy text removed)

A generation is code-like if ≥ 3 lines (and ≥ 30% of its lines) look like code — imports, declarations, braces, semicolons, comments — or if under 50% of its characters are letters (number sequences, tables). Metrics below are recomputed on the remaining generations.

Limit of the heuristic: random samples of the remaining ≥ 50-token spans in the prompt-free settings still contain much templated text — open-access license and citation boilerplate, wiki page templates, methods-section boilerplate and task-instruction templates for Comma; coding-exercise statements, software licenses and SQL dumps for Olmo 3. How often these spans occur in the training data is measured in 5.7.

| Model | Data | Prompt | Code-like | NVR [95% CI] | ALS [95% CI] | R50 [95% CI] |
|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 29.2% | 0.0290 [0.0262, 0.0319] | 29.82 [29.20, 30.47] | 10.7% [10.0, 11.5] |
| Comma | CP | Minimal Cue | 8.9% | 0.0366 [0.0338, 0.0395] | 28.87 [28.20, 29.57] | 10.9% [10.2, 11.5] |
| Comma | CP | Generic | 4.9% | 0.0023 [0.0018, 0.0028] | 13.54 [13.21, 13.93] | 0.8% [0.6, 1.1] |
| Comma | CP | Specific | 5.1% | 0.0033 [0.0025, 0.0041] | 15.29 [14.93, 15.68] | 1.7% [1.4, 2.0] |
| Comma | CP | Prefix | 47.1% | 0.0203 [0.0140, 0.0276] | 24.70 [22.76, 26.85] | 7.9% [6.2, 9.7] |
| DFM | CP | Unconditional | 4.0% | 0.0053 [0.0040, 0.0067] | 8.97 [8.77, 9.18] | 1.0% [0.8, 1.2] |
| DFM | CP | Minimal Cue | 8.2% | 0.0316 [0.0287, 0.0345] | 18.06 [17.47, 18.65] | 6.5% [6.0, 7.0] |
| DFM | CP | Generic | 5.6% | 0.0063 [0.0054, 0.0074] | 13.06 [12.79, 13.35] | 0.5% [0.4, 0.7] |
| DFM | CP | Specific | 5.4% | 0.0027 [0.0021, 0.0034] | 14.06 [13.71, 14.44] | 1.1% [0.9, 1.4] |
| DFM | CP | Prefix | 47.0% | 0.0169 [0.0109, 0.0238] | 21.80 [20.26, 23.51] | 5.5% [4.1, 7.0] |
| DFM | DW | Unconditional | 4.0% | 0.0512 [0.0485, 0.0539] | 27.00 [26.55, 27.44] | 11.5% [10.8, 12.1] |
| DFM | DW | Minimal Cue | 0.7% | 0.0276 [0.0255, 0.0297] | 20.25 [19.98, 20.53] | 3.3% [2.9, 3.6] |
| DFM | DW | Generic | 0.5% | 0.0008 [0.0005, 0.0012] | 15.51 [15.32, 15.70] | 0.4% [0.2, 0.5] |
| DFM | DW | Specific | 1.4% | 0.0033 [0.0025, 0.0041] | 16.67 [16.08, 17.27] | 2.7% [2.2, 3.2] |
| DFM | DW | Prefix | 2.0% | 0.0464 [0.0406, 0.0527] | 25.56 [24.17, 27.04] | 10.2% [8.7, 11.8] |
| Olmo 3 32B | D3 | Unconditional | 82.3% | 0.0207 [0.0163, 0.0255] | 26.34 [24.51, 28.29] | 8.7% [7.4, 10.1] |
| Olmo 3 32B | D3 | Minimal Cue | 10.9% | 0.0116 [0.0103, 0.0129] | 25.44 [24.78, 26.14] | 7.7% [7.1, 8.2] |
| Olmo 3 32B | D3 | Generic | 0.6% | 0.0021 [0.0014, 0.0031] | 15.02 [14.77, 15.31] | 0.7% [0.5, 1.0] |
| Olmo 3 32B | D3 | Specific | 0.8% | 0.0011 [0.0005, 0.0022] | 17.38 [16.96, 17.84] | 1.6% [1.2, 2.0] |
| Olmo 3 32B | D3 | Prefix | 16.0% | 0.0075 [0.0044, 0.0114] | 24.16 [22.87, 25.58] | 6.3% [5.0, 7.7] |

| Model | Data | Metric (natural language) | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_NVR | 0.588 [0.508, 0.679] | 0.643 [0.568, 0.725] | 0.100 [0.070, 0.146] | 0.139 [0.098, 0.199] |
| Comma | CP | PM_R50 | 0.576 [0.523, 0.634] | 0.579 [0.526, 0.636] | 0.095 [0.066, 0.133] | 0.175 [0.138, 0.222] |
| DFM | CP | PM_NVR | 0.239 [0.168, 0.340] | 0.652 [0.566, 0.744] | 0.273 [0.203, 0.372] | 0.140 [0.096, 0.206] |
| DFM | CP | PM_R50 | 0.155 [0.117, 0.205] | 0.543 [0.479, 0.618] | 0.090 [0.062, 0.128] | 0.172 [0.128, 0.231] |
| DFM | DW | PM_NVR | 0.524 [0.490, 0.560] | 0.373 [0.339, 0.409] | 0.017 [0.010, 0.026] | 0.066 [0.050, 0.084] |
| DFM | DW | PM_R50 | 0.530 [0.490, 0.571] | 0.242 [0.210, 0.278] | 0.034 [0.022, 0.048] | 0.210 [0.170, 0.253] |
| Olmo 3 32B | D3 | PM_NVR | 0.733 [0.628, 0.830] | 0.606 [0.499, 0.727] | 0.220 [0.135, 0.350] | 0.130 [0.051, 0.258] |
| Olmo 3 32B | D3 | PM_R50 | 0.581 [0.518, 0.645] | 0.550 [0.497, 0.608] | 0.106 [0.072, 0.149] | 0.203 [0.155, 0.260] |

### 5.4 Concentration of long matches

For generations with a ≥ 50-token span: distinct training documents retrieved for that span (≤ 10 per span), and the share of these generations whose span retrieved one of the 10 most frequently hit documents. A high share means a few documents account for the matches. A low share does not rule out duplication: text present in many documents (e.g. code boilerplate) retrieves a different sample of them each time, so it is measured with exact occurrence counts in 5.7.

| Model | Data | Prompt | Generations with R50 | Distinct documents | Top-10 documents' share |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 1767 | 5839 | 1% |
| Comma | CP | Minimal Cue | 1087 | 3972 | 1% |
| Comma | CP | Generic | 94 | 432 | 16% |
| Comma | CP | Specific | 210 | 845 | 4% |
| Comma | CP | Prefix | 1358 | 5020 | 1% |
| DFM | CP | Unconditional | 194 | 629 | 2% |
| DFM | CP | Minimal Cue | 647 | 2351 | 2% |
| DFM | CP | Generic | 59 | 255 | 10% |
| DFM | CP | Specific | 139 | 537 | 8% |
| DFM | CP | Prefix | 1139 | 4339 | 4% |
| DFM | DW | Unconditional | 1100 | 2624 | 2% |
| DFM | DW | Minimal Cue | 325 | 815 | 5% |
| DFM | DW | Generic | 37 | 93 | 30% |
| DFM | DW | Specific | 282 | 648 | 6% |
| DFM | DW | Prefix | 1008 | 1825 | 5% |
| Olmo 3 32B | D3 | Unconditional | 4136 | 10455 | 0% |
| Olmo 3 32B | D3 | Minimal Cue | 863 | 2891 | 4% |
| Olmo 3 32B | D3 | Generic | 77 | 248 | 17% |
| Olmo 3 32B | D3 | Specific | 170 | 728 | 6% |
| Olmo 3 32B | D3 | Prefix | 901 | 2827 | 7% |

### 5.5 Repeated sampling: at least one of 10 generations

Share of prompts with a full match of ≥ 20 tokens / a ≥ 50-token span in at least one of their 10 generations (an (n = 10)-style rate, Hayes et al., 2025), and in how many of the 10 on average when it happens. Repetition loops do not count (5.2).

| Model | Data | Prompt | Full match (≥ 20 tokens) in ≥ 1 of 10 | Mean of 10 | ≥ 50-token span in ≥ 1 of 10 | Mean of 10 |
|---|---|---|---|---|---|---|
| Comma | CP | Generic | 0.1% [0.0, 0.3] | 1.0 | 7.2% [5.7, 8.8] | 1.3 |
| Comma | CP | Specific | 0.1% [0.0, 0.3] | 1.0 | 16.2% [13.9, 18.5] | 1.3 |
| Comma | CP | Prefix | 0.9% [0.4, 1.5] | 3.7 | 35.9% [32.9, 38.9] | 3.8 |
| DFM | CP | Generic | 0.2% [0.0, 0.5] | 1.0 | 5.5% [4.1, 7.0] | 1.1 |
| DFM | CP | Specific | 0.2% [0.0, 0.5] | 1.0 | 11.1% [9.2, 13.1] | 1.3 |
| DFM | CP | Prefix | 0.5% [0.1, 1.0] | 2.8 | 31.4% [28.5, 34.3] | 3.6 |
| DFM | DW | Generic | 0.0% [0.0, 0.0] | 0.0 | 3.3% [2.2, 4.5] | 1.1 |
| DFM | DW | Specific | 0.1% [0.0, 0.3] | 1.0 | 15.5% [13.3, 17.8] | 1.8 |
| DFM | DW | Prefix | 2.7% [1.7, 3.8] | 3.1 | 21.1% [18.6, 23.7] | 4.8 |
| Olmo 3 32B | D3 | Generic | 0.1% [0.0, 0.3] | 1.0 | 5.9% [4.5, 7.4] | 1.3 |
| Olmo 3 32B | D3 | Specific | 0.1% [0.0, 0.3] | 3.0 | 10.7% [8.8, 12.7] | 1.6 |
| Olmo 3 32B | D3 | Prefix | 0.4% [0.1, 0.8] | 4.8 | 22.2% [19.6, 24.8] | 4.1 |

### 5.6 Discoverable extraction of the prefix documents

Whether a prefix generation reproduces the true continuation of its own source document: the next 50 Llama-2 tokens, compared after whitespace normalization. This is the standard targeted definition (Carlini et al., 2023; Hayes et al., 2025); the other tables count matches with any training document.

| Model | Data | Generations | Per generation [95% CI] | In ≥ 1 of 10 [95% CI] | Mean of 10 |
|---|---|---|---|---|---|
| Comma | CP | 9988 | 2.15% [1.35, 3.01] | 3.1% [2.0, 4.2] | 6.9 |
| DFM | CP | 9982 | 1.93% [1.17, 2.76] | 2.9% [1.9, 4.0] | 6.7 |
| DFM | DW | 9996 | 1.04% [0.55, 1.62] | 2.5% [1.6, 3.5] | 4.2 |
| Olmo 3 32B | D3 | 9988 | 1.28% [0.67, 1.98] | 1.8% [1.0, 2.7] | 7.1 |

### 5.7 Duplication of the long matches

Exact occurrence counts in the training index (infini-gram, Llama-2 tokens) of the ≥ 50-token spans, measuring the duplication that 5.3 and 5.4 could not. For each generation's longest span: the median number of occurrences of its first 50 tokens and their distribution, and the share whose whole span occurs once. Counts are occurrences, not documents. Spans that start or end mid-word are trimmed by up to 5 tokens at each end to match the index tokenization; spans still not found (2 generations) are left out. In Dolma 3 the copies of a span are often adjacent near-duplicate documents of the same source file (checked on a sample of two-copy spans), so a single copy is rare there even for prefix; compare Olmo 3 with the < 10 threshold.

| Model | Data | Prompt | Generations with R50 | Median occurrences (50 tokens) | 1 | 2–9 | 10–99 | ≥ 100 | Whole span occurs once |
|---|---|---|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 1767 | 25 | 18% | 24% | 18% | 40% | 50% |
| Comma | CP | Minimal Cue | 1087 | 28 | 20% | 20% | 18% | 42% | 42% |
| Comma | CP | Generic | 94 | 10 | 14% | 35% | 17% | 34% | 31% |
| Comma | CP | Specific | 210 | 14 | 18% | 25% | 28% | 29% | 39% |
| Comma | CP | Prefix | 1358 | 25 | 19% | 24% | 17% | 40% | 44% |
| DFM | CP | Unconditional | 194 | 66 | 14% | 23% | 16% | 47% | 54% |
| DFM | CP | Minimal Cue | 647 | 149 | 15% | 18% | 14% | 53% | 45% |
| DFM | CP | Generic | 59 | 11 | 19% | 29% | 24% | 29% | 41% |
| DFM | CP | Specific | 139 | 12 | 20% | 26% | 25% | 29% | 37% |
| DFM | CP | Prefix | 1139 | 36 | 16% | 24% | 18% | 43% | 42% |
| DFM | DW | Unconditional | 1100 | 4 | 20% | 51% | 23% | 6% | 52% |
| DFM | DW | Minimal Cue | 325 | 4 | 19% | 53% | 21% | 7% | 44% |
| DFM | DW | Generic | 37 | 5 | 24% | 41% | 19% | 16% | 57% |
| DFM | DW | Specific | 282 | 5 | 21% | 44% | 26% | 9% | 53% |
| DFM | DW | Prefix | 1008 | 5 | 22% | 48% | 22% | 8% | 50% |
| Olmo 3 32B | D3 | Unconditional | 4136 | 18 | 0% | 39% | 30% | 31% | 1% |
| Olmo 3 32B | D3 | Minimal Cue | 862 | 26 | 3% | 31% | 31% | 35% | 7% |
| Olmo 3 32B | D3 | Generic | 77 | 21 | 6% | 34% | 39% | 21% | 10% |
| Olmo 3 32B | D3 | Specific | 170 | 36 | 6% | 25% | 29% | 40% | 17% |
| Olmo 3 32B | D3 | Prefix | 901 | 71 | 1% | 28% | 24% | 46% | 5% |

R50 restricted to rare text: generations with a ≥ 50-token span whose first 50 tokens occur fewer than 10 times / exactly once in the training data, and the propensity scores on these generations.

| Model | Data | Prompt | R50, < 10 copies [95% CI] | R50, 1 copy [95% CI] |
|---|---|---|---|---|
| Comma | CP | Unconditional | 12.0% [11.4, 12.6] | 6.6% [6.1, 7.1] |
| Comma | CP | Minimal Cue | 6.3% [5.8, 6.8] | 3.6% [3.2, 4.0] |
| Comma | CP | Generic | 0.6% [0.4, 0.8] | 0.3% [0.1, 0.5] |
| Comma | CP | Specific | 1.4% [1.1, 1.7] | 0.8% [0.6, 1.0] |
| Comma | CP | Prefix | 8.7% [7.6, 9.9] | 5.1% [4.3, 5.9] |
| DFM | CP | Unconditional | 1.2% [1.0, 1.4] | 0.7% [0.5, 0.8] |
| DFM | CP | Minimal Cue | 3.4% [3.1, 3.8] | 2.1% [1.8, 2.3] |
| DFM | CP | Generic | 0.4% [0.2, 0.5] | 0.2% [0.1, 0.3] |
| DFM | CP | Specific | 0.9% [0.7, 1.1] | 0.5% [0.3, 0.6] |
| DFM | CP | Prefix | 7.1% [6.1, 8.1] | 3.9% [3.2, 4.6] |
| DFM | DW | Unconditional | 9.7% [9.1, 10.3] | 4.6% [4.2, 5.0] |
| DFM | DW | Minimal Cue | 2.9% [2.6, 3.3] | 1.2% [1.0, 1.4] |
| DFM | DW | Generic | 0.3% [0.2, 0.4] | 0.1% [0.1, 0.2] |
| DFM | DW | Specific | 2.4% [1.9, 2.8] | 1.1% [0.8, 1.4] |
| DFM | DW | Prefix | 8.6% [7.3, 10.0] | 3.8% [3.1, 4.5] |
| Olmo 3 32B | D3 | Unconditional | 30.0% [29.1, 30.9] | 0.4% [0.3, 0.5] |
| Olmo 3 32B | D3 | Minimal Cue | 4.0% [3.7, 4.4] | 0.5% [0.4, 0.6] |
| Olmo 3 32B | D3 | Generic | 0.4% [0.3, 0.6] | 0.1% [0.0, 0.1] |
| Olmo 3 32B | D3 | Specific | 0.9% [0.6, 1.2] | 0.3% [0.2, 0.4] |
| Olmo 3 32B | D3 | Prefix | 4.4% [3.6, 5.3] | 0.4% [0.2, 0.7] |

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_R50 (< 10 copies) | 0.580 [0.546, 0.615] | 0.420 [0.384, 0.459] | 0.062 [0.041, 0.087] | 0.138 [0.111, 0.167] |
| Comma | CP | PM_R50 (1 copy) | 0.567 [0.525, 0.610] | 0.417 [0.373, 0.463] | 0.049 [0.024, 0.083] | 0.132 [0.102, 0.166] |
| DFM | CP | PM_R50 (< 10 copies) | 0.146 [0.119, 0.176] | 0.326 [0.288, 0.367] | 0.048 [0.032, 0.067] | 0.116 [0.089, 0.146] |
| DFM | CP | PM_R50 (1 copy) | 0.145 [0.110, 0.184] | 0.345 [0.297, 0.397] | 0.042 [0.022, 0.065] | 0.106 [0.074, 0.144] |
| DFM | DW | PM_R50 (< 10 copies) | 0.530 [0.489, 0.573] | 0.253 [0.219, 0.292] | 0.033 [0.021, 0.047] | 0.215 [0.174, 0.260] |
| DFM | DW | PM_R50 (1 copy) | 0.547 [0.496, 0.601] | 0.237 [0.193, 0.287] | 0.033 [0.015, 0.055] | 0.224 [0.169, 0.285] |
| Olmo 3 32B | D3 | PM_R50 (< 10 copies) | 0.871 [0.849, 0.893] | 0.477 [0.425, 0.533] | 0.087 [0.057, 0.123] | 0.166 [0.117, 0.222] |
| Olmo 3 32B | D3 | PM_R50 (1 copy) | 0.482 [0.353, 0.641] | 0.532 [0.404, 0.684] | 0.154 [0.061, 0.294] | 0.389 [0.244, 0.560] |

Discoverable extraction (5.6) by duplication of the source: per-generation extraction rate grouped by how often the prefix with its true 50-token continuation (100 tokens) occurs in the training data (number of prompts in parentheses).

| Model | Data | 1 occurrence | 2–9 occurrences | ≥ 10 occurrences |
|---|---|---|---|---|
| Comma | CP | 0.9% (938) | 1.1% (36) | 48.5% (26) |
| DFM | CP | 0.8% (938) | 0.6% (36) | 45.0% (26) |
| DFM | DW | 0.6% (941) | 4.4% (57) | 100.0% (2) |
| Olmo 3 32B | D3 | 0.2% (86) | 0.9% (703) | 2.8% (211) |

## 6. The paper's claims, re-checked on these results

Statuses follow fixed rules: point estimates for orderings and the 0.5 threshold, and the sign of 95% CIs for differences. *Partly* means some but not all of the stated comparisons hold. Status and evidence use the main (filtered) results; the last column gives the status with degenerate outputs included, in bold where it differs.

| Claim (paper section) | Status | Evidence | Unfiltered status |
|---|---|---|---|
| Prefix yields higher NVR and ALS point estimates than generic and specific prompting, for every model–corpus pair (Abstract, §5.2, §6) | holds | 12/12 comparisons; exceptions: none | holds |
| … also for Olmo 3 on Dolma 3 (new) | holds | 4/4; exceptions: none | holds |
| … also against the new prompt-free settings (unconditional, minimal cue) | partly holds | 7/16; exceptions: Comma CP Unconditional NVR, Comma CP Unconditional ALS, Comma CP Minimal Cue NVR, DFM CP Minimal Cue NVR, DFM DW Unconditional NVR, DFM DW Unconditional ALS, Olmo 3 32B D3 Unconditional NVR, Olmo 3 32B D3 Unconditional ALS, Olmo 3 32B D3 Minimal Cue NVR | partly holds |
| Propensity scores are generally below the neutral 0.5 (Abstract, §5.2, §6) — generic and specific | partly holds | 15/16 below 0.5; at/above: DFM CP Generic PM_FMR 0.606 [0.481, 0.747] | partly holds |
| … for the new prompt-free settings | partly holds | 8/16 below 0.5; at/above: Comma CP Unconditional PM_NVR 0.548 [0.490, 0.614]; Comma CP Minimal Cue PM_NVR 0.538 [0.480, 0.602]; DFM CP Minimal Cue PM_NVR 0.523 [0.464, 0.591]; DFM CP Minimal Cue PM_FMR 0.657 [0.538, 0.787]; DFM DW Unconditional PM_NVR 0.508 [0.473, 0.543]; Olmo 3 32B D3 Unconditional PM_NVR 0.744 [0.674, 0.811]; Olmo 3 32B D3 Minimal Cue PM_NVR 0.586 [0.497, 0.679]; Olmo 3 32B D3 Minimal Cue PM_FMR 0.579 [0.396, 0.820] | partly holds |
| Comma produces longer verbatim spans than DFM on Common Pile under generic and prefix prompting (§5.2) | holds | DFM − Comma: ALS Generic -0.50 [-0.94, -0.08] ↓; ALS Prefix -2.22 [-3.06, -1.42] ↓ | **partly holds** |
| Comma shows more full-generation memorization of Common Pile than DFM (specific, prefix) (§5.2) | does not hold | DFM − Comma: FMR Specific +0.0012 [+0.0000, +0.0024] ↑; FMR Prefix -0.0019 [-0.0044, +0.0001] ≈ | does not hold |
| Specific prompts match the prefix FMR for Comma on Common Pile (§5.1) | does not hold | specific − prefix -0.0039 [-0.0074, -0.0011] ↓ | does not hold |
| From Comma to DFM, specific-prompt PM_FMR decreases while PM_NVR increases (§5.2) | does not hold | PM_FMR Δ +0.208 [+0.054, +0.356] ↑; PM_NVR Δ -0.005 [-0.031, +0.020] ≈ | does not hold |
| For DFM, Common Pile yields longer ALS than Dynaword in every setting (§5.2) | partly holds | CP − DW: Unconditional -15.94 [-16.46, -15.42] ↓; Minimal Cue -1.67 [-2.28, -1.05] ↓; Generic -2.15 [-2.49, -1.79] ↓; Specific -2.10 [-2.97, -1.25] ↓; Prefix +4.80 [+2.53, +7.07] ↑ | partly holds |
| For DFM, prefix FMR is higher on Dynaword than on Common Pile (§5.2) | holds | DW − CP +0.0330 [+0.0248, +0.0415] ↑ | holds |
| Memorization is essentially stable across DFM training stages (§5.3) | does not hold | 20/60 stage − final paired differences significant (see Section 4) | does not hold |

Section 5 tests whether the prompt-free exceptions come from code, templated or duplicated text (5.3, 5.7). PM_FMR exceptions rest largely on short full matches: with full matches of ≥ 20 / ≥ 50 tokens only, see PM_FMR20 / PM_FMR50 in 5.2.

## Appendix — Unfiltered results (every generation, degenerate outputs included)

The same tables without the degenerate-output filter, as the SimpleTrace summaries and the paper's metric definitions count them.

### Table A (unfiltered)

| Model | Data | Prompt | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | Excluded |
|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 0.0353 [0.0327, 0.0379] | 0.0019 [0.0011, 0.0028] | 35.85 [35.17, 36.52] | 0 |
| Comma | CP | Minimal Cue | 0.0338 [0.0313, 0.0365] | 0.0035 [0.0024, 0.0047] | 29.12 [28.49, 29.78] | 0 |
| Comma | CP | Generic | 0.0023 [0.0018, 0.0028] | 0.0014 [0.0007, 0.0022] | 13.88 [13.53, 14.27] | 0 |
| Comma | CP | Specific | 0.0034 [0.0027, 0.0042] | 0.0019 [0.0010, 0.0029] | 16.34 [15.88, 16.83] | 0 |
| Comma | CP | Prefix | 0.0290 [0.0226, 0.0362] | 0.0056 [0.0028, 0.0089] | 32.48 [30.62, 34.43] | 0 |
| DFM | CP | Unconditional | 0.0072 [0.0058, 0.0087] | 0.0015 [0.0008, 0.0023] | 10.31 [10.04, 10.61] | 0 |
| DFM | CP | Minimal Cue | 0.0286 [0.0260, 0.0313] | 0.0071 [0.0055, 0.0087] | 18.71 [18.15, 19.29] | 0 |
| DFM | CP | Generic | 0.0061 [0.0052, 0.0071] | 0.0059 [0.0044, 0.0075] | 13.49 [13.19, 13.81] | 0 |
| DFM | CP | Specific | 0.0029 [0.0023, 0.0035] | 0.0067 [0.0048, 0.0088] | 17.57 [16.60, 18.60] | 0 |
| DFM | CP | Prefix | 0.0260 [0.0201, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.33 [28.63, 32.08] | 0 |
| DFM | DW | Unconditional | 0.0475 [0.0450, 0.0500] | 0.0287 [0.0255, 0.0320] | 26.25 [25.83, 26.69] | 0 |
| DFM | DW | Minimal Cue | 0.0273 [0.0253, 0.0294] | 0.0208 [0.0180, 0.0237] | 20.30 [20.02, 20.60] | 0 |
| DFM | DW | Generic | 0.0008 [0.0005, 0.0012] | 0.0019 [0.0010, 0.0030] | 16.08 [15.70, 16.50] | 0 |
| DFM | DW | Specific | 0.0034 [0.0026, 0.0043] | 0.0084 [0.0062, 0.0110] | 19.90 [18.74, 21.14] | 0 |
| DFM | DW | Prefix | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.43 [24.04, 26.88] | 0 |
| Olmo 3 32B | D3 | Unconditional | 0.0249 [0.0235, 0.0263] | 0.0007 [0.0002, 0.0013] | 53.49 [52.69, 54.31] | 0 |
| Olmo 3 32B | D3 | Minimal Cue | 0.0121 [0.0109, 0.0133] | 0.0040 [0.0028, 0.0053] | 26.54 [25.90, 27.19] | 0 |
| Olmo 3 32B | D3 | Generic | 0.0021 [0.0015, 0.0031] | 0.0002 [0.0000, 0.0005] | 15.10 [14.83, 15.41] | 0 |
| Olmo 3 32B | D3 | Specific | 0.0012 [0.0005, 0.0023] | 0.0006 [0.0001, 0.0014] | 17.68 [17.19, 18.23] | 0 |
| Olmo 3 32B | D3 | Prefix | 0.0085 [0.0058, 0.0120] | 0.0029 [0.0009, 0.0058] | 26.88 [25.46, 28.38] | 0 |

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3. Every generation counted (Excluded is 0). FMR counts full matches of any length, as in the paper; under generic and specific prompts most of them are short outputs (often under 10 tokens) that occur verbatim in the training data. FMR20 / FMR50, counting only full matches of ≥ 20 / ≥ 50 tokens, are in Section 5.2.

### Table B (unfiltered)

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_NVR | 0.548 [0.490, 0.614] | 0.538 [0.480, 0.603] | 0.074 [0.054, 0.100] | 0.105 [0.078, 0.141] |
| Comma | CP | PM_FMR | 0.253 [0.141, 0.426] | 0.385 [0.256, 0.561] | 0.200 [0.101, 0.353] | 0.253 [0.136, 0.424] |
| DFM | CP | PM_NVR | 0.218 [0.170, 0.277] | 0.523 [0.464, 0.591] | 0.191 [0.151, 0.241] | 0.101 [0.075, 0.134] |
| DFM | CP | PM_FMR | 0.288 [0.158, 0.460] | 0.657 [0.538, 0.788] | 0.615 [0.491, 0.755] | 0.644 [0.512, 0.779] |
| DFM | DW | PM_NVR | 0.508 [0.473, 0.544] | 0.372 [0.338, 0.410] | 0.017 [0.010, 0.026] | 0.069 [0.053, 0.087] |
| DFM | DW | PM_FMR | 0.439 [0.382, 0.505] | 0.362 [0.306, 0.428] | 0.049 [0.025, 0.081] | 0.186 [0.137, 0.247] |
| Olmo 3 32B | D3 | PM_NVR | 0.745 [0.675, 0.812] | 0.586 [0.497, 0.679] | 0.201 [0.130, 0.298] | 0.124 [0.054, 0.229] |
| Olmo 3 32B | D3 | PM_FMR | 0.194 [0.065, 0.467] | 0.580 [0.397, 0.820] | 0.065 [0.000, 0.235] | 0.171 [0.023, 0.455] |

PM = f_setting / (f_setting + f_prefix); 0.5 is neutral, lower values mean lower propensity relative to capability. Setting and prefix sets are resampled independently. PM_FMR inherits FMR's short full matches (see Table A); PM_FMR20 / PM_FMR50 and the long-span scores PM_R50 / PM_NVR50 are in Section 5.2.


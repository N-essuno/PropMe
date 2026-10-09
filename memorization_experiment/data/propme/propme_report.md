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

Both conditions are needed: repetition alone also flags generations that repeat a natural-language sentence (common under generic prompts; their longest training span is ~8–10 tokens, so they do not inflate any metric), and a low letter share alone also flags code, tables and logs, which are kept because their verbatim reproduction is memorization (their share is examined separately in Section 5.3).

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

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3. Excluded: degenerate generations left out.

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

PM = f_setting / (f_setting + f_prefix); 0.5 is neutral, lower values mean lower propensity relative to capability. Setting and prefix sets are resampled independently.

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

R20 / R50: share of generations whose longest training span has ≥ 20 / ≥ 50 tokens. NVR50: NVR counting only documents retrieved through ≥ 50-token spans (over all retrieved documents). At 50 tokens the false-positive floor of matched non-member controls is ~0 for natural text (Cooper et al., 2026).

| Model | Data | Prompt | R20 [95% CI] | R50 [95% CI] | NVR50 [95% CI] |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 67.3% [66.4, 68.3] | 17.8% [17.0, 18.5] | 0.0116 [0.0106, 0.0126] |
| Comma | CP | Minimal Cue | 44.3% [43.4, 45.3] | 11.0% [10.4, 11.6] | 0.0110 [0.0100, 0.0121] |
| Comma | CP | Generic | 8.4% [7.6, 9.1] | 1.3% [1.0, 1.6] | 0.0006 [0.0004, 0.0009] |
| Comma | CP | Specific | 15.7% [14.6, 16.8] | 2.3% [1.9, 2.6] | 0.0011 [0.0008, 0.0014] |
| Comma | CP | Prefix | 53.8% [51.4, 56.2] | 14.1% [12.5, 15.8] | 0.0099 [0.0074, 0.0125] |
| DFM | CP | Unconditional | 6.8% [6.3, 7.3] | 2.0% [1.7, 2.3] | 0.0023 [0.0018, 0.0029] |
| DFM | CP | Minimal Cue | 19.5% [18.7, 20.3] | 6.6% [6.1, 7.0] | 0.0086 [0.0076, 0.0096] |
| DFM | CP | Generic | 9.9% [9.1, 10.7] | 0.8% [0.7, 1.0] | 0.0005 [0.0003, 0.0007] |
| DFM | CP | Specific | 12.9% [11.9, 14.0] | 1.6% [1.4, 1.9] | 0.0007 [0.0005, 0.0010] |
| DFM | CP | Prefix | 51.4% [49.0, 53.7] | 11.9% [10.4, 13.4] | 0.0088 [0.0064, 0.0114] |
| DFM | DW | Unconditional | 49.5% [48.5, 50.5] | 11.0% [10.4, 11.6] | 0.0093 [0.0084, 0.0101] |
| DFM | DW | Minimal Cue | 34.3% [33.4, 35.2] | 3.3% [2.9, 3.6] | 0.0039 [0.0033, 0.0047] |
| DFM | DW | Generic | 16.8% [15.8, 17.8] | 0.5% [0.3, 0.6] | 0.0001 [0.0000, 0.0002] |
| DFM | DW | Specific | 22.0% [20.4, 23.5] | 3.0% [2.5, 3.6] | 0.0012 [0.0009, 0.0017] |
| DFM | DW | Prefix | 41.6% [39.1, 44.1] | 10.1% [8.6, 11.7] | 0.0108 [0.0082, 0.0136] |
| Olmo 3 32B | D3 | Unconditional | 84.4% [83.7, 85.1] | 41.7% [40.8, 42.7] | 0.0111 [0.0104, 0.0118] |
| Olmo 3 32B | D3 | Minimal Cue | 39.5% [38.5, 40.4] | 8.7% [8.1, 9.2] | 0.0048 [0.0042, 0.0053] |
| Olmo 3 32B | D3 | Generic | 9.3% [8.5, 10.0] | 0.9% [0.7, 1.2] | 0.0005 [0.0002, 0.0010] |
| Olmo 3 32B | D3 | Specific | 20.5% [19.3, 21.8] | 1.7% [1.3, 2.2] | 0.0004 [0.0002, 0.0007] |
| Olmo 3 32B | D3 | Prefix | 46.0% [43.8, 48.3] | 9.1% [7.7, 10.6] | 0.0036 [0.0024, 0.0050] |

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_R50 | 0.558 [0.528, 0.589] | 0.439 [0.408, 0.472] | 0.084 [0.066, 0.105] | 0.138 [0.115, 0.162] |
| Comma | CP | PM_NVR50 | 0.540 [0.477, 0.613] | 0.528 [0.463, 0.600] | 0.060 [0.037, 0.091] | 0.101 [0.071, 0.140] |
| DFM | CP | PM_R50 | 0.143 [0.121, 0.168] | 0.356 [0.323, 0.392] | 0.065 [0.050, 0.081] | 0.122 [0.099, 0.147] |
| DFM | CP | PM_NVR50 | 0.209 [0.155, 0.280] | 0.493 [0.421, 0.579] | 0.056 [0.034, 0.087] | 0.077 [0.052, 0.112] |
| DFM | DW | PM_R50 | 0.522 [0.483, 0.565] | 0.245 [0.213, 0.283] | 0.043 [0.029, 0.059] | 0.231 [0.191, 0.276] |
| DFM | DW | PM_NVR50 | 0.462 [0.401, 0.532] | 0.268 [0.214, 0.335] | 0.011 [0.002, 0.024] | 0.103 [0.069, 0.149] |
| Olmo 3 32B | D3 | PM_R50 | 0.821 [0.797, 0.845] | 0.488 [0.446, 0.534] | 0.094 [0.068, 0.126] | 0.159 [0.124, 0.199] |
| Olmo 3 32B | D3 | PM_NVR50 | 0.757 [0.686, 0.824] | 0.572 [0.482, 0.670] | 0.119 [0.046, 0.238] | 0.107 [0.053, 0.180] |

### 5.3 Natural-language generations only (code and symbol-heavy text removed)

A generation is code-like if ≥ 3 lines (and ≥ 30% of its lines) look like code — imports, declarations, braces, semicolons, comments — or if under 50% of its characters are letters (number sequences, tables). Metrics below are recomputed on the remaining generations.

Limit of the heuristic: random samples of the remaining ≥ 50-token spans in the prompt-free settings still contain much templated text — open-access license and citation boilerplate, wiki page templates, methods-section boilerplate and task-instruction templates for Comma; coding-exercise statements, software licenses and SQL dumps for Olmo 3. How often these spans occur in the training data is measured in 5.7.

| Model | Data | Prompt | Code-like | NVR [95% CI] | ALS [95% CI] | R50 [95% CI] |
|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 29.2% | 0.0290 [0.0262, 0.0319] | 29.82 [29.20, 30.44] | 10.8% [10.0, 11.5] |
| Comma | CP | Minimal Cue | 8.9% | 0.0366 [0.0339, 0.0395] | 28.87 [28.21, 29.56] | 10.9% [10.3, 11.6] |
| Comma | CP | Generic | 4.9% | 0.0023 [0.0018, 0.0028] | 13.54 [13.21, 13.93] | 1.2% [0.9, 1.5] |
| Comma | CP | Specific | 5.1% | 0.0033 [0.0025, 0.0041] | 15.29 [14.93, 15.67] | 1.7% [1.4, 2.1] |
| Comma | CP | Prefix | 47.1% | 0.0203 [0.0140, 0.0278] | 24.70 [22.79, 26.89] | 8.4% [6.8, 10.2] |
| DFM | CP | Unconditional | 4.0% | 0.0053 [0.0040, 0.0067] | 8.97 [8.77, 9.19] | 1.0% [0.8, 1.2] |
| DFM | CP | Minimal Cue | 8.2% | 0.0316 [0.0288, 0.0345] | 18.06 [17.49, 18.65] | 6.6% [6.1, 7.1] |
| DFM | CP | Generic | 5.6% | 0.0063 [0.0054, 0.0074] | 13.06 [12.79, 13.35] | 0.7% [0.6, 0.9] |
| DFM | CP | Specific | 5.4% | 0.0027 [0.0022, 0.0034] | 14.06 [13.70, 14.44] | 1.3% [1.0, 1.5] |
| DFM | CP | Prefix | 47.0% | 0.0169 [0.0109, 0.0239] | 21.80 [20.24, 23.60] | 5.8% [4.4, 7.3] |
| DFM | DW | Unconditional | 4.0% | 0.0512 [0.0486, 0.0540] | 27.00 [26.56, 27.43] | 11.5% [10.8, 12.1] |
| DFM | DW | Minimal Cue | 0.7% | 0.0276 [0.0256, 0.0297] | 20.25 [19.97, 20.52] | 3.3% [2.9, 3.6] |
| DFM | DW | Generic | 0.5% | 0.0008 [0.0005, 0.0012] | 15.51 [15.32, 15.70] | 0.4% [0.3, 0.6] |
| DFM | DW | Specific | 1.4% | 0.0033 [0.0025, 0.0042] | 16.67 [16.08, 17.29] | 2.8% [2.3, 3.3] |
| DFM | DW | Prefix | 2.0% | 0.0464 [0.0405, 0.0525] | 25.56 [24.16, 27.01] | 10.2% [8.6, 11.8] |
| Olmo 3 32B | D3 | Unconditional | 82.3% | 0.0207 [0.0164, 0.0255] | 26.34 [24.47, 28.30] | 8.8% [7.5, 10.2] |
| Olmo 3 32B | D3 | Minimal Cue | 10.9% | 0.0116 [0.0104, 0.0129] | 25.44 [24.78, 26.11] | 7.7% [7.1, 8.3] |
| Olmo 3 32B | D3 | Generic | 0.6% | 0.0021 [0.0014, 0.0031] | 15.02 [14.77, 15.31] | 0.9% [0.7, 1.2] |
| Olmo 3 32B | D3 | Specific | 0.8% | 0.0011 [0.0005, 0.0022] | 17.38 [16.97, 17.85] | 1.6% [1.2, 2.0] |
| Olmo 3 32B | D3 | Prefix | 16.0% | 0.0075 [0.0044, 0.0115] | 24.16 [22.85, 25.54] | 6.4% [5.1, 7.7] |

| Model | Data | Metric (natural language) | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_NVR | 0.588 [0.506, 0.678] | 0.643 [0.566, 0.725] | 0.100 [0.070, 0.145] | 0.139 [0.097, 0.200] |
| Comma | CP | PM_R50 | 0.560 [0.509, 0.616] | 0.564 [0.514, 0.620] | 0.122 [0.092, 0.159] | 0.172 [0.136, 0.214] |
| DFM | CP | PM_NVR | 0.239 [0.166, 0.341] | 0.652 [0.565, 0.745] | 0.273 [0.203, 0.373] | 0.140 [0.096, 0.210] |
| DFM | CP | PM_R50 | 0.148 [0.112, 0.195] | 0.532 [0.470, 0.603] | 0.114 [0.083, 0.155] | 0.180 [0.137, 0.236] |
| DFM | DW | PM_NVR | 0.524 [0.491, 0.560] | 0.373 [0.339, 0.411] | 0.017 [0.010, 0.026] | 0.066 [0.050, 0.085] |
| DFM | DW | PM_R50 | 0.530 [0.491, 0.573] | 0.243 [0.212, 0.282] | 0.041 [0.028, 0.057] | 0.214 [0.175, 0.258] |
| Olmo 3 32B | D3 | PM_NVR | 0.733 [0.627, 0.827] | 0.606 [0.498, 0.725] | 0.220 [0.134, 0.350] | 0.130 [0.051, 0.261] |
| Olmo 3 32B | D3 | PM_R50 | 0.581 [0.519, 0.646] | 0.547 [0.495, 0.606] | 0.125 [0.089, 0.170] | 0.203 [0.157, 0.260] |

### 5.4 Concentration of long matches

For generations with a ≥ 50-token span: distinct training documents retrieved for that span (≤ 10 per span), and the share of these generations whose span retrieved one of the 10 most frequently hit documents. A high share means a few documents account for the matches. A low share does not rule out duplication: text present in many documents (e.g. code boilerplate) retrieves a different sample of them each time, so it is measured with exact occurrence counts in 5.7.

| Model | Data | Prompt | Generations with R50 | Distinct documents | Top-10 documents' share |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 1777 | 5865 | 1% |
| Comma | CP | Minimal Cue | 1103 | 4016 | 1% |
| Comma | CP | Generic | 130 | 469 | 20% |
| Comma | CP | Specific | 225 | 874 | 4% |
| Comma | CP | Prefix | 1410 | 5194 | 1% |
| DFM | CP | Unconditional | 198 | 646 | 2% |
| DFM | CP | Minimal Cue | 655 | 2365 | 2% |
| DFM | CP | Generic | 82 | 282 | 16% |
| DFM | CP | Specific | 163 | 554 | 15% |
| DFM | CP | Prefix | 1184 | 4456 | 5% |
| DFM | DW | Unconditional | 1102 | 2631 | 2% |
| DFM | DW | Minimal Cue | 328 | 822 | 5% |
| DFM | DW | Generic | 45 | 98 | 36% |
| DFM | DW | Specific | 301 | 675 | 7% |
| DFM | DW | Prefix | 1010 | 1829 | 5% |
| Olmo 3 32B | D3 | Unconditional | 4170 | 10541 | 0% |
| Olmo 3 32B | D3 | Minimal Cue | 868 | 2906 | 4% |
| Olmo 3 32B | D3 | Generic | 94 | 278 | 14% |
| Olmo 3 32B | D3 | Specific | 172 | 730 | 6% |
| Olmo 3 32B | D3 | Prefix | 909 | 2839 | 7% |

### 5.5 Repeated sampling: at least one of 10 generations

Share of prompts with a full match / a ≥ 50-token span in at least one of their 10 generations (an (n = 10)-style rate, Hayes et al., 2025), and in how many of the 10 on average when it happens.

| Model | Data | Prompt | Full match in ≥ 1 of 10 | Mean of 10 | ≥ 50-token span in ≥ 1 of 10 | Mean of 10 |
|---|---|---|---|---|---|---|
| Comma | CP | Generic | 1.4% [0.7, 2.2] | 1.0 | 10.4% [8.6, 12.4] | 1.2 |
| Comma | CP | Specific | 1.5% [0.8, 2.3] | 1.1 | 17.3% [15.0, 19.6] | 1.3 |
| Comma | CP | Prefix | 2.7% [1.7, 3.7] | 2.1 | 37.7% [34.7, 40.7] | 3.7 |
| DFM | CP | Generic | 5.3% [3.9, 6.7] | 1.1 | 7.8% [6.2, 9.5] | 1.1 |
| DFM | CP | Specific | 2.7% [1.7, 3.7] | 1.1 | 13.4% [11.3, 15.6] | 1.2 |
| DFM | CP | Prefix | 2.1% [1.2, 3.0] | 1.8 | 33.4% [30.4, 36.4] | 3.5 |
| DFM | DW | Generic | 0.9% [0.4, 1.5] | 1.2 | 4.0% [2.9, 5.3] | 1.1 |
| DFM | DW | Specific | 3.4% [2.3, 4.5] | 1.5 | 16.8% [14.5, 19.1] | 1.8 |
| DFM | DW | Prefix | 12.1% [10.1, 14.2] | 3.0 | 21.2% [18.7, 23.8] | 4.8 |
| Olmo 3 32B | D3 | Generic | 0.2% [0.0, 0.5] | 1.0 | 7.4% [5.8, 9.1] | 1.3 |
| Olmo 3 32B | D3 | Specific | 0.4% [0.1, 0.8] | 1.5 | 10.9% [9.0, 12.9] | 1.6 |
| Olmo 3 32B | D3 | Prefix | 1.1% [0.5, 1.8] | 2.6 | 22.7% [20.1, 25.4] | 4.0 |

### 5.6 Discoverable extraction of the prefix documents

Whether a prefix generation reproduces the true continuation of its own source document: the next 50 Llama-2 tokens, compared after whitespace normalization. This is the standard targeted definition (Carlini et al., 2023; Hayes et al., 2025); the other tables count matches with any training document.

| Model | Data | Generations | Per generation [95% CI] | In ≥ 1 of 10 [95% CI] | Mean of 10 |
|---|---|---|---|---|---|
| Comma | CP | 9988 | 2.15% [1.36, 3.02] | 3.1% [2.0, 4.2] | 6.9 |
| DFM | CP | 9982 | 1.93% [1.17, 2.77] | 2.9% [1.9, 4.0] | 6.7 |
| DFM | DW | 9996 | 1.04% [0.56, 1.60] | 2.5% [1.6, 3.5] | 4.2 |
| Olmo 3 32B | D3 | 9988 | 1.28% [0.66, 1.99] | 1.8% [1.0, 2.7] | 7.1 |

### 5.7 Duplication of the long matches

Exact occurrence counts in the training index (infini-gram, Llama-2 tokens) of the ≥ 50-token spans, measuring the duplication that 5.3 and 5.4 could not. For each generation's longest span: the median number of occurrences of its first 50 tokens and their distribution, and the share whose whole span occurs once. Counts are occurrences, not documents. Spans that start or end mid-word are trimmed by up to 5 tokens at each end to match the index tokenization; spans still not found (2 generations) are left out. In Dolma 3 the copies of a span are often adjacent near-duplicate documents of the same source file (checked on a sample of two-copy spans), so a single copy is rare there even for prefix; compare Olmo 3 with the < 10 threshold.

| Model | Data | Prompt | Generations with R50 | Median occurrences (50 tokens) | 1 | 2–9 | 10–99 | ≥ 100 | Whole span occurs once |
|---|---|---|---|---|---|---|---|---|---|
| Comma | CP | Unconditional | 1777 | 26 | 18% | 24% | 18% | 40% | 50% |
| Comma | CP | Minimal Cue | 1103 | 27 | 19% | 20% | 18% | 42% | 42% |
| Comma | CP | Generic | 130 | 20 | 12% | 28% | 23% | 36% | 39% |
| Comma | CP | Specific | 225 | 16 | 17% | 24% | 28% | 31% | 40% |
| Comma | CP | Prefix | 1410 | 28 | 19% | 23% | 17% | 40% | 43% |
| DFM | CP | Unconditional | 198 | 71 | 14% | 22% | 16% | 48% | 53% |
| DFM | CP | Minimal Cue | 655 | 144 | 15% | 18% | 14% | 53% | 45% |
| DFM | CP | Generic | 82 | 13 | 15% | 26% | 32% | 28% | 46% |
| DFM | CP | Specific | 163 | 18 | 17% | 25% | 22% | 36% | 38% |
| DFM | CP | Prefix | 1184 | 45 | 15% | 23% | 17% | 44% | 41% |
| DFM | DW | Unconditional | 1102 | 4 | 20% | 50% | 23% | 6% | 52% |
| DFM | DW | Minimal Cue | 328 | 4 | 19% | 53% | 21% | 7% | 44% |
| DFM | DW | Generic | 45 | 4 | 24% | 44% | 16% | 16% | 62% |
| DFM | DW | Specific | 301 | 5 | 20% | 42% | 26% | 13% | 52% |
| DFM | DW | Prefix | 1010 | 5 | 22% | 48% | 22% | 8% | 50% |
| Olmo 3 32B | D3 | Unconditional | 4170 | 18 | 0% | 39% | 30% | 31% | 1% |
| Olmo 3 32B | D3 | Minimal Cue | 867 | 26 | 3% | 31% | 30% | 35% | 7% |
| Olmo 3 32B | D3 | Generic | 94 | 21 | 5% | 33% | 38% | 23% | 13% |
| Olmo 3 32B | D3 | Specific | 172 | 37 | 6% | 25% | 28% | 41% | 17% |
| Olmo 3 32B | D3 | Prefix | 909 | 70 | 2% | 28% | 25% | 46% | 5% |

R50 restricted to rare text: generations with a ≥ 50-token span whose first 50 tokens occur fewer than 10 times / exactly once in the training data, and the propensity scores on these generations.

| Model | Data | Prompt | R50, < 10 copies [95% CI] | R50, 1 copy [95% CI] |
|---|---|---|---|---|
| Comma | CP | Unconditional | 12.0% [11.4, 12.7] | 6.6% [6.1, 7.1] |
| Comma | CP | Minimal Cue | 6.4% [5.9, 6.9] | 3.7% [3.3, 4.0] |
| Comma | CP | Generic | 0.7% [0.5, 1.0] | 0.4% [0.2, 0.5] |
| Comma | CP | Specific | 1.4% [1.2, 1.7] | 0.8% [0.6, 1.0] |
| Comma | CP | Prefix | 9.0% [7.8, 10.1] | 5.1% [4.4, 6.0] |
| DFM | CP | Unconditional | 1.2% [1.0, 1.4] | 0.7% [0.5, 0.8] |
| DFM | CP | Minimal Cue | 3.5% [3.1, 3.8] | 2.1% [1.8, 2.4] |
| DFM | CP | Generic | 0.4% [0.3, 0.6] | 0.2% [0.1, 0.3] |
| DFM | CP | Specific | 1.1% [0.9, 1.3] | 0.5% [0.3, 0.6] |
| DFM | CP | Prefix | 7.3% [6.3, 8.3] | 4.0% [3.3, 4.7] |
| DFM | DW | Unconditional | 9.7% [9.1, 10.3] | 4.6% [4.2, 5.0] |
| DFM | DW | Minimal Cue | 3.0% [2.6, 3.3] | 1.2% [1.0, 1.4] |
| DFM | DW | Generic | 0.4% [0.2, 0.5] | 0.2% [0.1, 0.2] |
| DFM | DW | Specific | 2.4% [2.0, 2.9] | 1.1% [0.8, 1.4] |
| DFM | DW | Prefix | 8.6% [7.3, 10.0] | 3.8% [3.1, 4.6] |
| Olmo 3 32B | D3 | Unconditional | 30.2% [29.3, 31.1] | 0.4% [0.3, 0.6] |
| Olmo 3 32B | D3 | Minimal Cue | 4.1% [3.7, 4.5] | 0.5% [0.4, 0.7] |
| Olmo 3 32B | D3 | Generic | 0.5% [0.3, 0.7] | 0.1% [0.0, 0.1] |
| Olmo 3 32B | D3 | Specific | 0.9% [0.6, 1.2] | 0.3% [0.2, 0.4] |
| Olmo 3 32B | D3 | Prefix | 4.5% [3.6, 5.4] | 0.5% [0.3, 0.7] |

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_R50 (< 10 copies) | 0.573 [0.540, 0.609] | 0.416 [0.382, 0.454] | 0.072 [0.051, 0.099] | 0.138 [0.112, 0.167] |
| Comma | CP | PM_R50 (1 copy) | 0.563 [0.522, 0.606] | 0.415 [0.372, 0.462] | 0.064 [0.037, 0.098] | 0.132 [0.101, 0.166] |
| DFM | CP | PM_R50 (< 10 copies) | 0.143 [0.117, 0.173] | 0.323 [0.287, 0.363] | 0.056 [0.039, 0.075] | 0.130 [0.102, 0.161] |
| DFM | CP | PM_R50 (1 copy) | 0.144 [0.110, 0.183] | 0.343 [0.297, 0.396] | 0.043 [0.024, 0.066] | 0.108 [0.076, 0.145] |
| DFM | DW | PM_R50 (< 10 copies) | 0.529 [0.490, 0.574] | 0.255 [0.221, 0.295] | 0.040 [0.027, 0.056] | 0.220 [0.179, 0.267] |
| DFM | DW | PM_R50 (1 copy) | 0.546 [0.496, 0.600] | 0.236 [0.192, 0.288] | 0.038 [0.018, 0.062] | 0.227 [0.172, 0.289] |
| Olmo 3 32B | D3 | PM_R50 (< 10 copies) | 0.871 [0.849, 0.893] | 0.478 [0.426, 0.534] | 0.099 [0.067, 0.138] | 0.164 [0.117, 0.219] |
| Olmo 3 32B | D3 | PM_R50 (1 copy) | 0.477 [0.349, 0.630] | 0.525 [0.400, 0.673] | 0.148 [0.058, 0.278] | 0.378 [0.238, 0.544] |

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

Section 5 tests whether the prompt-free exceptions come from code, templated or duplicated text (5.3, 5.7).

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

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3. Every generation counted (Excluded is 0).

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

PM = f_setting / (f_setting + f_prefix); 0.5 is neutral, lower values mean lower propensity relative to capability. Setting and prefix sets are resampled independently.


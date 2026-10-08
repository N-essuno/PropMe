# PropMe results report

Generated 2026-10-08 by `05_propensity_metrics/make_propme_report.py` from the SimpleTrace traces in `memorization_experiment/data/propme/`.

## Setup and changes since the paper

| | Paper (ARR, 12 Oct) | This report |
|---|---|---|
| Models / corpora | Comma (CP); DFM Decoder stages 1, 2, final (CP, DW) | same, plus **Olmo 3 32B on Dolma 3 (D3)** |
| Decoding | greedy (temperature 0), 1 generation per prompt | sampling, temperature 0.7, top-p 0.95, 256 new tokens, **10 generations per prompt** |
| Prompts per setting | 100 | 1000 prompts × 10 = 10000 generations (prompt-free settings: 1000 generations) |
| Generic | GPT-generated plausible prompts | Tatoeba sentences not verbatim in the index (random 1000-prompt sample) |
| Specific | GPT-generated, dataset-inspired | sentences from unseen sources close to the training data (00_prepare_data/propensity_settings, 1000 per corpus) |
| Prefix (capability) | first 50 tokens of training documents | same construction, random 1000-prompt sample |
| New propensity settings | — | **Unconditional** (start-of-document token only) and **Minimal cue** (start-of-document token + one frequent word drawn per generation) |
| Confidence intervals | Wilson 95% for NVR and FMR | percentile bootstrap 95% (B = 10,000) for NVR, FMR, ALS and PM; resampling prompts (each with its 10 generations) for generic, specific and prefix, generations for unconditional and minimal cue |

Settings are listed from the least to the most targeted prompting: Unconditional, Minimal cue, Generic, Specific (propensity settings) and Prefix (capability setting). A full-match ratio of 0 is shown with its rule-of-three upper bound 3/n (n = independent units). ↑ / ↓ mark differences whose 95% CI lies above / below 0, ≈ those whose CI includes 0. † PM is 0 by convention when both values are 0 (paper, Eq. 1).

## Table A — Memorization metrics across model–corpus pairs (cf. paper Tables 1 and 3)

| Model | Data | Prompt | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 0.0293 [0.0223, 0.0366] | 0 (no match; ≤ 0.0030) | 34.79 [32.85, 36.89] |
| Comma | CP | Minimal Cue | 0.0316 [0.0246, 0.0391] | 0.0030 [0.0000, 0.0070] | 27.77 [25.93, 29.71] |
| Comma | CP | Generic | 0.0023 [0.0018, 0.0028] | 0.0014 [0.0007, 0.0022] | 13.88 [13.53, 14.27] |
| Comma | CP | Specific | 0.0034 [0.0027, 0.0042] | 0.0019 [0.0010, 0.0029] | 16.34 [15.88, 16.83] |
| Comma | CP | Prefix | 0.0290 [0.0226, 0.0362] | 0.0056 [0.0028, 0.0089] | 32.48 [30.62, 34.43] |
| DFM | CP | Unconditional | 0.0079 [0.0035, 0.0130] | 0 (no match; ≤ 0.0030) | 10.73 [9.74, 11.89] |
| DFM | CP | Minimal Cue | 0.0312 [0.0222, 0.0409] | 0.0070 [0.0020, 0.0130] | 17.22 [15.87, 18.68] |
| DFM | CP | Generic | 0.0061 [0.0052, 0.0071] | 0.0059 [0.0044, 0.0075] | 13.49 [13.19, 13.81] |
| DFM | CP | Specific | 0.0029 [0.0023, 0.0035] | 0.0067 [0.0048, 0.0088] | 17.57 [16.60, 18.60] |
| DFM | CP | Prefix | 0.0260 [0.0201, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.33 [28.63, 32.08] |
| DFM | DW | Unconditional | 0.0539 [0.0457, 0.0629] | 0.0300 [0.0200, 0.0410] | 26.13 [24.84, 27.51] |
| DFM | DW | Minimal Cue | 0.0294 [0.0229, 0.0367] | 0.0200 [0.0120, 0.0290] | 20.23 [19.43, 21.07] |
| DFM | DW | Generic | 0.0008 [0.0005, 0.0012] | 0.0019 [0.0010, 0.0030] | 16.08 [15.70, 16.50] |
| DFM | DW | Specific | 0.0034 [0.0026, 0.0043] | 0.0084 [0.0062, 0.0110] | 19.90 [18.74, 21.14] |
| DFM | DW | Prefix | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.43 [24.04, 26.88] |
| Olmo 3 32B | D3 | Unconditional | 0.0218 [0.0174, 0.0269] | 0 (no match; ≤ 0.0030) | 53.53 [51.09, 56.07] |
| Olmo 3 32B | D3 | Minimal Cue | 0.0124 [0.0086, 0.0166] | 0.0020 [0.0000, 0.0050] | 26.52 [24.47, 28.68] |
| Olmo 3 32B | D3 | Generic | 0.0021 [0.0015, 0.0031] | 0.0002 [0.0000, 0.0005] | 15.10 [14.83, 15.41] |
| Olmo 3 32B | D3 | Specific | 0.0012 [0.0005, 0.0023] | 0.0006 [0.0001, 0.0014] | 17.68 [17.19, 18.23] |
| Olmo 3 32B | D3 | Prefix | 0.0085 [0.0058, 0.0120] | 0.0029 [0.0009, 0.0058] | 26.88 [25.46, 28.38] |

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3.

## Table B — Propensity memorization scores against prefix (cf. paper Tables 2 and 4)

| Model | Data | Metric | Unconditional | Minimal Cue | Generic | Specific |
|---|---|---|---|---|---|---|
| Comma | CP | PM_NVR | 0.502 [0.417, 0.586] | 0.521 [0.439, 0.602] | 0.074 [0.054, 0.100] | 0.105 [0.078, 0.141] |
| Comma | CP | PM_FMR | 0.000 [0.000, 0.000] | 0.349 [0.000, 0.606] | 0.200 [0.101, 0.353] | 0.253 [0.136, 0.424] |
| DFM | CP | PM_NVR | 0.233 [0.119, 0.352] | 0.545 [0.443, 0.638] | 0.191 [0.151, 0.241] | 0.101 [0.075, 0.134] |
| DFM | CP | PM_FMR | 0.000 [0.000, 0.000] | 0.654 [0.370, 0.820] | 0.615 [0.491, 0.755] | 0.644 [0.512, 0.779] |
| DFM | DW | PM_NVR | 0.540 [0.487, 0.590] | 0.390 [0.325, 0.454] | 0.017 [0.010, 0.026] | 0.069 [0.053, 0.087] |
| DFM | DW | PM_FMR | 0.450 [0.343, 0.550] | 0.353 [0.235, 0.458] | 0.049 [0.025, 0.081] | 0.186 [0.137, 0.247] |
| Olmo 3 32B | D3 | PM_NVR | 0.719 [0.631, 0.798] | 0.593 [0.470, 0.705] | 0.201 [0.130, 0.298] | 0.124 [0.054, 0.229] |
| Olmo 3 32B | D3 | PM_FMR | 0.000 [0.000, 0.000] | 0.408 [0.000, 0.755] | 0.065 [0.000, 0.235] | 0.171 [0.023, 0.455] |

PM = f_setting / (f_setting + f_prefix); 0.5 is neutral, lower values mean lower propensity relative to capability. Setting and prefix sets are resampled independently.

## 1. How memorization varies across settings, per model and corpus

### Comma on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0003 [-0.0100, +0.0100] ≈ (prefix ×1.0) | -0.0056 [-0.0090, -0.0029] ↓ | +2.31 [-0.50, +5.09] ≈ (prefix ×0.9) |
| Minimal Cue | +0.0026 [-0.0075, +0.0126] ≈ (prefix ×0.9) | -0.0026 [-0.0070, +0.0020] ≈ (prefix ×1.9) | -4.70 [-7.38, -1.98] ↓ (prefix ×1.2) |
| Generic | -0.0267 [-0.0341, -0.0202] ↓ (prefix ×12.6) | -0.0042 [-0.0077, -0.0014] ↓ (prefix ×4.0) | -18.60 [-20.62, -16.69] ↓ (prefix ×2.3) |
| Specific | -0.0256 [-0.0330, -0.0191] ↓ (prefix ×8.5) | -0.0037 [-0.0072, -0.0009] ↓ (prefix ×2.9) | -16.14 [-18.16, -14.21] ↓ (prefix ×2.0) |

- **NVR ranking:** Minimal Cue (0.0316) > Unconditional (0.0293) > Prefix (0.0290) > Specific (0.0034) > Generic (0.0023).
- **FMR ranking:** Prefix (0.0056) > Minimal Cue (0.0030) > Specific (0.0019) > Generic (0.0014) > Unconditional (0.0000).
- **ALS ranking:** Unconditional (34.79) > Prefix (32.48) > Minimal Cue (27.77) > Specific (16.34) > Generic (13.88).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.502 [0.417, 0.586]; Minimal Cue PM_NVR = 0.521 [0.439, 0.602].

### DFM on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | -0.0181 [-0.0262, -0.0103] ↓ (prefix ×3.3) | -0.0037 [-0.0056, -0.0020] ↓ | -19.59 [-21.65, -17.56] ↓ (prefix ×2.8) |
| Minimal Cue | +0.0052 [-0.0062, +0.0169] ≈ (prefix ×0.8) | +0.0033 [-0.0017, +0.0092] ≈ (prefix ×0.5) | -13.11 [-15.40, -10.88] ↓ (prefix ×1.8) |
| Generic | -0.0199 [-0.0266, -0.0138] ↓ (prefix ×4.2) | +0.0022 [-0.0002, +0.0045] ≈ (prefix ×0.6) | -16.84 [-18.63, -15.14] ↓ (prefix ×2.2) |
| Specific | -0.0231 [-0.0297, -0.0171] ↓ (prefix ×8.9) | +0.0030 [+0.0003, +0.0056] ↑ (prefix ×0.6) | -12.76 [-14.79, -10.81] ↓ (prefix ×1.7) |

- **NVR ranking:** Minimal Cue (0.0312) > Prefix (0.0260) > Unconditional (0.0079) > Generic (0.0061) > Specific (0.0029).
- **FMR ranking:** Minimal Cue (0.0070) > Specific (0.0067) > Generic (0.0059) > Prefix (0.0037) > Unconditional (0.0000).
- **ALS ranking:** Prefix (30.33) > Specific (17.57) > Minimal Cue (17.22) > Generic (13.49) > Unconditional (10.73).
- **Propensity at or above neutral (PM ≥ 0.5):** Minimal Cue PM_NVR = 0.545 [0.443, 0.638]; Minimal Cue PM_FMR = 0.654 [0.370, 0.820]; Generic PM_FMR = 0.615 [0.491, 0.755]; Specific PM_FMR = 0.644 [0.512, 0.779].

### DFM on Dynaword

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0079 [-0.0025, +0.0187] ≈ (prefix ×0.9) | -0.0067 [-0.0201, +0.0070] ≈ (prefix ×1.2) | +0.70 [-1.27, +2.62] ≈ (prefix ×1.0) |
| Minimal Cue | -0.0166 [-0.0257, -0.0073] ↓ (prefix ×1.6) | -0.0167 [-0.0282, -0.0048] ↓ (prefix ×1.8) | -5.20 [-6.86, -3.61] ↓ (prefix ×1.3) |
| Generic | -0.0452 [-0.0514, -0.0394] ↓ (prefix ×57.3) | -0.0348 [-0.0432, -0.0269] ↓ (prefix ×19.3) | -9.36 [-10.88, -7.90] ↓ (prefix ×1.6) |
| Specific | -0.0426 [-0.0489, -0.0368] ↓ (prefix ×13.6) | -0.0283 [-0.0370, -0.0201] ↓ (prefix ×4.4) | -5.53 [-7.42, -3.69] ↓ (prefix ×1.3) |

- **NVR ranking:** Unconditional (0.0539) > Prefix (0.0460) > Minimal Cue (0.0294) > Specific (0.0034) > Generic (0.0008).
- **FMR ranking:** Prefix (0.0367) > Unconditional (0.0300) > Minimal Cue (0.0200) > Specific (0.0084) > Generic (0.0019).
- **ALS ranking:** Unconditional (26.13) > Prefix (25.43) > Minimal Cue (20.23) > Specific (19.90) > Generic (16.08).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.540 [0.487, 0.590].

### Olmo 3 32B on Dolma 3

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0133 [+0.0079, +0.0191] ↑ (prefix ×0.4) | -0.0029 [-0.0057, -0.0009] ↓ | +26.65 [+23.79, +29.53] ↑ (prefix ×0.5) |
| Minimal Cue | +0.0039 [-0.0012, +0.0090] ≈ (prefix ×0.7) | -0.0009 [-0.0046, +0.0030] ≈ (prefix ×1.4) | -0.36 [-2.91, +2.16] ≈ (prefix ×1.0) |
| Generic | -0.0064 [-0.0099, -0.0035] ↓ (prefix ×4.0) | -0.0027 [-0.0055, -0.0007] ↓ (prefix ×14.5) | -11.78 [-13.35, -10.37] ↓ (prefix ×1.8) |
| Specific | -0.0073 [-0.0108, -0.0044] ↓ (prefix ×7.1) | -0.0023 [-0.0052, -0.0001] ↓ (prefix ×4.8) | -9.20 [-10.81, -7.74] ↓ (prefix ×1.5) |

- **NVR ranking:** Unconditional (0.0218) > Minimal Cue (0.0124) > Prefix (0.0085) > Generic (0.0021) > Specific (0.0012).
- **FMR ranking:** Prefix (0.0029) > Minimal Cue (0.0020) > Specific (0.0006) > Generic (0.0002) > Unconditional (0.0000).
- **ALS ranking:** Unconditional (53.53) > Prefix (26.88) > Minimal Cue (26.52) > Specific (17.68) > Generic (15.10).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.719 [0.631, 0.798]; Minimal Cue PM_NVR = 0.593 [0.470, 0.705].

## 2. Continual pre-training: DFM against Comma on Common Pile (cf. paper Section 5.2)

DFM Decoder is Comma continually pre-trained on two-thirds Dynaword and one-third Common Pile. Paired differences DFM − Comma on the same prompts (unconditional: independent samples):

### DFM Stage 1 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0215 [-0.0300, -0.0132] ↓ | +0.0030 [+0.0000, +0.0070] ≈ | -24.20 [-26.53, -21.90] ↓ | independent |
| Minimal Cue | -0.0002 [-0.0088, +0.0081] ≈ | +0.0030 [-0.0030, +0.0090] ≈ | -9.82 [-11.63, -7.96] ↓ | paired |
| Generic | +0.0034 [+0.0025, +0.0044] ↑ | +0.0050 [+0.0034, +0.0066] ↑ | -0.22 [-0.74, +0.28] ≈ | paired |
| Specific | -0.0002 [-0.0011, +0.0007] ≈ | +0.0056 [+0.0034, +0.0079] ↑ | +1.32 [+0.17, +2.61] ↑ | paired |
| Prefix | -0.0032 [-0.0060, -0.0007] ↓ | -0.0014 [-0.0041, +0.0008] ≈ | -2.58 [-3.46, -1.72] ↓ | paired |

### DFM Stage 2 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0229 [-0.0313, -0.0150] ↓ | +0.0000 [+0.0000, +0.0000] ≈ | -24.15 [-26.43, -22.00] ↓ | independent |
| Minimal Cue | -0.0020 [-0.0105, +0.0064] ≈ | +0.0020 [-0.0030, +0.0080] ≈ | -9.88 [-11.61, -8.27] ↓ | paired |
| Generic | +0.0028 [+0.0019, +0.0036] ↑ | +0.0033 [+0.0019, +0.0047] ↑ | -0.49 [-0.92, -0.06] ↓ | paired |
| Specific | -0.0002 [-0.0010, +0.0006] ≈ | +0.0038 [+0.0019, +0.0059] ↑ | +0.74 [-0.30, +1.90] ≈ | paired |
| Prefix | -0.0022 [-0.0050, +0.0004] ≈ | -0.0009 [-0.0028, +0.0007] ≈ | -2.16 [-3.02, -1.30] ↓ | paired |

### DFM − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0214 [-0.0302, -0.0130] ↓ | +0.0000 [+0.0000, +0.0000] ≈ | -24.06 [-26.34, -21.84] ↓ | independent |
| Minimal Cue | -0.0005 [-0.0099, +0.0089] ≈ | +0.0040 [-0.0010, +0.0100] ≈ | -10.55 [-12.25, -8.93] ↓ | paired |
| Generic | +0.0038 [+0.0029, +0.0048] ↑ | +0.0045 [+0.0030, +0.0061] ↑ | -0.39 [-0.84, +0.06] ≈ | paired |
| Specific | -0.0005 [-0.0013, +0.0003] ≈ | +0.0048 [+0.0029, +0.0068] ↑ | +1.22 [+0.27, +2.23] ↑ | paired |
| Prefix | -0.0030 [-0.0056, -0.0005] ↓ | -0.0019 [-0.0044, +0.0001] ≈ | -2.15 [-2.98, -1.35] ↓ | paired |

### Propensity shift, Comma → DFM (final)

| Prompt | PM_NVR | PM_FMR |
|---|---|---|
| Unconditional | 0.502 [0.417, 0.586] → 0.233 [0.119, 0.352]; Δ -0.269 [-0.398, -0.143] ↓ | 0.000 [0.000, 0.000] → 0.000 [0.000, 0.000]; Δ +0.000 [+0.000, +0.000] ≈ |
| Minimal Cue | 0.521 [0.439, 0.602] → 0.545 [0.443, 0.638]; Δ +0.024 [-0.058, +0.097] ≈ | 0.349 [0.000, 0.606] → 0.654 [0.370, 0.820]; Δ +0.305 [-0.000, +0.630] ≈ |
| Generic | 0.074 [0.054, 0.100] → 0.191 [0.151, 0.241]; Δ +0.117 [+0.087, +0.154] ↑ | 0.200 [0.101, 0.353] → 0.615 [0.491, 0.755]; Δ +0.415 [+0.284, +0.541] ↑ |
| Specific | 0.105 [0.078, 0.141] → 0.101 [0.075, 0.134]; Δ -0.005 [-0.031, +0.020] ≈ | 0.253 [0.136, 0.424] → 0.644 [0.512, 0.779]; Δ +0.391 [+0.242, +0.520] ↑ |

## 3. Common Pile and Dynaword memorization profiles of DFM (cf. paper Section 5.2)

CP / DW values of DFM (final) and their difference CP − DW (different prompt sets: independent resampling):

| Prompt | ALS CP / DW | FMR CP / DW | NVR CP / DW |
|---|---|---|---|
| Unconditional | 10.73 / 26.13; Δ -15.39 [-17.11, -13.69] ↓ | 0.0000 / 0.0300; Δ -0.0300 [-0.0410, -0.0200] ↓ | 0.0079 / 0.0539; Δ -0.0460 [-0.0559, -0.0365] ↓ |
| Minimal Cue | 17.22 / 20.23; Δ -3.01 [-4.65, -1.35] ↓ | 0.0070 / 0.0200; Δ -0.0130 [-0.0230, -0.0030] ↓ | 0.0312 / 0.0294; Δ +0.0018 [-0.0099, +0.0140] ≈ |
| Generic | 13.49 / 16.08; Δ -2.59 [-3.11, -2.09] ↓ | 0.0059 / 0.0019; Δ +0.0040 [+0.0021, +0.0059] ↑ | 0.0061 / 0.0008; Δ +0.0053 [+0.0043, +0.0064] ↑ |
| Specific | 17.57 / 19.90; Δ -2.33 [-3.93, -0.78] ↓ | 0.0067 / 0.0084; Δ -0.0017 [-0.0049, +0.0014] ≈ | 0.0029 / 0.0034; Δ -0.0005 [-0.0015, +0.0005] ≈ |
| Prefix | 30.33 / 25.43; Δ +4.90 [+2.61, +7.15] ↑ | 0.0037 / 0.0367; Δ -0.0330 [-0.0417, -0.0249] ↓ | 0.0260 / 0.0460; Δ -0.0200 [-0.0285, -0.0113] ↓ |

## 4. Memorization across DFM training stages (cf. paper Section 5.3 and Appendix C)

### Dynaword

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0471 [0.0394, 0.0553] | 0.0290 [0.0190, 0.0400] | 25.65 [24.20, 27.33] | 0.552 [0.497, 0.605] | 0.464 [0.353, 0.563] |
| Unconditional | Stage 2 | 0.0492 [0.0415, 0.0574] | 0.0180 [0.0100, 0.0270] | 27.44 [25.97, 28.99] | 0.533 [0.481, 0.585] | 0.342 [0.223, 0.452] |
| Unconditional | Final | 0.0539 [0.0457, 0.0629] | 0.0300 [0.0200, 0.0410] | 26.13 [24.84, 27.51] | 0.540 [0.487, 0.590] | 0.450 [0.343, 0.550] |
| Minimal Cue | Stage 1 | 0.0223 [0.0165, 0.0287] | 0.0210 [0.0130, 0.0300] | 19.29 [18.59, 20.02] | 0.368 [0.298, 0.439] | 0.385 [0.266, 0.494] |
| Minimal Cue | Stage 2 | 0.0238 [0.0175, 0.0306] | 0.0210 [0.0130, 0.0300] | 19.83 [19.09, 20.62] | 0.356 [0.285, 0.426] | 0.377 [0.259, 0.482] |
| Minimal Cue | Final | 0.0294 [0.0229, 0.0367] | 0.0200 [0.0120, 0.0290] | 20.23 [19.43, 21.07] | 0.390 [0.325, 0.454] | 0.353 [0.235, 0.458] |
| Generic | Stage 1 | 0.0005 [0.0003, 0.0007] | 0.0015 [0.0006, 0.0027] | 15.95 [15.59, 16.36] | 0.012 [0.008, 0.018] | 0.043 [0.018, 0.077] |
| Generic | Stage 2 | 0.0006 [0.0004, 0.0010] | 0.0015 [0.0007, 0.0025] | 16.00 [15.66, 16.38] | 0.015 [0.009, 0.022] | 0.041 [0.018, 0.072] |
| Generic | Final | 0.0008 [0.0005, 0.0012] | 0.0019 [0.0010, 0.0030] | 16.08 [15.70, 16.50] | 0.017 [0.010, 0.026] | 0.049 [0.025, 0.081] |
| Specific | Stage 1 | 0.0031 [0.0023, 0.0039] | 0.0065 [0.0046, 0.0086] | 18.97 [17.90, 20.09] | 0.074 [0.056, 0.095] | 0.162 [0.114, 0.221] |
| Specific | Stage 2 | 0.0036 [0.0026, 0.0047] | 0.0065 [0.0043, 0.0091] | 18.28 [17.35, 19.27] | 0.077 [0.056, 0.102] | 0.158 [0.106, 0.222] |
| Specific | Final | 0.0034 [0.0026, 0.0043] | 0.0084 [0.0062, 0.0110] | 19.90 [18.74, 21.14] | 0.069 [0.053, 0.087] | 0.186 [0.137, 0.247] |
| Prefix | Stage 1 | 0.0382 [0.0333, 0.0435] | 0.0335 [0.0260, 0.0412] | 22.87 [21.78, 24.03] | — | — |
| Prefix | Stage 2 | 0.0431 [0.0374, 0.0489] | 0.0347 [0.0269, 0.0427] | 24.14 [22.90, 25.41] | — | — |
| Prefix | Final | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.43 [24.04, 26.88] | — | — |

Significant paired differences from the final checkpoint (8 of 30 stage–setting–metric comparisons):

- Stage 1 − Final, Minimal Cue, NVR: -0.0071 [-0.0138, -0.0007] ↓
- Stage 1 − Final, Minimal Cue, ALS: -0.94 [-1.63, -0.24] ↓
- Stage 2 − Final, Specific, FMR: -0.0019 [-0.0035, -0.0003] ↓
- Stage 2 − Final, Specific, ALS: -1.62 [-2.36, -0.96] ↓
- Stage 1 − Final, Prefix, NVR: -0.0077 [-0.0114, -0.0044] ↓
- Stage 1 − Final, Prefix, ALS: -2.56 [-3.29, -1.88] ↓
- Stage 2 − Final, Prefix, NVR: -0.0029 [-0.0053, -0.0006] ↓
- Stage 2 − Final, Prefix, ALS: -1.29 [-1.82, -0.82] ↓

### Common Pile

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0078 [0.0038, 0.0126] | 0.0030 [0.0000, 0.0070] | 10.60 [9.59, 11.92] | 0.233 [0.128, 0.346] | 0.417 [0.000, 0.667] |
| Unconditional | Stage 2 | 0.0064 [0.0027, 0.0107] | 0 (no match; ≤ 0.0030) | 10.65 [9.69, 11.74] | 0.193 [0.093, 0.301] | 0.000 [0.000, 0.000] |
| Unconditional | Final | 0.0079 [0.0035, 0.0130] | 0 (no match; ≤ 0.0030) | 10.73 [9.74, 11.89] | 0.233 [0.119, 0.352] | 0.000 [0.000, 0.000] |
| Minimal Cue | Stage 1 | 0.0314 [0.0231, 0.0406] | 0.0060 [0.0020, 0.0110] | 17.96 [16.37, 19.72] | 0.549 [0.456, 0.636] | 0.588 [0.278, 0.779] |
| Minimal Cue | Stage 2 | 0.0297 [0.0218, 0.0383] | 0.0050 [0.0010, 0.0100] | 17.90 [16.43, 19.47] | 0.525 [0.430, 0.615] | 0.515 [0.185, 0.722] |
| Minimal Cue | Final | 0.0312 [0.0222, 0.0409] | 0.0070 [0.0020, 0.0130] | 17.22 [15.87, 18.68] | 0.545 [0.443, 0.638] | 0.654 [0.370, 0.820] |
| Generic | Stage 1 | 0.0057 [0.0048, 0.0068] | 0.0064 [0.0048, 0.0081] | 13.66 [13.28, 14.07] | 0.181 [0.143, 0.230] | 0.604 [0.474, 0.753] |
| Generic | Stage 2 | 0.0051 [0.0043, 0.0059] | 0.0047 [0.0033, 0.0061] | 13.39 [13.08, 13.72] | 0.159 [0.125, 0.203] | 0.500 [0.368, 0.662] |
| Generic | Final | 0.0061 [0.0052, 0.0071] | 0.0059 [0.0044, 0.0075] | 13.49 [13.19, 13.81] | 0.191 [0.151, 0.241] | 0.615 [0.491, 0.755] |
| Specific | Stage 1 | 0.0032 [0.0025, 0.0040] | 0.0075 [0.0054, 0.0099] | 17.66 [16.49, 18.97] | 0.111 [0.082, 0.149] | 0.641 [0.504, 0.782] |
| Specific | Stage 2 | 0.0032 [0.0026, 0.0039] | 0.0057 [0.0039, 0.0078] | 17.08 [16.04, 18.26] | 0.108 [0.082, 0.144] | 0.548 [0.404, 0.705] |
| Specific | Final | 0.0029 [0.0023, 0.0035] | 0.0067 [0.0048, 0.0088] | 17.57 [16.60, 18.60] | 0.101 [0.075, 0.134] | 0.644 [0.512, 0.779] |
| Prefix | Stage 1 | 0.0258 [0.0200, 0.0319] | 0.0042 [0.0022, 0.0066] | 29.90 [28.23, 31.65] | — | — |
| Prefix | Stage 2 | 0.0268 [0.0208, 0.0333] | 0.0047 [0.0026, 0.0072] | 30.32 [28.58, 32.16] | — | — |
| Prefix | Final | 0.0260 [0.0201, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.33 [28.63, 32.08] | — | — |

Significant paired differences from the final checkpoint (1 of 30 stage–setting–metric comparisons):

- Stage 2 − Final, Generic, NVR: -0.0011 [-0.0019, -0.0002] ↓

## 5. The paper's claims, re-checked on these results

Statuses follow fixed rules: point estimates for orderings and the 0.5 threshold, and the sign of 95% CIs for differences. *Partly* means some but not all of the stated comparisons hold.

| Claim (paper section) | Status | Evidence |
|---|---|---|
| Prefix yields higher NVR and ALS point estimates than generic and specific prompting, for every model–corpus pair (Abstract, §5.2, §6) | holds | 12/12 comparisons; exceptions: none |
| … also for Olmo 3 on Dolma 3 (new) | holds | 4/4; exceptions: none |
| … also against the new prompt-free settings (unconditional, minimal cue) | partly holds | 7/16; exceptions: Comma CP Unconditional NVR, Comma CP Unconditional ALS, Comma CP Minimal Cue NVR, DFM CP Minimal Cue NVR, DFM DW Unconditional NVR, DFM DW Unconditional ALS, Olmo 3 32B D3 Unconditional NVR, Olmo 3 32B D3 Unconditional ALS, Olmo 3 32B D3 Minimal Cue NVR |
| Propensity scores are generally below the neutral 0.5 (Abstract, §5.2, §6) — generic and specific | partly holds | 14/16 below 0.5; at/above: DFM CP Generic PM_FMR 0.615 [0.491, 0.755]; DFM CP Specific PM_FMR 0.644 [0.512, 0.779] |
| … for the new prompt-free settings | partly holds | 9/16 below 0.5; at/above: Comma CP Unconditional PM_NVR 0.502 [0.417, 0.586]; Comma CP Minimal Cue PM_NVR 0.521 [0.439, 0.602]; DFM CP Minimal Cue PM_NVR 0.545 [0.443, 0.638]; DFM CP Minimal Cue PM_FMR 0.654 [0.370, 0.820]; DFM DW Unconditional PM_NVR 0.540 [0.487, 0.590]; Olmo 3 32B D3 Unconditional PM_NVR 0.719 [0.631, 0.798]; Olmo 3 32B D3 Minimal Cue PM_NVR 0.593 [0.470, 0.705] |
| Comma produces longer verbatim spans than DFM on Common Pile under generic and prefix prompting (§5.2) | partly holds | DFM − Comma: ALS Generic -0.39 [-0.84, +0.06] ≈; ALS Prefix -2.15 [-2.98, -1.35] ↓ |
| Comma shows more full-generation memorization of Common Pile than DFM (specific, prefix) (§5.2) | does not hold | DFM − Comma: FMR Specific +0.0048 [+0.0029, +0.0068] ↑; FMR Prefix -0.0019 [-0.0044, +0.0001] ≈ |
| Specific prompts match the prefix FMR for Comma on Common Pile (§5.1) | does not hold | specific − prefix -0.0037 [-0.0072, -0.0009] ↓ |
| From Comma to DFM, specific-prompt PM_FMR decreases while PM_NVR increases (§5.2) | does not hold | PM_FMR Δ +0.391 [+0.242, +0.520] ↑; PM_NVR Δ -0.005 [-0.031, +0.020] ≈ |
| For DFM, Common Pile yields longer ALS than Dynaword in every setting (§5.2) | partly holds | CP − DW: Unconditional -15.39 [-17.11, -13.69] ↓; Minimal Cue -3.01 [-4.65, -1.35] ↓; Generic -2.59 [-3.11, -2.09] ↓; Specific -2.33 [-3.93, -0.78] ↓; Prefix +4.90 [+2.61, +7.15] ↑ |
| For DFM, prefix FMR is higher on Dynaword than on Common Pile (§5.2) | holds | DW − CP +0.0330 [+0.0248, +0.0415] ↑ |
| Memorization is essentially stable across DFM training stages (§5.3) | partly holds | 9/60 stage − final paired differences significant (see Section 4) |


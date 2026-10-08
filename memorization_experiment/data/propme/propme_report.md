# PropMe results report

Generated 2026-10-08 by `05_propensity_metrics/make_propme_report.py` from the SimpleTrace traces in `memorization_experiment/data/propme/`.

## Setup and changes since the paper

| | Paper (ARR, 12 Oct) | This report |
|---|---|---|
| Models / corpora | Comma (CP); DFM Decoder stages 1, 2, final (CP, DW) | same, plus **Olmo 3 32B on Dolma 3 (D3)** |
| Decoding | greedy (temperature 0), 1 generation per prompt | sampling, temperature 0.7, top-p 0.95, 256 new tokens, **10 generations per prompt** |
| Prompts per setting | 100 | 1000 prompts × 10 = 10000 generations (prompt-free settings: 10000 generations) |
| Generic | GPT-generated plausible prompts | Tatoeba sentences not verbatim in the index (random 1000-prompt sample) |
| Specific | GPT-generated, dataset-inspired | sentences from unseen sources close to the training data (00_prepare_data/propensity_settings, 1000 per corpus) |
| Prefix (capability) | first 50 tokens of training documents | same construction, random 1000-prompt sample |
| New propensity settings | — | **Unconditional** (start-of-document token only) and **Minimal cue** (start-of-document token + one frequent word drawn per generation) |
| Confidence intervals | Wilson 95% for NVR and FMR | percentile bootstrap 95% (B = 10,000) for NVR, FMR, ALS and PM; resampling prompts (each with its 10 generations) for generic, specific and prefix, generations for unconditional and minimal cue |

Settings are listed from the least to the most targeted prompting: Unconditional, Minimal cue, Generic, Specific (propensity settings) and Prefix (capability setting). A full-match ratio of 0 is shown with its rule-of-three upper bound 3/n (n = independent units). ↑ / ↓ mark differences whose 95% CI lies above / below 0, ≈ those whose CI includes 0. † PM is 0 by convention when both values are 0 (paper, Eq. 1).

## Table A — Memorization metrics across model–corpus pairs (cf. paper Tables 1 and 3)

| Model | Data | Prompt | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] |
|---|---|---|---|---|---|
| Comma | CP | Unconditional | 0.0353 [0.0327, 0.0379] | 0.0019 [0.0011, 0.0028] | 35.85 [35.17, 36.52] |
| Comma | CP | Minimal Cue | 0.0338 [0.0313, 0.0365] | 0.0035 [0.0024, 0.0047] | 29.12 [28.49, 29.78] |
| Comma | CP | Generic | 0.0023 [0.0018, 0.0028] | 0.0014 [0.0007, 0.0022] | 13.88 [13.53, 14.27] |
| Comma | CP | Specific | 0.0034 [0.0027, 0.0042] | 0.0019 [0.0010, 0.0029] | 16.34 [15.88, 16.83] |
| Comma | CP | Prefix | 0.0290 [0.0226, 0.0362] | 0.0056 [0.0028, 0.0089] | 32.48 [30.62, 34.43] |
| DFM | CP | Unconditional | 0.0072 [0.0058, 0.0087] | 0.0015 [0.0008, 0.0023] | 10.31 [10.04, 10.61] |
| DFM | CP | Minimal Cue | 0.0286 [0.0260, 0.0313] | 0.0071 [0.0055, 0.0087] | 18.71 [18.15, 19.29] |
| DFM | CP | Generic | 0.0061 [0.0052, 0.0071] | 0.0059 [0.0044, 0.0075] | 13.49 [13.19, 13.81] |
| DFM | CP | Specific | 0.0029 [0.0023, 0.0035] | 0.0067 [0.0048, 0.0088] | 17.57 [16.60, 18.60] |
| DFM | CP | Prefix | 0.0260 [0.0201, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.33 [28.63, 32.08] |
| DFM | DW | Unconditional | 0.0475 [0.0450, 0.0500] | 0.0287 [0.0255, 0.0320] | 26.25 [25.83, 26.69] |
| DFM | DW | Minimal Cue | 0.0273 [0.0253, 0.0294] | 0.0208 [0.0180, 0.0237] | 20.30 [20.02, 20.60] |
| DFM | DW | Generic | 0.0008 [0.0005, 0.0012] | 0.0019 [0.0010, 0.0030] | 16.08 [15.70, 16.50] |
| DFM | DW | Specific | 0.0034 [0.0026, 0.0043] | 0.0084 [0.0062, 0.0110] | 19.90 [18.74, 21.14] |
| DFM | DW | Prefix | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.43 [24.04, 26.88] |
| Olmo 3 32B | D3 | Unconditional | 0.0249 [0.0235, 0.0263] | 0.0007 [0.0002, 0.0013] | 53.49 [52.69, 54.31] |
| Olmo 3 32B | D3 | Minimal Cue | 0.0121 [0.0109, 0.0133] | 0.0040 [0.0028, 0.0053] | 26.54 [25.90, 27.19] |
| Olmo 3 32B | D3 | Generic | 0.0021 [0.0015, 0.0031] | 0.0002 [0.0000, 0.0005] | 15.10 [14.83, 15.41] |
| Olmo 3 32B | D3 | Specific | 0.0012 [0.0005, 0.0023] | 0.0006 [0.0001, 0.0014] | 17.68 [17.19, 18.23] |
| Olmo 3 32B | D3 | Prefix | 0.0085 [0.0058, 0.0120] | 0.0029 [0.0009, 0.0058] | 26.88 [25.46, 28.38] |

Higher values indicate stronger memorization signals. CP: Common Pile, DW: Dynaword, D3: Dolma 3.

## Table B — Propensity memorization scores against prefix (cf. paper Tables 2 and 4)

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

## 1. How memorization varies across settings, per model and corpus

### Comma on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0062 [-0.0016, +0.0132] ≈ (prefix ×0.8) | -0.0037 [-0.0072, -0.0008] ↓ (prefix ×2.9) | +3.37 [+1.27, +5.37] ↑ (prefix ×0.9) |
| Minimal Cue | +0.0048 [-0.0029, +0.0119] ≈ (prefix ×0.9) | -0.0021 [-0.0056, +0.0009] ≈ (prefix ×1.6) | -3.36 [-5.44, -1.36] ↓ (prefix ×1.1) |
| Generic | -0.0267 [-0.0341, -0.0202] ↓ (prefix ×12.6) | -0.0042 [-0.0077, -0.0014] ↓ (prefix ×4.0) | -18.60 [-20.62, -16.69] ↓ (prefix ×2.3) |
| Specific | -0.0256 [-0.0330, -0.0191] ↓ (prefix ×8.5) | -0.0037 [-0.0072, -0.0009] ↓ (prefix ×2.9) | -16.14 [-18.16, -14.21] ↓ (prefix ×2.0) |

- **NVR ranking:** Unconditional (0.0353) > Minimal Cue (0.0338) > Prefix (0.0290) > Specific (0.0034) > Generic (0.0023).
- **FMR ranking:** Prefix (0.0056) > Minimal Cue (0.0035) > Unconditional (0.0019) > Specific (0.0019) > Generic (0.0014).
- **ALS ranking:** Unconditional (35.85) > Prefix (32.48) > Minimal Cue (29.12) > Specific (16.34) > Generic (13.88).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.548 [0.490, 0.614]; Minimal Cue PM_NVR = 0.538 [0.480, 0.603].

### DFM on Common Pile

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | -0.0188 [-0.0254, -0.0127] ↓ (prefix ×3.6) | -0.0022 [-0.0043, -0.0003] ↓ (prefix ×2.5) | -20.01 [-21.80, -18.32] ↓ (prefix ×2.9) |
| Minimal Cue | +0.0026 [-0.0044, +0.0090] ≈ (prefix ×0.9) | +0.0034 [+0.0008, +0.0058] ↑ (prefix ×0.5) | -11.61 [-13.46, -9.83] ↓ (prefix ×1.6) |
| Generic | -0.0199 [-0.0266, -0.0138] ↓ (prefix ×4.2) | +0.0022 [-0.0002, +0.0045] ≈ (prefix ×0.6) | -16.84 [-18.63, -15.14] ↓ (prefix ×2.2) |
| Specific | -0.0231 [-0.0297, -0.0171] ↓ (prefix ×8.9) | +0.0030 [+0.0003, +0.0056] ↑ (prefix ×0.6) | -12.76 [-14.79, -10.81] ↓ (prefix ×1.7) |

- **NVR ranking:** Minimal Cue (0.0286) > Prefix (0.0260) > Unconditional (0.0072) > Generic (0.0061) > Specific (0.0029).
- **FMR ranking:** Minimal Cue (0.0071) > Specific (0.0067) > Generic (0.0059) > Prefix (0.0037) > Unconditional (0.0015).
- **ALS ranking:** Prefix (30.33) > Minimal Cue (18.71) > Specific (17.57) > Generic (13.49) > Unconditional (10.31).
- **Propensity at or above neutral (PM ≥ 0.5):** Minimal Cue PM_NVR = 0.523 [0.464, 0.591]; Minimal Cue PM_FMR = 0.657 [0.538, 0.788]; Generic PM_FMR = 0.615 [0.491, 0.755]; Specific PM_FMR = 0.644 [0.512, 0.779].

### DFM on Dynaword

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0015 [-0.0051, +0.0077] ≈ (prefix ×1.0) | -0.0080 [-0.0170, +0.0006] ≈ (prefix ×1.3) | +0.82 [-0.67, +2.28] ≈ (prefix ×1.0) |
| Minimal Cue | -0.0187 [-0.0252, -0.0125] ↓ (prefix ×1.7) | -0.0159 [-0.0248, -0.0076] ↓ (prefix ×1.8) | -5.13 [-6.59, -3.73] ↓ (prefix ×1.3) |
| Generic | -0.0452 [-0.0514, -0.0394] ↓ (prefix ×57.3) | -0.0348 [-0.0432, -0.0269] ↓ (prefix ×19.3) | -9.36 [-10.88, -7.90] ↓ (prefix ×1.6) |
| Specific | -0.0426 [-0.0489, -0.0368] ↓ (prefix ×13.6) | -0.0283 [-0.0370, -0.0201] ↓ (prefix ×4.4) | -5.53 [-7.42, -3.69] ↓ (prefix ×1.3) |

- **NVR ranking:** Unconditional (0.0475) > Prefix (0.0460) > Minimal Cue (0.0273) > Specific (0.0034) > Generic (0.0008).
- **FMR ranking:** Prefix (0.0367) > Unconditional (0.0287) > Minimal Cue (0.0208) > Specific (0.0084) > Generic (0.0019).
- **ALS ranking:** Unconditional (26.25) > Prefix (25.43) > Minimal Cue (20.30) > Specific (19.90) > Generic (16.08).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.508 [0.473, 0.544].

### Olmo 3 32B on Dolma 3

Difference from prefix (setting − prefix) and how many times larger prefix is:

| Setting | ΔNVR | ΔFMR | ΔALS |
|---|---|---|---|
| Unconditional | +0.0163 [+0.0127, +0.0194] ↑ (prefix ×0.3) | -0.0022 [-0.0050, -0.0001] ↓ (prefix ×4.1) | +26.61 [+24.90, +28.23] ↑ (prefix ×0.5) |
| Minimal Cue | +0.0036 [-0.0001, +0.0065] ≈ (prefix ×0.7) | +0.0011 [-0.0019, +0.0035] ≈ (prefix ×0.7) | -0.34 [-2.01, +1.22] ≈ (prefix ×1.0) |
| Generic | -0.0064 [-0.0099, -0.0035] ↓ (prefix ×4.0) | -0.0027 [-0.0055, -0.0007] ↓ (prefix ×14.5) | -11.78 [-13.35, -10.37] ↓ (prefix ×1.8) |
| Specific | -0.0073 [-0.0108, -0.0044] ↓ (prefix ×7.1) | -0.0023 [-0.0052, -0.0001] ↓ (prefix ×4.8) | -9.20 [-10.81, -7.74] ↓ (prefix ×1.5) |

- **NVR ranking:** Unconditional (0.0249) > Minimal Cue (0.0121) > Prefix (0.0085) > Generic (0.0021) > Specific (0.0012).
- **FMR ranking:** Minimal Cue (0.0040) > Prefix (0.0029) > Unconditional (0.0007) > Specific (0.0006) > Generic (0.0002).
- **ALS ranking:** Unconditional (53.49) > Prefix (26.88) > Minimal Cue (26.54) > Specific (17.68) > Generic (15.10).
- **Propensity at or above neutral (PM ≥ 0.5):** Unconditional PM_NVR = 0.745 [0.675, 0.812]; Minimal Cue PM_NVR = 0.586 [0.497, 0.679]; Minimal Cue PM_FMR = 0.580 [0.397, 0.820].

## 2. Continual pre-training: DFM against Comma on Common Pile (cf. paper Section 5.2)

DFM Decoder is Comma continually pre-trained on two-thirds Dynaword and one-third Common Pile. Paired differences DFM − Comma on the same prompts (unconditional: independent samples):

### DFM Stage 1 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0256 [-0.0288, -0.0225] ↓ | +0.0038 [+0.0021, +0.0056] ↑ | -25.06 [-25.81, -24.32] ↓ | independent |
| Minimal Cue | -0.0063 [-0.0089, -0.0037] ↓ | +0.0027 [+0.0009, +0.0045] ↑ | -10.20 [-10.81, -9.59] ↓ | paired |
| Generic | +0.0034 [+0.0025, +0.0044] ↑ | +0.0050 [+0.0034, +0.0066] ↑ | -0.22 [-0.74, +0.28] ≈ | paired |
| Specific | -0.0002 [-0.0011, +0.0007] ≈ | +0.0056 [+0.0034, +0.0079] ↑ | +1.32 [+0.17, +2.61] ↑ | paired |
| Prefix | -0.0032 [-0.0060, -0.0007] ↓ | -0.0014 [-0.0041, +0.0008] ≈ | -2.58 [-3.46, -1.72] ↓ | paired |

### DFM Stage 2 − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0301 [-0.0330, -0.0272] ↓ | +0.0009 [-0.0004, +0.0023] ≈ | -25.78 [-26.49, -25.08] ↓ | independent |
| Minimal Cue | -0.0067 [-0.0093, -0.0040] ↓ | +0.0026 [+0.0008, +0.0044] ↑ | -10.05 [-10.66, -9.44] ↓ | paired |
| Generic | +0.0028 [+0.0019, +0.0036] ↑ | +0.0033 [+0.0019, +0.0047] ↑ | -0.49 [-0.92, -0.06] ↓ | paired |
| Specific | -0.0002 [-0.0010, +0.0006] ≈ | +0.0038 [+0.0019, +0.0059] ↑ | +0.74 [-0.30, +1.90] ≈ | paired |
| Prefix | -0.0022 [-0.0050, +0.0004] ≈ | -0.0009 [-0.0028, +0.0007] ≈ | -2.16 [-3.02, -1.30] ↓ | paired |

### DFM − Comma

| Prompt | ΔNVR | ΔFMR | ΔALS | Resampling |
|---|---|---|---|---|
| Unconditional | -0.0280 [-0.0310, -0.0250] ↓ | -0.0004 [-0.0016, +0.0008] ≈ | -25.54 [-26.26, -24.82] ↓ | independent |
| Minimal Cue | -0.0053 [-0.0080, -0.0026] ↓ | +0.0036 [+0.0017, +0.0055] ↑ | -10.41 [-11.02, -9.78] ↓ | paired |
| Generic | +0.0038 [+0.0029, +0.0048] ↑ | +0.0045 [+0.0030, +0.0061] ↑ | -0.39 [-0.84, +0.06] ≈ | paired |
| Specific | -0.0005 [-0.0013, +0.0003] ≈ | +0.0048 [+0.0029, +0.0068] ↑ | +1.22 [+0.27, +2.23] ↑ | paired |
| Prefix | -0.0030 [-0.0056, -0.0005] ↓ | -0.0019 [-0.0044, +0.0001] ≈ | -2.15 [-2.98, -1.35] ↓ | paired |

### Propensity shift, Comma → DFM (final)

| Prompt | PM_NVR | PM_FMR |
|---|---|---|
| Unconditional | 0.548 [0.490, 0.614] → 0.218 [0.170, 0.277]; Δ -0.331 [-0.378, -0.284] ↓ | 0.253 [0.141, 0.426] → 0.288 [0.158, 0.460]; Δ +0.035 [-0.133, +0.196] ≈ |
| Minimal Cue | 0.538 [0.480, 0.603] → 0.523 [0.464, 0.591]; Δ -0.015 [-0.047, +0.017] ≈ | 0.385 [0.256, 0.561] → 0.657 [0.538, 0.788]; Δ +0.273 [+0.133, +0.402] ↑ |
| Generic | 0.074 [0.054, 0.100] → 0.191 [0.151, 0.241]; Δ +0.117 [+0.087, +0.154] ↑ | 0.200 [0.101, 0.353] → 0.615 [0.491, 0.755]; Δ +0.415 [+0.284, +0.541] ↑ |
| Specific | 0.105 [0.078, 0.141] → 0.101 [0.075, 0.134]; Δ -0.005 [-0.031, +0.020] ≈ | 0.253 [0.136, 0.424] → 0.644 [0.512, 0.779]; Δ +0.391 [+0.242, +0.520] ↑ |

## 3. Common Pile and Dynaword memorization profiles of DFM (cf. paper Section 5.2)

CP / DW values of DFM (final) and their difference CP − DW (different prompt sets: independent resampling):

| Prompt | ALS CP / DW | FMR CP / DW | NVR CP / DW |
|---|---|---|---|
| Unconditional | 10.31 / 26.25; Δ -15.94 [-16.46, -15.43] ↓ | 0.0015 / 0.0287; Δ -0.0272 [-0.0306, -0.0239] ↓ | 0.0072 / 0.0475; Δ -0.0402 [-0.0432, -0.0373] ↓ |
| Minimal Cue | 18.71 / 20.30; Δ -1.59 [-2.22, -0.95] ↓ | 0.0071 / 0.0208; Δ -0.0137 [-0.0170, -0.0105] ↓ | 0.0286 / 0.0273; Δ +0.0013 [-0.0020, +0.0046] ≈ |
| Generic | 13.49 / 16.08; Δ -2.59 [-3.11, -2.09] ↓ | 0.0059 / 0.0019; Δ +0.0040 [+0.0021, +0.0059] ↑ | 0.0061 / 0.0008; Δ +0.0053 [+0.0043, +0.0064] ↑ |
| Specific | 17.57 / 19.90; Δ -2.33 [-3.93, -0.78] ↓ | 0.0067 / 0.0084; Δ -0.0017 [-0.0049, +0.0014] ≈ | 0.0029 / 0.0034; Δ -0.0005 [-0.0015, +0.0005] ≈ |
| Prefix | 30.33 / 25.43; Δ +4.90 [+2.61, +7.15] ↑ | 0.0037 / 0.0367; Δ -0.0330 [-0.0417, -0.0249] ↓ | 0.0260 / 0.0460; Δ -0.0200 [-0.0285, -0.0113] ↓ |

## 4. Memorization across DFM training stages (cf. paper Section 5.3 and Appendix C)

### Dynaword

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0430 [0.0406, 0.0455] | 0.0367 [0.0331, 0.0403] | 25.00 [24.56, 25.48] | 0.529 [0.493, 0.566] | 0.523 [0.464, 0.588] |
| Unconditional | Stage 2 | 0.0450 [0.0425, 0.0476] | 0.0280 [0.0248, 0.0313] | 25.72 [25.31, 26.14] | 0.511 [0.476, 0.548] | 0.447 [0.388, 0.513] |
| Unconditional | Final | 0.0475 [0.0450, 0.0500] | 0.0287 [0.0255, 0.0320] | 26.25 [25.83, 26.69] | 0.508 [0.473, 0.544] | 0.439 [0.382, 0.505] |
| Minimal Cue | Stage 1 | 0.0245 [0.0225, 0.0265] | 0.0206 [0.0178, 0.0234] | 19.52 [19.24, 19.82] | 0.390 [0.353, 0.429] | 0.381 [0.322, 0.448] |
| Minimal Cue | Stage 2 | 0.0244 [0.0225, 0.0265] | 0.0192 [0.0166, 0.0220] | 19.73 [19.47, 20.01] | 0.362 [0.326, 0.400] | 0.356 [0.301, 0.423] |
| Minimal Cue | Final | 0.0273 [0.0253, 0.0294] | 0.0208 [0.0180, 0.0237] | 20.30 [20.02, 20.60] | 0.372 [0.338, 0.410] | 0.362 [0.306, 0.428] |
| Generic | Stage 1 | 0.0005 [0.0003, 0.0007] | 0.0015 [0.0006, 0.0027] | 15.95 [15.59, 16.36] | 0.012 [0.008, 0.018] | 0.043 [0.018, 0.077] |
| Generic | Stage 2 | 0.0006 [0.0004, 0.0010] | 0.0015 [0.0007, 0.0025] | 16.00 [15.66, 16.38] | 0.015 [0.009, 0.022] | 0.041 [0.018, 0.072] |
| Generic | Final | 0.0008 [0.0005, 0.0012] | 0.0019 [0.0010, 0.0030] | 16.08 [15.70, 16.50] | 0.017 [0.010, 0.026] | 0.049 [0.025, 0.081] |
| Specific | Stage 1 | 0.0031 [0.0023, 0.0039] | 0.0065 [0.0046, 0.0086] | 18.97 [17.90, 20.09] | 0.074 [0.056, 0.095] | 0.162 [0.114, 0.221] |
| Specific | Stage 2 | 0.0036 [0.0026, 0.0047] | 0.0065 [0.0043, 0.0091] | 18.28 [17.35, 19.27] | 0.077 [0.056, 0.102] | 0.158 [0.106, 0.222] |
| Specific | Final | 0.0034 [0.0026, 0.0043] | 0.0084 [0.0062, 0.0110] | 19.90 [18.74, 21.14] | 0.069 [0.053, 0.087] | 0.186 [0.137, 0.247] |
| Prefix | Stage 1 | 0.0382 [0.0333, 0.0435] | 0.0335 [0.0260, 0.0412] | 22.87 [21.78, 24.03] | — | — |
| Prefix | Stage 2 | 0.0431 [0.0374, 0.0489] | 0.0347 [0.0269, 0.0427] | 24.14 [22.90, 25.41] | — | — |
| Prefix | Final | 0.0460 [0.0402, 0.0520] | 0.0367 [0.0287, 0.0451] | 25.43 [24.04, 26.88] | — | — |

Significant paired differences from the final checkpoint (13 of 30 stage–setting–metric comparisons):

- Stage 1 − Final, Unconditional, NVR: -0.0045 [-0.0080, -0.0009] ↓
- Stage 1 − Final, Unconditional, FMR: +0.0080 [+0.0031, +0.0129] ↑
- Stage 1 − Final, Unconditional, ALS: -1.25 [-1.88, -0.62] ↓
- Stage 1 − Final, Minimal Cue, NVR: -0.0028 [-0.0047, -0.0009] ↓
- Stage 1 − Final, Minimal Cue, ALS: -0.78 [-1.06, -0.48] ↓
- Stage 2 − Final, Minimal Cue, NVR: -0.0029 [-0.0047, -0.0010] ↓
- Stage 2 − Final, Minimal Cue, ALS: -0.57 [-0.84, -0.29] ↓
- Stage 2 − Final, Specific, FMR: -0.0019 [-0.0035, -0.0003] ↓
- Stage 2 − Final, Specific, ALS: -1.62 [-2.36, -0.96] ↓
- Stage 1 − Final, Prefix, NVR: -0.0077 [-0.0114, -0.0044] ↓
- Stage 1 − Final, Prefix, ALS: -2.56 [-3.29, -1.88] ↓
- Stage 2 − Final, Prefix, NVR: -0.0029 [-0.0053, -0.0006] ↓
- Stage 2 − Final, Prefix, ALS: -1.29 [-1.82, -0.82] ↓

### Common Pile

| Prompt | Stage | NVR [95% CI] | FMR [95% CI] | ALS [95% CI] | PM_NVR | PM_FMR |
|---|---|---|---|---|---|---|
| Unconditional | Stage 1 | 0.0096 [0.0079, 0.0115] | 0.0057 [0.0043, 0.0073] | 10.79 [10.44, 11.16] | 0.272 [0.219, 0.336] | 0.576 [0.445, 0.726] |
| Unconditional | Stage 2 | 0.0051 [0.0040, 0.0063] | 0.0028 [0.0018, 0.0039] | 10.07 [9.80, 10.35] | 0.161 [0.122, 0.212] | 0.373 [0.247, 0.542] |
| Unconditional | Final | 0.0072 [0.0058, 0.0087] | 0.0015 [0.0008, 0.0023] | 10.31 [10.04, 10.61] | 0.218 [0.170, 0.277] | 0.288 [0.158, 0.460] |
| Minimal Cue | Stage 1 | 0.0275 [0.0250, 0.0301] | 0.0062 [0.0047, 0.0078] | 18.92 [18.38, 19.48] | 0.516 [0.457, 0.582] | 0.596 [0.465, 0.742] |
| Minimal Cue | Stage 2 | 0.0272 [0.0247, 0.0298] | 0.0061 [0.0046, 0.0077] | 19.07 [18.54, 19.64] | 0.503 [0.444, 0.571] | 0.565 [0.438, 0.713] |
| Minimal Cue | Final | 0.0286 [0.0260, 0.0313] | 0.0071 [0.0055, 0.0087] | 18.71 [18.15, 19.29] | 0.523 [0.464, 0.591] | 0.657 [0.538, 0.788] |
| Generic | Stage 1 | 0.0057 [0.0048, 0.0068] | 0.0064 [0.0048, 0.0081] | 13.66 [13.28, 14.07] | 0.181 [0.143, 0.230] | 0.604 [0.474, 0.753] |
| Generic | Stage 2 | 0.0051 [0.0043, 0.0059] | 0.0047 [0.0033, 0.0061] | 13.39 [13.08, 13.72] | 0.159 [0.125, 0.203] | 0.500 [0.368, 0.662] |
| Generic | Final | 0.0061 [0.0052, 0.0071] | 0.0059 [0.0044, 0.0075] | 13.49 [13.19, 13.81] | 0.191 [0.151, 0.241] | 0.615 [0.491, 0.755] |
| Specific | Stage 1 | 0.0032 [0.0025, 0.0040] | 0.0075 [0.0054, 0.0099] | 17.66 [16.49, 18.97] | 0.111 [0.082, 0.149] | 0.641 [0.504, 0.782] |
| Specific | Stage 2 | 0.0032 [0.0026, 0.0039] | 0.0057 [0.0039, 0.0078] | 17.08 [16.04, 18.26] | 0.108 [0.082, 0.144] | 0.548 [0.404, 0.705] |
| Specific | Final | 0.0029 [0.0023, 0.0035] | 0.0067 [0.0048, 0.0088] | 17.57 [16.60, 18.60] | 0.101 [0.075, 0.134] | 0.644 [0.512, 0.779] |
| Prefix | Stage 1 | 0.0258 [0.0200, 0.0319] | 0.0042 [0.0022, 0.0066] | 29.90 [28.23, 31.65] | — | — |
| Prefix | Stage 2 | 0.0268 [0.0208, 0.0333] | 0.0047 [0.0026, 0.0072] | 30.32 [28.58, 32.16] | — | — |
| Prefix | Final | 0.0260 [0.0201, 0.0325] | 0.0037 [0.0020, 0.0056] | 30.33 [28.63, 32.08] | — | — |

Significant paired differences from the final checkpoint (5 of 30 stage–setting–metric comparisons):

- Stage 1 − Final, Unconditional, NVR: +0.0024 [+0.0001, +0.0047] ↑
- Stage 1 − Final, Unconditional, FMR: +0.0042 [+0.0026, +0.0059] ↑
- Stage 1 − Final, Unconditional, ALS: +0.48 [+0.02, +0.93] ↑
- Stage 2 − Final, Unconditional, NVR: -0.0021 [-0.0040, -0.0003] ↓
- Stage 2 − Final, Generic, NVR: -0.0011 [-0.0019, -0.0002] ↓

## 5. The paper's claims, re-checked on these results

Statuses follow fixed rules: point estimates for orderings and the 0.5 threshold, and the sign of 95% CIs for differences. *Partly* means some but not all of the stated comparisons hold.

| Claim (paper section) | Status | Evidence |
|---|---|---|
| Prefix yields higher NVR and ALS point estimates than generic and specific prompting, for every model–corpus pair (Abstract, §5.2, §6) | holds | 12/12 comparisons; exceptions: none |
| … also for Olmo 3 on Dolma 3 (new) | holds | 4/4; exceptions: none |
| … also against the new prompt-free settings (unconditional, minimal cue) | partly holds | 7/16; exceptions: Comma CP Unconditional NVR, Comma CP Unconditional ALS, Comma CP Minimal Cue NVR, DFM CP Minimal Cue NVR, DFM DW Unconditional NVR, DFM DW Unconditional ALS, Olmo 3 32B D3 Unconditional NVR, Olmo 3 32B D3 Unconditional ALS, Olmo 3 32B D3 Minimal Cue NVR |
| Propensity scores are generally below the neutral 0.5 (Abstract, §5.2, §6) — generic and specific | partly holds | 14/16 below 0.5; at/above: DFM CP Generic PM_FMR 0.615 [0.491, 0.755]; DFM CP Specific PM_FMR 0.644 [0.512, 0.779] |
| … for the new prompt-free settings | partly holds | 8/16 below 0.5; at/above: Comma CP Unconditional PM_NVR 0.548 [0.490, 0.614]; Comma CP Minimal Cue PM_NVR 0.538 [0.480, 0.603]; DFM CP Minimal Cue PM_NVR 0.523 [0.464, 0.591]; DFM CP Minimal Cue PM_FMR 0.657 [0.538, 0.788]; DFM DW Unconditional PM_NVR 0.508 [0.473, 0.544]; Olmo 3 32B D3 Unconditional PM_NVR 0.745 [0.675, 0.812]; Olmo 3 32B D3 Minimal Cue PM_NVR 0.586 [0.497, 0.679]; Olmo 3 32B D3 Minimal Cue PM_FMR 0.580 [0.397, 0.820] |
| Comma produces longer verbatim spans than DFM on Common Pile under generic and prefix prompting (§5.2) | partly holds | DFM − Comma: ALS Generic -0.39 [-0.84, +0.06] ≈; ALS Prefix -2.15 [-2.98, -1.35] ↓ |
| Comma shows more full-generation memorization of Common Pile than DFM (specific, prefix) (§5.2) | does not hold | DFM − Comma: FMR Specific +0.0048 [+0.0029, +0.0068] ↑; FMR Prefix -0.0019 [-0.0044, +0.0001] ≈ |
| Specific prompts match the prefix FMR for Comma on Common Pile (§5.1) | does not hold | specific − prefix -0.0037 [-0.0072, -0.0009] ↓ |
| From Comma to DFM, specific-prompt PM_FMR decreases while PM_NVR increases (§5.2) | does not hold | PM_FMR Δ +0.391 [+0.242, +0.520] ↑; PM_NVR Δ -0.005 [-0.031, +0.020] ≈ |
| For DFM, Common Pile yields longer ALS than Dynaword in every setting (§5.2) | partly holds | CP − DW: Unconditional -15.94 [-16.46, -15.43] ↓; Minimal Cue -1.59 [-2.22, -0.95] ↓; Generic -2.59 [-3.11, -2.09] ↓; Specific -2.33 [-3.93, -0.78] ↓; Prefix +4.90 [+2.61, +7.15] ↑ |
| For DFM, prefix FMR is higher on Dynaword than on Common Pile (§5.2) | holds | DW − CP +0.0330 [+0.0248, +0.0415] ↑ |
| Memorization is essentially stable across DFM training stages (§5.3) | does not hold | 18/60 stage − final paired differences significant (see Section 4) |


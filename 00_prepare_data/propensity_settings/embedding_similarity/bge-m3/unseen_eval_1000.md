# Embedding similarity: unseen_cp vs tatoeba_eng, unseen_cp vs generic_cp_1000, unseen_d3 vs tatoeba_eng, unseen_d3 vs generic_d3_1000, unseen_dw vs tatoeba_dan, unseen_dw vs generic_dw_1000 to training docs

Model `BAAI/bge-m3`; 10000 random docs per corpus (first 512 tokens embedded); top-k = 10; query instruction: `(none)`.
`train` rows are sentences from other random docs of the same corpus (an in-distribution reference).

## commonpile

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.5478 [0.5459, 0.5496] | 0.5491 | 0.5144 | 0.3296 |
| generic_cp_1000 | 1000 | 0.5436 [0.5413, 0.5461] | 0.5442 | 0.5100 | 0.3246 |
| generic_d3_1000 | 1000 | 0.5351 [0.5326, 0.5375] | 0.5356 | 0.5025 | 0.3179 |
| unseen_cp | 1000 | 0.5530 [0.5507, 0.5553] | 0.5536 | 0.5220 | 0.3442 |
| unseen_d3 | 1000 | 0.5524 [0.5496, 0.5550] | 0.5537 | 0.5211 | 0.3391 |
| train_commonpile | 2000 | 0.5723 [0.5698, 0.5747] | 0.5704 | 0.5380 | 0.3491 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0052 [+0.0022, +0.0082] | 0.535 |
| topk_mean | +0.0076 [+0.0048, +0.0104] | 0.555 |
| mean_sim | +0.0146 [+0.0118, +0.0175] | 0.608 |

**unseen_cp - generic_cp_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0094 [+0.0057, +0.0128] | 0.567 |
| topk_mean | +0.0120 [+0.0089, +0.0150] | 0.591 |
| mean_sim | +0.0196 [+0.0165, +0.0230] | 0.647 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0046 [+0.0015, +0.0079] | 0.532 |
| topk_mean | +0.0067 [+0.0039, +0.0095] | 0.550 |
| mean_sim | +0.0096 [+0.0067, +0.0124] | 0.575 |

**unseen_d3 - generic_d3_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0173 [+0.0137, +0.0208] | 0.620 |
| topk_mean | +0.0186 [+0.0153, +0.0219] | 0.642 |
| mean_sim | +0.0213 [+0.0179, +0.0246] | 0.662 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.5549 (1246) | 0.5356 (665) | 0.5369 (54) | 0.5415 (22) | 0.5425 (13) |
| generic_cp_1000 | 0.5504 (552) | 0.5344 (394) | 0.5410 (32) | 0.5418 (13) | 0.5456 (9) |
| generic_d3_1000 | 0.5404 (515) | 0.5295 (446) | 0.5240 (25) | 0.5328 (12) | 0.5487 (2) |
| unseen_cp | 0.5616 (443) | 0.5467 (475) | 0.5379 (56) | 0.5502 (13) | 0.5556 (13) |
| unseen_d3 | 0.5571 (392) | 0.5493 (549) | 0.5437 (42) | 0.5635 (12) | 0.5573 (5) |
| train_commonpile | 0.5795 (299) | 0.5740 (683) | 0.5636 (414) | 0.5663 (326) | 0.5800 (278) |

generic_cp_1000 max_sim by domain (domain or source): tatoeba 0.5436 (n=1000)

generic_d3_1000 max_sim by domain (domain or source): tatoeba 0.5351 (n=1000)

unseen_cp max_sim by domain (domain or source): Eurlex 0.5343 (n=109), Eurovoc 0.5581 (n=109), GATT_library 0.5424 (n=16), NewZealand-PD-Newspapers 0.5535 (n=109), SEC 0.5464 (n=108), TEDEUTenders 0.6177 (n=2), US-PD-Newspapers 0.5497 (n=109), VoxPopuli 0.5514 (n=109), WTO 0.5535 (n=110), dotgov 0.5652 (n=109), wikipedia_new 0.5650 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.5565 (n=333), hplt 0.5600 (n=334), wikipedia_new 0.5406 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: wikiteam 52.5%/11.5%=4.58x; wikimedia 11.9%/6.9%=1.71x; biodiversity_heritage_library 8.3%/6.2%=1.35x; pre_1929_books 4.3%/0.2%=26.87x; data_provenance_initiative 4.1%/1.6%=2.52x; youtube 3.7%/0.5%=7.71x; stackv2_edu 2.1%/28.7%=0.07x; github_archive 2.0%/10.1%=0.2x
- generic_cp_1000: wikiteam 52.5%/11.5%=4.58x; wikimedia 12.4%/6.9%=1.79x; biodiversity_heritage_library 8.9%/6.2%=1.44x; data_provenance_initiative 4.3%/1.6%=2.64x; pre_1929_books 4.0%/0.2%=25.0x; youtube 3.9%/0.5%=8.12x; stackv2_edu 2.2%/28.7%=0.08x; uspto 1.8%/7.3%=0.25x
- generic_d3_1000: wikiteam 52.5%/11.5%=4.58x; wikimedia 16.1%/6.9%=2.32x; biodiversity_heritage_library 8.0%/6.2%=1.29x; youtube 3.8%/0.5%=7.92x; data_provenance_initiative 3.2%/1.6%=1.96x; pre_1929_books 3.2%/0.2%=20.0x; stackv2_edu 1.9%/28.7%=0.07x; uspto 1.6%/7.3%=0.22x
- unseen_cp: wikiteam 21.7%/11.5%=1.89x; biodiversity_heritage_library 21.6%/6.2%=3.49x; usgpo 7.1%/0.9%=7.98x; wikimedia 5.9%/6.9%=0.85x; github_archive 5.0%/10.1%=0.49x; uspto 4.8%/7.3%=0.66x; caselaw_access_project 4.1%/2.6%=1.6x; stackv2_edu 3.8%/28.7%=0.13x
- unseen_d3: wikiteam 24.1%/11.5%=2.1x; biodiversity_heritage_library 22.3%/6.2%=3.6x; wikimedia 6.6%/6.9%=0.95x; uspto 6.4%/7.3%=0.88x; stackv2_edu 6.4%/28.7%=0.22x; cccc 5.0%/2.8%=1.77x; pubmed 4.4%/1.5%=2.89x; github_archive 3.4%/10.1%=0.34x
- train_commonpile: wikiteam 23.8%/11.5%=2.08x; stackv2_edu 12.3%/28.7%=0.43x; biodiversity_heritage_library 11.8%/6.2%=1.91x; github_archive 11.7%/10.1%=1.15x; uspto 10.9%/7.3%=1.5x; stackexchange 5.1%/13.7%=0.37x; wikimedia 5.0%/6.9%=0.71x; caselaw_access_project 3.5%/2.6%=1.34x

## dolma3

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.5436 [0.5418, 0.5453] | 0.5441 | 0.5134 | 0.3329 |
| generic_cp_1000 | 1000 | 0.5401 [0.5376, 0.5425] | 0.5391 | 0.5102 | 0.3277 |
| generic_d3_1000 | 1000 | 0.5316 [0.5294, 0.5340] | 0.5298 | 0.5020 | 0.3205 |
| unseen_cp | 1000 | 0.5453 [0.5427, 0.5476] | 0.5472 | 0.5146 | 0.3411 |
| unseen_d3 | 1000 | 0.5561 [0.5532, 0.5587] | 0.5569 | 0.5235 | 0.3397 |
| train_dolma3 | 2000 | 0.5626 [0.5606, 0.5648] | 0.5609 | 0.5277 | 0.3373 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0017 [-0.0013, +0.0047] | 0.518 |
| topk_mean | +0.0012 [-0.0015, +0.0039] | 0.515 |
| mean_sim | +0.0083 [+0.0055, +0.0110] | 0.564 |

**unseen_cp - generic_cp_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0052 [+0.0017, +0.0088] | 0.546 |
| topk_mean | +0.0044 [+0.0013, +0.0075] | 0.543 |
| mean_sim | +0.0134 [+0.0101, +0.0167] | 0.606 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0125 [+0.0093, +0.0157] | 0.587 |
| topk_mean | +0.0101 [+0.0073, +0.0129] | 0.584 |
| mean_sim | +0.0069 [+0.0041, +0.0098] | 0.557 |

**unseen_d3 - generic_d3_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0244 [+0.0208, +0.0280] | 0.670 |
| topk_mean | +0.0215 [+0.0183, +0.0246] | 0.674 |
| mean_sim | +0.0193 [+0.0161, +0.0224] | 0.650 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.5492 (1246) | 0.5326 (665) | 0.5436 (54) | 0.5433 (22) | 0.5659 (13) |
| generic_cp_1000 | 0.5452 (552) | 0.5308 (394) | 0.5524 (32) | 0.5406 (13) | 0.5824 (9) |
| generic_d3_1000 | 0.5366 (515) | 0.5253 (446) | 0.5390 (25) | 0.5350 (12) | 0.5592 (2) |
| unseen_cp | 0.5535 (443) | 0.5393 (475) | 0.5239 (56) | 0.5614 (13) | 0.5591 (13) |
| unseen_d3 | 0.5584 (392) | 0.5547 (549) | 0.5494 (42) | 0.5492 (12) | 0.5931 (5) |
| train_dolma3 | 0.5545 (177) | 0.5599 (545) | 0.5643 (564) | 0.5674 (477) | 0.5614 (237) |

generic_cp_1000 max_sim by domain (domain or source): tatoeba 0.5401 (n=1000)

generic_d3_1000 max_sim by domain (domain or source): tatoeba 0.5316 (n=1000)

unseen_cp max_sim by domain (domain or source): Eurlex 0.5357 (n=109), Eurovoc 0.5535 (n=109), GATT_library 0.5408 (n=16), NewZealand-PD-Newspapers 0.5432 (n=109), SEC 0.5489 (n=108), TEDEUTenders 0.6111 (n=2), US-PD-Newspapers 0.5391 (n=109), VoxPopuli 0.5619 (n=109), WTO 0.5361 (n=110), dotgov 0.5543 (n=109), wikipedia_new 0.5348 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.5646 (n=333), hplt 0.5711 (n=334), wikipedia_new 0.5325 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: common_crawl 98.0%/80.7%=1.21x; stack_edu 1.0%/14.4%=0.07x; finemath 0.5%/2.7%=0.19x; olmocr_science_pdfs 0.3%/1.8%=0.16x; rpj 0.1%/0.2%=0.42x; dolma1_7 0.1%/0.1%=0.5x
- generic_cp_1000: common_crawl 98.1%/80.7%=1.22x; stack_edu 0.9%/14.4%=0.06x; finemath 0.4%/2.7%=0.15x; olmocr_science_pdfs 0.4%/1.8%=0.22x; rpj 0.1%/0.2%=0.42x; dolma1_7 0.1%/0.1%=1.0x
- generic_d3_1000: common_crawl 98.6%/80.7%=1.22x; stack_edu 1.0%/14.4%=0.07x; finemath 0.2%/2.7%=0.07x; olmocr_science_pdfs 0.2%/1.8%=0.11x
- unseen_cp: common_crawl 87.4%/80.7%=1.08x; stack_edu 4.1%/14.4%=0.28x; rpj 3.3%/0.2%=13.75x; olmocr_science_pdfs 3.0%/1.8%=1.62x; finemath 1.9%/2.7%=0.71x; dolma1_7 0.3%/0.1%=3.0x
- unseen_d3: common_crawl 91.2%/80.7%=1.13x; stack_edu 3.8%/14.4%=0.26x; olmocr_science_pdfs 2.1%/1.8%=1.14x; finemath 2.0%/2.7%=0.75x; dolma1_7 0.5%/0.1%=5.0x; rpj 0.4%/0.2%=1.67x
- train_dolma3: common_crawl 90.5%/80.7%=1.12x; stack_edu 5.2%/14.4%=0.36x; olmocr_science_pdfs 2.1%/1.8%=1.14x; finemath 1.7%/2.7%=0.63x; rpj 0.4%/0.2%=1.88x; dolma1_7 0.1%/0.1%=1.0x

Nearest-doc subsource (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: entertainment 25.1%/7.8%=3.22x; literature 22.1%/3.3%=6.74x; health 7.5%/8.3%=0.9x; science_math_and_technology 6.2%/14.4%=0.43x; crime_and_law 5.0%/3.0%=1.65x; games 4.0%/4.9%=0.81x; religion 3.6%/1.4%=2.61x; politics 3.1%/1.6%=1.93x
- generic_cp_1000: entertainment 24.6%/7.8%=3.15x; literature 22.4%/3.3%=6.83x; health 7.2%/8.3%=0.86x; science_math_and_technology 6.0%/14.4%=0.42x; crime_and_law 5.2%/3.0%=1.73x; games 3.8%/4.9%=0.77x; religion 3.5%/1.4%=2.5x; politics 3.5%/1.6%=2.15x
- generic_d3_1000: entertainment 29.2%/7.8%=3.74x; literature 23.0%/3.3%=7.01x; health 6.0%/8.3%=0.72x; crime_and_law 5.4%/3.0%=1.8x; science_math_and_technology 4.7%/14.4%=0.33x; religion 3.7%/1.4%=2.64x; education_and_jobs 3.3%/4.5%=0.74x; politics 3.3%/1.6%=2.02x
- unseen_cp: science_math_and_technology 16.7%/14.4%=1.16x; politics 11.1%/1.6%=6.81x; crime_and_law 8.5%/3.0%=2.83x; finance_and_business 7.9%/4.0%=1.98x; literature 6.1%/3.3%=1.86x; health 5.7%/8.3%=0.68x; software 4.3%/5.3%=0.81x; entertainment 4.3%/7.8%=0.55x
- unseen_d3: science_math_and_technology 19.4%/14.4%=1.35x; health 9.8%/8.3%=1.18x; literature 8.7%/3.3%=2.65x; education_and_jobs 8.4%/4.5%=1.88x; entertainment 5.9%/7.8%=0.76x; finance_and_business 4.0%/4.0%=1.0x; history_and_geography 3.6%/1.8%=2.05x; games 3.6%/4.9%=0.73x
- train_dolma3: science_math_and_technology 16.6%/14.4%=1.16x; health 10.7%/8.3%=1.28x; entertainment 8.3%/7.8%=1.07x; software_development 6.9%/9.3%=0.74x; literature 6.1%/3.3%=1.86x; software 6.0%/5.3%=1.15x; games 5.9%/4.9%=1.2x; finance_and_business 5.3%/4.0%=1.34x

## dynaword1212

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_dan | 2000 | 0.5309 [0.5290, 0.5327] | 0.5298 | 0.4974 | 0.3235 |
| generic_dw_1000 | 1000 | 0.5273 [0.5249, 0.5300] | 0.5263 | 0.4935 | 0.3185 |
| unseen_dw | 1000 | 0.5398 [0.5371, 0.5425] | 0.5410 | 0.5073 | 0.3354 |
| train_dynaword1212 | 2000 | 0.6065 [0.6034, 0.6101] | 0.5950 | 0.5690 | 0.3617 |

**unseen_dw - tatoeba_dan**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0089 [+0.0056, +0.0121] | 0.564 |
| topk_mean | +0.0099 [+0.0068, +0.0129] | 0.580 |
| mean_sim | +0.0119 [+0.0088, +0.0151] | 0.595 |

**unseen_dw - generic_dw_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0124 [+0.0085, +0.0162] | 0.588 |
| topk_mean | +0.0138 [+0.0105, +0.0173] | 0.608 |
| mean_sim | +0.0169 [+0.0133, +0.0204] | 0.628 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_dan | 0.5353 (870) | 0.5294 (982) | 0.5141 (109) | 0.5091 (30) | 0.5346 (9) |
| generic_dw_1000 | 0.5300 (362) | 0.5277 (556) | 0.5132 (60) | 0.5060 (17) | 0.5411 (5) |
| unseen_dw | 0.5464 (248) | 0.5382 (598) | 0.5312 (118) | 0.5424 (26) | 0.5610 (10) |
| train_dynaword1212 | 0.5642 (56) | 0.5788 (257) | 0.5809 (287) | 0.5952 (429) | 0.6289 (971) |

generic_dw_1000 max_sim by domain (domain or source): tatoeba 0.5273 (n=1000)

unseen_dw max_sim by domain (domain or source): dakultur 0.5009 (n=70), depbank 0.5370 (n=69), folketingets-dokumenter 0.5618 (n=70), hvadvilduhelst 0.4823 (n=68), jvj 0.5244 (n=11), kalliope 0.5302 (n=69), kb_administrative_publication 0.5575 (n=69), kb_historical_letters 0.5460 (n=69), logir 0.5633 (n=69), mosel_voxpopuli 0.5538 (n=70), mosel_youtubecommons 0.5376 (n=70), municipality_meetings 0.5815 (n=70), nordjyllandnews 0.5475 (n=70), synne 0.5291 (n=18), tidsskrift-dk 0.5381 (n=69), wikipedia_new 0.5229 (n=69)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_dan: enevaeldens_nyheder 45.9%/83.0%=0.55x; opensubtitles 18.6%/0.6%=31.08x; dannet 16.4%/0.3%=54.5x; hest 3.1%/0.3%=11.92x; ai-aktindsigt 2.7%/3.7%=0.74x; wiki 2.5%/5.7%=0.45x; ncc_maalfrid 2.4%/0.6%=3.87x; ncc_newspaper 2.1%/0.1%=19.09x
- generic_dw_1000: enevaeldens_nyheder 46.1%/83.0%=0.56x; opensubtitles 18.3%/0.6%=30.5x; dannet 17.1%/0.3%=57.0x; hest 3.3%/0.3%=12.69x; wiki 2.9%/5.7%=0.51x; ncc_maalfrid 2.2%/0.6%=3.55x; ai-aktindsigt 1.9%/3.7%=0.52x; ncc_newspaper 1.9%/0.1%=17.27x
- unseen_dw: enevaeldens_nyheder 54.5%/83.0%=0.66x; ai-aktindsigt 11.8%/3.7%=3.22x; retsinformationdk 5.8%/1.9%=3.05x; dannet 5.1%/0.3%=17.0x; opensubtitles 4.3%/0.6%=7.17x; wiki 3.6%/5.7%=0.63x; danske-taler 3.2%/0.1%=35.56x; ncc_maalfrid 1.7%/0.6%=2.74x
- train_dynaword1212: enevaeldens_nyheder 88.8%/83.0%=1.07x; ai-aktindsigt 3.5%/3.7%=0.94x; wiki 2.2%/5.7%=0.39x; retsinformationdk 1.4%/1.9%=0.74x; cellar 0.7%/1.1%=0.58x; dannet 0.6%/0.3%=2.0x; tv2r 0.6%/1.1%=0.55x; health_hovedstaden 0.5%/0.6%=0.95x

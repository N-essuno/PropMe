# Embedding similarity: unseen_cp vs tatoeba_eng, unseen_cp vs generic_cp, unseen_d3 vs tatoeba_eng, unseen_d3 vs generic_d3, unseen_dw vs tatoeba_dan, unseen_dw vs generic_dw to training docs

Model `Qwen/Qwen3-Embedding-8B`; 10000 random docs per corpus (first 512 tokens embedded); top-k = 10; query instruction: `Given a sentence, retrieve documents that match its topic, genre and writing style`.
`train` rows are sentences from other random docs of the same corpus (an in-distribution reference).

## commonpile

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.4614 [0.4595, 0.4635] | 0.4587 | 0.4097 | 0.1578 |
| generic_cp | 1642 | 0.4627 [0.4606, 0.4648] | 0.4604 | 0.4108 | 0.1575 |
| generic_d3 | 1174 | 0.4635 [0.4610, 0.4659] | 0.4612 | 0.4117 | 0.1579 |
| unseen_cp | 1000 | 0.4852 [0.4822, 0.4882] | 0.4808 | 0.4325 | 0.1764 |
| unseen_d3 | 1000 | 0.4748 [0.4712, 0.4783] | 0.4691 | 0.4182 | 0.1634 |
| train_commonpile | 2000 | 0.5165 [0.5132, 0.5196] | 0.5128 | 0.4601 | 0.1873 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0238 [+0.0200, +0.0274] | 0.641 |
| topk_mean | +0.0228 [+0.0201, +0.0255] | 0.676 |
| mean_sim | +0.0186 [+0.0168, +0.0205] | 0.717 |

**unseen_cp - generic_cp**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0225 [+0.0187, +0.0262] | 0.632 |
| topk_mean | +0.0218 [+0.0190, +0.0247] | 0.666 |
| mean_sim | +0.0189 [+0.0171, +0.0208] | 0.721 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0133 [+0.0097, +0.0172] | 0.567 |
| topk_mean | +0.0085 [+0.0056, +0.0114] | 0.547 |
| mean_sim | +0.0055 [+0.0037, +0.0074] | 0.556 |

**unseen_d3 - generic_d3**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0113 [+0.0068, +0.0153] | 0.554 |
| topk_mean | +0.0065 [+0.0033, +0.0097] | 0.530 |
| mean_sim | +0.0055 [+0.0036, +0.0074] | 0.556 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.4595 (1246) | 0.4653 (665) | 0.4643 (54) | 0.4599 (22) | 0.4418 (13) |
| generic_cp | 0.4613 (919) | 0.4650 (640) | 0.4651 (52) | 0.4601 (21) | 0.4432 (10) |
| generic_d3 | 0.4620 (603) | 0.4657 (522) | 0.4616 (30) | 0.4594 (14) | 0.4293 (5) |
| unseen_cp | 0.4832 (443) | 0.4879 (475) | 0.4879 (56) | 0.4682 (13) | 0.4620 (13) |
| unseen_d3 | 0.4750 (392) | 0.4751 (549) | 0.4673 (42) | 0.4772 (12) | 0.4763 (5) |
| train_commonpile | 0.5000 (299) | 0.5211 (683) | 0.5167 (414) | 0.5145 (326) | 0.5249 (278) |

generic_cp max_sim by domain (domain or source): tatoeba 0.4627 (n=1642)

generic_d3 max_sim by domain (domain or source): tatoeba 0.4635 (n=1174)

unseen_cp max_sim by domain (domain or source): Eurlex 0.4795 (n=109), Eurovoc 0.4838 (n=109), GATT_library 0.4633 (n=16), NewZealand-PD-Newspapers 0.5068 (n=109), SEC 0.4602 (n=108), TEDEUTenders 0.4884 (n=2), US-PD-Newspapers 0.4886 (n=109), VoxPopuli 0.4767 (n=109), WTO 0.4776 (n=110), dotgov 0.4892 (n=109), wikipedia_new 0.5073 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.4761 (n=333), hplt 0.4705 (n=334), wikipedia_new 0.4777 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: wikiteam 38.0%/11.5%=3.31x; wikimedia 20.6%/6.9%=2.98x; biodiversity_heritage_library 10.7%/6.2%=1.73x; pre_1929_books 5.8%/0.2%=36.25x; youtube 5.6%/0.5%=11.67x; data_provenance_initiative 5.1%/1.6%=3.16x; cccc 3.0%/2.8%=1.08x; library_of_congress 2.5%/0.2%=15.31x
- generic_cp: wikiteam 33.9%/11.5%=2.95x; wikimedia 23.7%/6.9%=3.41x; biodiversity_heritage_library 11.9%/6.2%=1.92x; youtube 6.3%/0.5%=13.2x; pre_1929_books 5.9%/0.2%=36.54x; data_provenance_initiative 4.8%/1.6%=2.95x; cccc 3.1%/2.8%=1.1x; library_of_congress 2.4%/0.2%=14.84x
- generic_d3: wikiteam 30.4%/11.5%=2.65x; wikimedia 28.7%/6.9%=4.14x; biodiversity_heritage_library 11.3%/6.2%=1.83x; youtube 6.6%/0.5%=13.84x; pre_1929_books 6.0%/0.2%=37.8x; data_provenance_initiative 4.5%/1.6%=2.77x; cccc 3.1%/2.8%=1.09x; library_of_congress 2.4%/0.2%=14.91x
- unseen_cp: biodiversity_heritage_library 26.4%/6.2%=4.26x; wikiteam 13.9%/11.5%=1.21x; usgpo 8.8%/0.9%=9.89x; wikimedia 8.5%/6.9%=1.22x; caselaw_access_project 7.8%/2.6%=3.04x; data_provenance_initiative 4.6%/1.6%=2.82x; youtube 4.0%/0.5%=8.33x; cccc 3.6%/2.8%=1.28x
- unseen_d3: biodiversity_heritage_library 22.2%/6.2%=3.59x; wikiteam 17.8%/11.5%=1.55x; wikimedia 12.4%/6.9%=1.79x; stackv2_edu 5.3%/28.7%=0.18x; cccc 5.2%/2.8%=1.84x; uspto 4.2%/7.3%=0.58x; youtube 3.9%/0.5%=8.12x; data_provenance_initiative 3.7%/1.6%=2.27x
- train_commonpile: wikiteam 19.6%/11.5%=1.7x; stackv2_edu 13.2%/28.7%=0.46x; biodiversity_heritage_library 12.8%/6.2%=2.08x; github_archive 11.1%/10.1%=1.09x; uspto 9.2%/7.3%=1.27x; wikimedia 6.2%/6.9%=0.89x; stackexchange 5.5%/13.7%=0.4x; caselaw_access_project 4.2%/2.6%=1.65x

## dolma3

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.4629 [0.4609, 0.4647] | 0.4588 | 0.4088 | 0.1532 |
| generic_cp | 1642 | 0.4653 [0.4633, 0.4673] | 0.4616 | 0.4099 | 0.1526 |
| generic_d3 | 1174 | 0.4649 [0.4625, 0.4674] | 0.4619 | 0.4098 | 0.1528 |
| unseen_cp | 1000 | 0.4673 [0.4644, 0.4702] | 0.4616 | 0.4147 | 0.1662 |
| unseen_d3 | 1000 | 0.4852 [0.4819, 0.4886] | 0.4804 | 0.4226 | 0.1578 |
| train_dolma3 | 2000 | 0.4893 [0.4864, 0.4919] | 0.4826 | 0.4263 | 0.1578 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0044 [+0.0010, +0.0079] | 0.525 |
| topk_mean | +0.0059 [+0.0038, +0.0081] | 0.555 |
| mean_sim | +0.0130 [+0.0115, +0.0145] | 0.684 |

**unseen_cp - generic_cp**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0020 [-0.0014, +0.0055] | 0.508 |
| topk_mean | +0.0048 [+0.0024, +0.0071] | 0.543 |
| mean_sim | +0.0135 [+0.0121, +0.0151] | 0.692 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0223 [+0.0185, +0.0260] | 0.628 |
| topk_mean | +0.0139 [+0.0115, +0.0163] | 0.624 |
| mean_sim | +0.0046 [+0.0031, +0.0061] | 0.565 |

**unseen_d3 - generic_d3**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0203 [+0.0161, +0.0242] | 0.614 |
| topk_mean | +0.0129 [+0.0104, +0.0157] | 0.615 |
| mean_sim | +0.0050 [+0.0033, +0.0066] | 0.571 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.4600 (1246) | 0.4672 (665) | 0.4702 (54) | 0.4776 (22) | 0.4747 (13) |
| generic_cp | 0.4632 (919) | 0.4672 (640) | 0.4704 (52) | 0.4780 (21) | 0.4882 (10) |
| generic_d3 | 0.4622 (603) | 0.4678 (522) | 0.4661 (30) | 0.4792 (14) | 0.4440 (5) |
| unseen_cp | 0.4710 (443) | 0.4655 (475) | 0.4557 (56) | 0.4794 (13) | 0.4470 (13) |
| unseen_d3 | 0.4850 (392) | 0.4854 (549) | 0.4821 (42) | 0.4549 (12) | 0.5842 (5) |
| train_dolma3 | 0.4680 (177) | 0.4956 (545) | 0.4953 (564) | 0.4889 (477) | 0.4770 (237) |

generic_cp max_sim by domain (domain or source): tatoeba 0.4653 (n=1642)

generic_d3 max_sim by domain (domain or source): tatoeba 0.4649 (n=1174)

unseen_cp max_sim by domain (domain or source): Eurlex 0.4721 (n=109), Eurovoc 0.4779 (n=109), GATT_library 0.4553 (n=16), NewZealand-PD-Newspapers 0.4665 (n=109), SEC 0.4614 (n=108), TEDEUTenders 0.4828 (n=2), US-PD-Newspapers 0.4676 (n=109), VoxPopuli 0.4799 (n=109), WTO 0.4535 (n=110), dotgov 0.4794 (n=109), wikipedia_new 0.4495 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.4970 (n=333), hplt 0.4939 (n=334), wikipedia_new 0.4646 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: common_crawl 98.0%/80.7%=1.21x; stack_edu 0.9%/14.4%=0.06x; olmocr_science_pdfs 0.8%/1.8%=0.41x; finemath 0.4%/2.7%=0.15x; dolma1_7 0.1%/0.1%=0.5x
- generic_cp: common_crawl 98.0%/80.7%=1.21x; stack_edu 0.9%/14.4%=0.06x; olmocr_science_pdfs 0.6%/1.8%=0.33x; finemath 0.4%/2.7%=0.16x; dolma1_7 0.1%/0.1%=0.61x
- generic_d3: common_crawl 98.0%/80.7%=1.21x; stack_edu 1.0%/14.4%=0.07x; olmocr_science_pdfs 0.6%/1.8%=0.32x; finemath 0.3%/2.7%=0.13x
- unseen_cp: common_crawl 89.7%/80.7%=1.11x; olmocr_science_pdfs 6.5%/1.8%=3.51x; stack_edu 1.9%/14.4%=0.13x; finemath 1.4%/2.7%=0.52x; rpj 0.3%/0.2%=1.25x; dolma1_7 0.2%/0.1%=2.0x
- unseen_d3: common_crawl 92.8%/80.7%=1.15x; olmocr_science_pdfs 3.5%/1.8%=1.89x; stack_edu 2.0%/14.4%=0.14x; finemath 1.5%/2.7%=0.56x; dolma1_7 0.2%/0.1%=2.0x
- train_dolma3: common_crawl 91.8%/80.7%=1.14x; stack_edu 3.4%/14.4%=0.23x; olmocr_science_pdfs 2.6%/1.8%=1.43x; finemath 2.0%/2.7%=0.75x; rpj 0.1%/0.2%=0.63x

Nearest-doc subsource (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: literature 47.0%/3.3%=14.34x; entertainment 14.1%/7.8%=1.8x; education_and_jobs 4.3%/4.5%=0.98x; science_math_and_technology 4.2%/14.4%=0.29x; health 4.0%/8.3%=0.48x; games 2.9%/4.9%=0.59x; crime_and_law 2.7%/3.0%=0.9x; religion 2.4%/1.4%=1.68x
- generic_cp: literature 48.0%/3.3%=14.65x; entertainment 13.0%/7.8%=1.66x; science_math_and_technology 4.5%/14.4%=0.31x; health 3.8%/8.3%=0.46x; education_and_jobs 3.7%/4.5%=0.83x; crime_and_law 2.9%/3.0%=0.97x; games 2.7%/4.9%=0.55x; food_and_dining 2.4%/1.9%=1.28x
- generic_d3: literature 51.0%/3.3%=15.56x; entertainment 12.8%/7.8%=1.64x; health 4.1%/8.3%=0.49x; crime_and_law 3.4%/3.0%=1.14x; science_math_and_technology 3.3%/14.4%=0.23x; education_and_jobs 3.2%/4.5%=0.73x; games 2.6%/4.9%=0.52x; religion 2.4%/1.4%=1.7x
- unseen_cp: science_math_and_technology 19.0%/14.4%=1.32x; literature 11.5%/3.3%=3.51x; crime_and_law 9.6%/3.0%=3.2x; politics 9.0%/1.6%=5.52x; finance_and_business 8.8%/4.0%=2.2x; health 5.2%/8.3%=0.62x; software 4.7%/5.3%=0.89x; entertainment 4.5%/7.8%=0.58x
- unseen_d3: science_math_and_technology 20.9%/14.4%=1.45x; education_and_jobs 9.2%/4.5%=2.06x; health 9.1%/8.3%=1.09x; literature 6.6%/3.3%=2.01x; entertainment 5.7%/7.8%=0.73x; history_and_geography 4.5%/1.8%=2.56x; software 4.3%/5.3%=0.81x; games 4.3%/4.9%=0.88x
- train_dolma3: science_math_and_technology 15.4%/14.4%=1.07x; software_development 9.0%/9.3%=0.97x; health 8.9%/8.3%=1.07x; entertainment 8.8%/7.8%=1.13x; literature 7.8%/3.3%=2.38x; education_and_jobs 6.7%/4.5%=1.5x; software 6.3%/5.3%=1.2x; games 4.9%/4.9%=1.0x

## dynaword1212

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_dan | 2000 | 0.4527 [0.4507, 0.4547] | 0.4482 | 0.4094 | 0.2133 |
| generic_dw | 1664 | 0.4560 [0.4536, 0.4582] | 0.4517 | 0.4123 | 0.2136 |
| unseen_dw | 1000 | 0.4985 [0.4952, 0.5020] | 0.4970 | 0.4514 | 0.2319 |
| train_dynaword1212 | 2000 | 0.5732 [0.5697, 0.5766] | 0.5705 | 0.5290 | 0.2791 |

**unseen_dw - tatoeba_dan**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0459 [+0.0419, +0.0500] | 0.740 |
| topk_mean | +0.0420 [+0.0387, +0.0453] | 0.757 |
| mean_sim | +0.0186 [+0.0162, +0.0209] | 0.676 |

**unseen_dw - generic_dw**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0426 [+0.0383, +0.0466] | 0.725 |
| topk_mean | +0.0391 [+0.0356, +0.0426] | 0.740 |
| mean_sim | +0.0183 [+0.0159, +0.0209] | 0.671 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_dan | 0.4418 (870) | 0.4597 (982) | 0.4678 (109) | 0.4699 (30) | 0.5050 (9) |
| generic_dw | 0.4458 (590) | 0.4601 (926) | 0.4678 (109) | 0.4699 (30) | 0.5050 (9) |
| unseen_dw | 0.4791 (248) | 0.5026 (598) | 0.5136 (118) | 0.5147 (26) | 0.5201 (10) |
| train_dynaword1212 | 0.4883 (56) | 0.5311 (257) | 0.5435 (287) | 0.5673 (429) | 0.6007 (971) |

generic_dw max_sim by domain (domain or source): tatoeba 0.4560 (n=1664)

unseen_dw max_sim by domain (domain or source): dakultur 0.4938 (n=70), depbank 0.4902 (n=69), folketingets-dokumenter 0.5184 (n=70), hvadvilduhelst 0.4457 (n=68), jvj 0.4410 (n=11), kalliope 0.4866 (n=69), kb_administrative_publication 0.5106 (n=69), kb_historical_letters 0.5176 (n=69), logir 0.5214 (n=69), mosel_voxpopuli 0.5061 (n=70), mosel_youtubecommons 0.4543 (n=70), municipality_meetings 0.5673 (n=70), nordjyllandnews 0.5104 (n=70), synne 0.4701 (n=18), tidsskrift-dk 0.4753 (n=69), wikipedia_new 0.4970 (n=69)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_dan: enevaeldens_nyheder 48.2%/83.0%=0.58x; opensubtitles 20.4%/0.6%=34.0x; wiki 12.2%/5.7%=2.14x; hest 3.5%/0.3%=13.65x; tv2r 2.1%/1.1%=1.93x; ai-aktindsigt 1.9%/3.7%=0.52x; dannet 1.6%/0.3%=5.33x; wikisource 1.5%/0.1%=30.0x
- generic_dw: enevaeldens_nyheder 49.0%/83.0%=0.59x; opensubtitles 19.5%/0.6%=32.45x; wiki 13.0%/5.7%=2.29x; hest 3.4%/0.3%=12.94x; tv2r 2.2%/1.1%=1.98x; ai-aktindsigt 1.7%/3.7%=0.47x; wikisource 1.5%/0.1%=30.05x; ncc_books 1.4%/0.1%=14.42x
- unseen_dw: enevaeldens_nyheder 44.3%/83.0%=0.53x; wiki 13.0%/5.7%=2.28x; ai-aktindsigt 9.7%/3.7%=2.64x; retsinformationdk 9.3%/1.9%=4.89x; opensubtitles 4.2%/0.6%=7.0x; cellar 3.9%/1.1%=3.45x; tv2r 3.2%/1.1%=2.94x; ncc_books 3.2%/0.1%=32.0x
- train_dynaword1212: enevaeldens_nyheder 84.7%/83.0%=1.02x; wiki 5.4%/5.7%=0.95x; ai-aktindsigt 2.9%/3.7%=0.8x; retsinformationdk 2.1%/1.9%=1.08x; cellar 1.1%/1.1%=0.93x; tv2r 0.8%/1.1%=0.73x; opensubtitles 0.5%/0.6%=0.83x; ncc_maalfrid 0.5%/0.6%=0.81x

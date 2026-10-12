# Embedding similarity: unseen_cp vs tatoeba_eng, unseen_cp vs generic_cp_1000, unseen_d3 vs tatoeba_eng, unseen_d3 vs generic_d3_1000, unseen_dw vs tatoeba_dan, unseen_dw vs generic_dw_1000 to training docs

Model `Qwen/Qwen3-Embedding-8B`; 10000 random docs per corpus (first 512 tokens embedded); top-k = 10; query instruction: `Given a sentence, retrieve documents that match its topic, genre and writing style`.
`train` rows are sentences from other random docs of the same corpus (an in-distribution reference).

## commonpile

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.4614 [0.4595, 0.4635] | 0.4587 | 0.4097 | 0.1578 |
| generic_cp_1000 | 1000 | 0.4628 [0.4600, 0.4658] | 0.4599 | 0.4106 | 0.1573 |
| generic_d3_1000 | 1000 | 0.4632 [0.4605, 0.4661] | 0.4609 | 0.4118 | 0.1581 |
| unseen_cp | 1000 | 0.4852 [0.4822, 0.4881] | 0.4808 | 0.4325 | 0.1764 |
| unseen_d3 | 1000 | 0.4748 [0.4714, 0.4781] | 0.4691 | 0.4182 | 0.1634 |
| train_commonpile | 2000 | 0.5165 [0.5133, 0.5198] | 0.5128 | 0.4601 | 0.1873 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0238 [+0.0202, +0.0273] | 0.641 |
| topk_mean | +0.0228 [+0.0200, +0.0255] | 0.676 |
| mean_sim | +0.0186 [+0.0167, +0.0203] | 0.717 |

**unseen_cp - generic_cp_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0224 [+0.0184, +0.0266] | 0.631 |
| topk_mean | +0.0220 [+0.0189, +0.0250] | 0.667 |
| mean_sim | +0.0191 [+0.0171, +0.0210] | 0.723 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0133 [+0.0095, +0.0173] | 0.567 |
| topk_mean | +0.0085 [+0.0055, +0.0114] | 0.547 |
| mean_sim | +0.0055 [+0.0037, +0.0074] | 0.556 |

**unseen_d3 - generic_d3_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0115 [+0.0072, +0.0157] | 0.555 |
| topk_mean | +0.0064 [+0.0033, +0.0097] | 0.529 |
| mean_sim | +0.0053 [+0.0033, +0.0072] | 0.553 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.4595 (1246) | 0.4653 (665) | 0.4643 (54) | 0.4599 (22) | 0.4418 (13) |
| generic_cp_1000 | 0.4619 (552) | 0.4651 (394) | 0.4614 (32) | 0.4563 (13) | 0.4327 (9) |
| generic_d3_1000 | 0.4621 (515) | 0.4654 (446) | 0.4600 (25) | 0.4469 (12) | 0.4150 (2) |
| unseen_cp | 0.4832 (443) | 0.4879 (475) | 0.4879 (56) | 0.4682 (13) | 0.4620 (13) |
| unseen_d3 | 0.4750 (392) | 0.4751 (549) | 0.4673 (42) | 0.4772 (12) | 0.4763 (5) |
| train_commonpile | 0.5000 (299) | 0.5211 (683) | 0.5167 (414) | 0.5145 (326) | 0.5249 (278) |

generic_cp_1000 max_sim by domain (domain or source): tatoeba 0.4628 (n=1000)

generic_d3_1000 max_sim by domain (domain or source): tatoeba 0.4632 (n=1000)

unseen_cp max_sim by domain (domain or source): Eurlex 0.4795 (n=109), Eurovoc 0.4838 (n=109), GATT_library 0.4633 (n=16), NewZealand-PD-Newspapers 0.5068 (n=109), SEC 0.4602 (n=108), TEDEUTenders 0.4884 (n=2), US-PD-Newspapers 0.4886 (n=109), VoxPopuli 0.4767 (n=109), WTO 0.4776 (n=110), dotgov 0.4892 (n=109), wikipedia_new 0.5073 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.4761 (n=333), hplt 0.4705 (n=334), wikipedia_new 0.4777 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: wikiteam 38.0%/11.5%=3.31x; wikimedia 20.6%/6.9%=2.98x; biodiversity_heritage_library 10.7%/6.2%=1.73x; pre_1929_books 5.8%/0.2%=36.25x; youtube 5.6%/0.5%=11.67x; data_provenance_initiative 5.1%/1.6%=3.16x; cccc 3.0%/2.8%=1.08x; library_of_congress 2.5%/0.2%=15.31x
- generic_cp_1000: wikiteam 34.2%/11.5%=2.98x; wikimedia 23.7%/6.9%=3.41x; biodiversity_heritage_library 11.9%/6.2%=1.92x; pre_1929_books 5.7%/0.2%=35.62x; youtube 5.5%/0.5%=11.46x; data_provenance_initiative 5.0%/1.6%=3.07x; cccc 3.6%/2.8%=1.28x; library_of_congress 2.5%/0.2%=15.62x
- generic_d3_1000: wikiteam 30.3%/11.5%=2.64x; wikimedia 28.4%/6.9%=4.09x; biodiversity_heritage_library 11.4%/6.2%=1.84x; youtube 6.7%/0.5%=13.96x; pre_1929_books 5.9%/0.2%=36.87x; data_provenance_initiative 4.6%/1.6%=2.82x; cccc 2.8%/2.8%=0.99x; library_of_congress 2.6%/0.2%=16.25x
- unseen_cp: biodiversity_heritage_library 26.4%/6.2%=4.26x; wikiteam 13.9%/11.5%=1.21x; usgpo 8.8%/0.9%=9.89x; wikimedia 8.5%/6.9%=1.22x; caselaw_access_project 7.8%/2.6%=3.04x; data_provenance_initiative 4.6%/1.6%=2.82x; youtube 4.0%/0.5%=8.33x; cccc 3.6%/2.8%=1.28x
- unseen_d3: biodiversity_heritage_library 22.2%/6.2%=3.59x; wikiteam 17.8%/11.5%=1.55x; wikimedia 12.4%/6.9%=1.79x; stackv2_edu 5.3%/28.7%=0.18x; cccc 5.2%/2.8%=1.84x; uspto 4.2%/7.3%=0.58x; youtube 3.9%/0.5%=8.12x; data_provenance_initiative 3.7%/1.6%=2.27x
- train_commonpile: wikiteam 19.6%/11.5%=1.7x; stackv2_edu 13.2%/28.7%=0.46x; biodiversity_heritage_library 12.8%/6.2%=2.08x; github_archive 11.1%/10.1%=1.09x; uspto 9.2%/7.3%=1.27x; wikimedia 6.2%/6.9%=0.89x; stackexchange 5.5%/13.7%=0.4x; caselaw_access_project 4.2%/2.6%=1.65x

## dolma3

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_eng | 2000 | 0.4629 [0.4609, 0.4647] | 0.4588 | 0.4088 | 0.1532 |
| generic_cp_1000 | 1000 | 0.4651 [0.4626, 0.4679] | 0.4616 | 0.4100 | 0.1525 |
| generic_d3_1000 | 1000 | 0.4643 [0.4616, 0.4668] | 0.4614 | 0.4093 | 0.1530 |
| unseen_cp | 1000 | 0.4673 [0.4645, 0.4702] | 0.4616 | 0.4147 | 0.1662 |
| unseen_d3 | 1000 | 0.4852 [0.4818, 0.4883] | 0.4804 | 0.4226 | 0.1578 |
| train_dolma3 | 2000 | 0.4893 [0.4866, 0.4919] | 0.4826 | 0.4263 | 0.1578 |

**unseen_cp - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0044 [+0.0010, +0.0080] | 0.525 |
| topk_mean | +0.0059 [+0.0038, +0.0081] | 0.555 |
| mean_sim | +0.0130 [+0.0115, +0.0145] | 0.684 |

**unseen_cp - generic_cp_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0022 [-0.0017, +0.0058] | 0.509 |
| topk_mean | +0.0046 [+0.0022, +0.0070] | 0.542 |
| mean_sim | +0.0137 [+0.0120, +0.0152] | 0.693 |

**unseen_d3 - tatoeba_eng**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0223 [+0.0185, +0.0261] | 0.628 |
| topk_mean | +0.0139 [+0.0116, +0.0164] | 0.624 |
| mean_sim | +0.0046 [+0.0030, +0.0062] | 0.565 |

**unseen_d3 - generic_d3_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0209 [+0.0167, +0.0252] | 0.619 |
| topk_mean | +0.0133 [+0.0106, +0.0160] | 0.619 |
| mean_sim | +0.0048 [+0.0031, +0.0064] | 0.568 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_eng | 0.4600 (1246) | 0.4672 (665) | 0.4702 (54) | 0.4776 (22) | 0.4747 (13) |
| generic_cp_1000 | 0.4630 (552) | 0.4667 (394) | 0.4739 (32) | 0.4708 (13) | 0.4868 (9) |
| generic_d3_1000 | 0.4608 (515) | 0.4675 (446) | 0.4730 (25) | 0.4810 (12) | 0.4303 (2) |
| unseen_cp | 0.4710 (443) | 0.4655 (475) | 0.4557 (56) | 0.4794 (13) | 0.4470 (13) |
| unseen_d3 | 0.4850 (392) | 0.4854 (549) | 0.4821 (42) | 0.4549 (12) | 0.5842 (5) |
| train_dolma3 | 0.4680 (177) | 0.4956 (545) | 0.4953 (564) | 0.4889 (477) | 0.4770 (237) |

generic_cp_1000 max_sim by domain (domain or source): tatoeba 0.4651 (n=1000)

generic_d3_1000 max_sim by domain (domain or source): tatoeba 0.4643 (n=1000)

unseen_cp max_sim by domain (domain or source): Eurlex 0.4721 (n=109), Eurovoc 0.4779 (n=109), GATT_library 0.4553 (n=16), NewZealand-PD-Newspapers 0.4665 (n=109), SEC 0.4614 (n=108), TEDEUTenders 0.4828 (n=2), US-PD-Newspapers 0.4676 (n=109), VoxPopuli 0.4799 (n=109), WTO 0.4535 (n=110), dotgov 0.4794 (n=109), wikipedia_new 0.4495 (n=110)

unseen_d3 max_sim by domain (domain or source): finepdfs 0.4970 (n=333), hplt 0.4939 (n=334), wikipedia_new 0.4646 (n=333)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: common_crawl 98.0%/80.7%=1.21x; stack_edu 0.9%/14.4%=0.06x; olmocr_science_pdfs 0.8%/1.8%=0.41x; finemath 0.4%/2.7%=0.15x; dolma1_7 0.1%/0.1%=0.5x
- generic_cp_1000: common_crawl 98.3%/80.7%=1.22x; stack_edu 0.9%/14.4%=0.06x; olmocr_science_pdfs 0.6%/1.8%=0.32x; dolma1_7 0.1%/0.1%=1.0x; finemath 0.1%/2.7%=0.04x
- generic_d3_1000: common_crawl 98.1%/80.7%=1.22x; stack_edu 1.1%/14.4%=0.08x; olmocr_science_pdfs 0.5%/1.8%=0.27x; finemath 0.3%/2.7%=0.11x
- unseen_cp: common_crawl 89.7%/80.7%=1.11x; olmocr_science_pdfs 6.5%/1.8%=3.51x; stack_edu 1.9%/14.4%=0.13x; finemath 1.4%/2.7%=0.52x; rpj 0.3%/0.2%=1.25x; dolma1_7 0.2%/0.1%=2.0x
- unseen_d3: common_crawl 92.8%/80.7%=1.15x; olmocr_science_pdfs 3.5%/1.8%=1.89x; stack_edu 2.0%/14.4%=0.14x; finemath 1.5%/2.7%=0.56x; dolma1_7 0.2%/0.1%=2.0x
- train_dolma3: common_crawl 91.8%/80.7%=1.14x; stack_edu 3.4%/14.4%=0.23x; olmocr_science_pdfs 2.6%/1.8%=1.43x; finemath 2.0%/2.7%=0.75x; rpj 0.1%/0.2%=0.63x

Nearest-doc subsource (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_eng: literature 47.0%/3.3%=14.34x; entertainment 14.1%/7.8%=1.8x; education_and_jobs 4.3%/4.5%=0.98x; science_math_and_technology 4.2%/14.4%=0.29x; health 4.0%/8.3%=0.48x; games 2.9%/4.9%=0.59x; crime_and_law 2.7%/3.0%=0.9x; religion 2.4%/1.4%=1.68x
- generic_cp_1000: literature 46.6%/3.3%=14.21x; entertainment 13.2%/7.8%=1.69x; science_math_and_technology 4.8%/14.4%=0.33x; education_and_jobs 4.3%/4.5%=0.96x; health 3.9%/8.3%=0.47x; games 3.0%/4.9%=0.61x; crime_and_law 2.9%/3.0%=0.97x; religion 2.6%/1.4%=1.86x
- generic_d3_1000: literature 51.5%/3.3%=15.7x; entertainment 12.8%/7.8%=1.64x; health 4.0%/8.3%=0.48x; crime_and_law 3.8%/3.0%=1.27x; education_and_jobs 3.3%/4.5%=0.74x; science_math_and_technology 3.0%/14.4%=0.21x; religion 2.2%/1.4%=1.57x; games 2.2%/4.9%=0.45x
- unseen_cp: science_math_and_technology 19.0%/14.4%=1.32x; literature 11.5%/3.3%=3.51x; crime_and_law 9.6%/3.0%=3.2x; politics 9.0%/1.6%=5.52x; finance_and_business 8.8%/4.0%=2.2x; health 5.2%/8.3%=0.62x; software 4.7%/5.3%=0.89x; entertainment 4.5%/7.8%=0.58x
- unseen_d3: science_math_and_technology 20.9%/14.4%=1.45x; education_and_jobs 9.2%/4.5%=2.06x; health 9.1%/8.3%=1.09x; literature 6.6%/3.3%=2.01x; entertainment 5.7%/7.8%=0.73x; history_and_geography 4.5%/1.8%=2.56x; software 4.3%/5.3%=0.81x; games 4.3%/4.9%=0.88x
- train_dolma3: science_math_and_technology 15.4%/14.4%=1.07x; software_development 9.0%/9.3%=0.97x; health 8.9%/8.3%=1.07x; entertainment 8.8%/7.8%=1.13x; literature 7.8%/3.3%=2.38x; education_and_jobs 6.7%/4.5%=1.5x; software 6.3%/5.3%=1.2x; games 4.9%/4.9%=1.0x

## dynaword1212

| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |
|---|---|---|---|---|---|
| tatoeba_dan | 2000 | 0.4527 [0.4506, 0.4547] | 0.4482 | 0.4094 | 0.2133 |
| generic_dw_1000 | 1000 | 0.4553 [0.4524, 0.4581] | 0.4508 | 0.4119 | 0.2137 |
| unseen_dw | 1000 | 0.4985 [0.4954, 0.5020] | 0.4970 | 0.4514 | 0.2319 |
| train_dynaword1212 | 2000 | 0.5732 [0.5696, 0.5768] | 0.5705 | 0.5290 | 0.2791 |

**unseen_dw - tatoeba_dan**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0459 [+0.0419, +0.0499] | 0.740 |
| topk_mean | +0.0420 [+0.0384, +0.0455] | 0.757 |
| mean_sim | +0.0186 [+0.0164, +0.0208] | 0.676 |

**unseen_dw - generic_dw_1000**

| metric | diff [95% CI] | AUC |
|---|---|---|
| max_sim | +0.0432 [+0.0391, +0.0478] | 0.729 |
| topk_mean | +0.0396 [+0.0359, +0.0434] | 0.744 |
| mean_sim | +0.0182 [+0.0156, +0.0209] | 0.671 |

max_sim by Llama-2 token length (mean, n):

| set | 1-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| tatoeba_dan | 0.4418 (870) | 0.4597 (982) | 0.4678 (109) | 0.4699 (30) | 0.5050 (9) |
| generic_dw_1000 | 0.4467 (362) | 0.4593 (556) | 0.4604 (60) | 0.4759 (17) | 0.5068 (5) |
| unseen_dw | 0.4791 (248) | 0.5026 (598) | 0.5136 (118) | 0.5147 (26) | 0.5201 (10) |
| train_dynaword1212 | 0.4883 (56) | 0.5311 (257) | 0.5435 (287) | 0.5673 (429) | 0.6007 (971) |

generic_dw_1000 max_sim by domain (domain or source): tatoeba 0.4553 (n=1000)

unseen_dw max_sim by domain (domain or source): dakultur 0.4938 (n=70), depbank 0.4902 (n=69), folketingets-dokumenter 0.5184 (n=70), hvadvilduhelst 0.4457 (n=68), jvj 0.4410 (n=11), kalliope 0.4866 (n=69), kb_administrative_publication 0.5106 (n=69), kb_historical_letters 0.5176 (n=69), logir 0.5214 (n=69), mosel_voxpopuli 0.5061 (n=70), mosel_youtubecommons 0.4543 (n=70), municipality_meetings 0.5673 (n=70), nordjyllandnews 0.5104 (n=70), synne 0.4701 (n=18), tidsskrift-dk 0.4753 (n=69), wikipedia_new 0.4970 (n=69)

Nearest-doc source (share among nearest docs / share in sample = enrichment), top 8:

- tatoeba_dan: enevaeldens_nyheder 48.2%/83.0%=0.58x; opensubtitles 20.4%/0.6%=34.0x; wiki 12.2%/5.7%=2.14x; hest 3.5%/0.3%=13.65x; tv2r 2.1%/1.1%=1.93x; ai-aktindsigt 1.9%/3.7%=0.52x; dannet 1.6%/0.3%=5.33x; wikisource 1.5%/0.1%=30.0x
- generic_dw_1000: enevaeldens_nyheder 48.9%/83.0%=0.59x; opensubtitles 19.1%/0.6%=31.83x; wiki 13.3%/5.7%=2.33x; hest 3.4%/0.3%=13.08x; ai-aktindsigt 2.2%/3.7%=0.6x; tv2r 1.9%/1.1%=1.74x; wikisource 1.6%/0.1%=32.0x; dannet 1.5%/0.3%=5.0x
- unseen_dw: enevaeldens_nyheder 44.3%/83.0%=0.53x; wiki 13.0%/5.7%=2.28x; ai-aktindsigt 9.7%/3.7%=2.64x; retsinformationdk 9.3%/1.9%=4.89x; opensubtitles 4.2%/0.6%=7.0x; cellar 3.9%/1.1%=3.45x; tv2r 3.2%/1.1%=2.94x; ncc_books 3.2%/0.1%=32.0x
- train_dynaword1212: enevaeldens_nyheder 84.7%/83.0%=1.02x; wiki 5.4%/5.7%=0.95x; ai-aktindsigt 2.9%/3.7%=0.8x; retsinformationdk 2.1%/1.9%=1.08x; cellar 1.1%/1.1%=0.93x; tv2r 0.8%/1.1%=0.73x; opensubtitles 0.5%/0.6%=0.83x; ncc_maalfrid 0.5%/0.6%=0.81x

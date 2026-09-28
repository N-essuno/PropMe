# Files excluded from the DFM9 filtered index data

This snapshot is derived from
`/work/olmotrace/mimir_propme/dfm9_memorisation_sources/manifest.tsv`.
It lists manifest artifacts that are wholly excluded: audit evidence and the
unresolved D-05 proxy. Files subject only to row- or message-level filtering
are not called fully excluded because some of their records are emitted.

The preprocessing command also writes the same category-grouped report to
`<output-root>/_reports/excluded_files.md` for the effective run.

## A

No manifest data files are fully excluded.

## B

### B-01

- `B/B-01/legal__registers__dfm9-rlve-prompt-expression-audit.csv` — audit evidence

### B-02

- `B/B-02/legal__registers__dfm9-longalign-content-groups.csv` — audit evidence
- `B/B-02/legal__registers__dfm9-longalign-copyright-marker-rows.csv` — audit evidence

### B-03

- `B/B-03/legal__registers__dfm9-euroblocks-embedded-seed-documents.csv` — audit evidence

### B-04

- `B/B-04/legal__registers__dfm9-euroblocks-seed-risk.csv` — audit evidence

### B-05

- `B/B-05/legal__registers__dfm9-sapient-instruction-family-inventory.csv` — audit evidence

## C

### C-05_C-08

- `C/C-05_C-08/legal__registers__dfm9-sapient-instruction-family-inventory.csv` — audit evidence

### C-09

- `C/C-09/legal__registers__dfm9-tulu3-mixture-component-audit.csv` — audit evidence

### C-10

- `C/C-10/legal__registers__dfm9-tulu-v2-sciriff-if-sft-component-audit.csv` — audit evidence

### C-11_C-14

- `C/C-11_C-14/legal__registers__dfm9-openhermes-component-audit.csv` — audit evidence

## D

### D-02

- `D/D-02/legal__registers__dfm9-tulu-v2-sciriff-if-sft-component-audit.csv` — audit evidence

### D-04

- `D/D-04/legal__registers__dfm9-dolci-toolu-component-audit.csv` — audit evidence

### D-05

- `D/D-05/data__downloads__datasets__dfm_dyna_instruct__data__apertus-sft-mixture__apertus-sft-mixture.parquet` — unresolved selector: the supplied columns cannot identify Mixture-of-Thoughts rows
- `D/D-05/legal__registers__dfm9-mot-expression-risk.csv` — audit evidence

### D-07

- `D/D-07/legal__registers__dfm9-triviaqa-current-reservation-probe.json` — audit evidence
- `D/D-07/legal__registers__dfm9-triviaqa-source-grouping.json` — audit evidence
- `D/D-07/legal__registers__dfm9-triviaqa-source-rights.csv` — audit evidence

## Scope note

The empty D-10 `natural_reasoning` and `omni_math` artifacts are fatal source
inventory errors, not intentional exclusions, so they are not listed above.

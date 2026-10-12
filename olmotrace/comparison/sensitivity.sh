#!/bin/bash
# Concurrency sensitivity on the worst case (Dolma3 verbatim, 64 texts), sized for a 96 GB / 32-CPU cgroup.
# SimpleTrace bounds page tables to 1 GB per worker, so 16x2 and 32x1 are safe for it.
# OLMoTrace has no bound (78.8 GB at 8 workers), so it gets one 16x2 attempt under a 60 GB guard;
# 32x1 is not run for it (it OOM-killed the container on 2026-10-06).
set -e
cd "$(dirname "$0")/../.."
PROPME_DATA_ROOT="${PROPME_DATA_ROOT:-$PWD/propme_data}"
R="python olmotrace/comparison/run_bench.py --inputs olmotrace/comparison/inputs/dolma3_verbatim.jsonl \
   --index-dir $PROPME_DATA_ROOT/indexes/dolma3_index_link \
   --unigram-probs-path 02_unigram_probs/unigram_probs_dolma3_link.json --max-unreclaimable-gb 60"
$R --tool st_text --tag warmup --num-workers 8 --find-threads 4 > /dev/null   # warm the page cache
for rep in 1 2; do
  $R --tool st_text --tag dolma3__dolma3_verbatim__w16t2r$rep --num-workers 16 --find-threads 2
  $R --tool st_text --tag dolma3__dolma3_verbatim__w32t1r$rep --num-workers 32 --find-threads 1
done
$R --tool ot --tag dolma3__dolma3_verbatim__w16t2r1 --num-workers 16 --find-threads 2 || true
echo SENSITIVITY_DONE

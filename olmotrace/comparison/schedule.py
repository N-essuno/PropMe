"""Run the end-to-end comparison schedule (resumable: finished runs are skipped).

Cold phase: each input set is split into 3 slices; the three tool variants
run on every slice in a Latin-square order, so each variant runs first (cold
page cache) on exactly one slice and second/third (warm) on the others.
Warm phase: two repetitions on the full set, in mirrored order
(ot, st_text, st_mixed, st_mixed, st_text, ot).

python olmotrace/comparison/schedule.py [--only dolma3 commonpile dynaword]
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMP = os.path.join(REPO, "olmotrace", "comparison")
# Root of the large data (indexes/, raw/), outside the repository by default.
IDX = os.path.join(os.environ.get("PROPME_DATA_ROOT", os.path.join(REPO, "propme_data")), "indexes")
CORPORA = {
    "dolma3": ([f"{IDX}/dolma3_index_link"], "02_unigram_probs/unigram_probs_dolma3_link.json",
               ["dolma3_verbatim", "english_novel"]),
    "commonpile": ([f"{IDX}/commonpile_index/common_pile_train_index"],
                   "02_unigram_probs/unigram_probs_common_pile_train.json",
                   ["commonpile_verbatim", "english_novel"]),
    "dynaword": ([f"{IDX}/dynaword_index"], "02_unigram_probs/unigram_probs_dynaword.json",
                 ["dynaword_generations", "dynaword_verbatim"]),
}
LATIN = [["ot", "st_text", "st_mixed"], ["st_text", "st_mixed", "ot"], ["st_mixed", "ot", "st_text"]]
WARM = [("warm1", ["ot", "st_text", "st_mixed"]), ("warm2", ["st_mixed", "st_text", "ot"])]


def done_runs() -> set:
    path = os.path.join(CMP, "outputs", "e2e", "runs.jsonl")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["tag"], r["tool"]) for r in map(json.loads, f) if r["returncode"] == 0}


def split(name: str) -> list[str]:
    with open(os.path.join(CMP, "inputs", f"{name}.jsonl")) as f:
        rows = [l for l in f if l.strip()]
    paths = []
    for k in range(3):
        path = os.path.join(CMP, "inputs", "slices", f"{name}_s{k}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(rows[k::3])  # interleaved so slices have the same mix of settings
        paths.append(path)
    return paths


def run(tool, tag, inputs, index_dirs, unigram, workers, threads):
    if (tag, tool) in done_runs():
        print(f"skip {tag} {tool}", flush=True)
        return
    cmd = [sys.executable, os.path.join(CMP, "run_bench.py"), "--tool", tool, "--tag", tag,
           "--inputs", os.path.relpath(inputs, REPO), "--index-dir", *index_dirs,
           "--unigram-probs-path", unigram, "--num-workers", str(workers), "--find-threads", str(threads)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=list(CORPORA))
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--find-threads", type=int, default=4)
    a = p.parse_args()
    for corpus in a.only:
        index_dirs, unigram, sets = CORPORA[corpus]
        for name in sets:
            for k, (path, order) in enumerate(zip(split(name), LATIN)):
                for tool in order:
                    run(tool, f"{corpus}__{name}__cold_s{k}", path, index_dirs, unigram,
                        a.num_workers, a.find_threads)
            full = os.path.join(CMP, "inputs", f"{name}.jsonl")
            for rep, order in WARM:
                for tool in order:
                    run(tool, f"{corpus}__{name}__{rep}", full, index_dirs, unigram,
                        a.num_workers, a.find_threads)


if __name__ == "__main__":
    main()

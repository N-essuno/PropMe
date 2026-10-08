#!/usr/bin/env python3
"""Download the raw data for the unseen prompt sets into sources.RAW_DIR.

Subcommands (each skips files that are already there):

    dynaword1212   Danish Dynaword @ 1.2.12 (9e230b35) minus DFM's excluded
                   subsets: the DFM training data, for the rebuilt index.
    dynaword_new   Dynaword @ main, the subsets DFM did not see (added after
                   1.2.12, or excluded from its training).
    common_corpus  A seeded random subset of Common Corpus shards. Shards mix
                   all collections, so a random subset is a random sample.
    hplt           Seeded random shards of the Dolma 3.5 HPLT pool, top
                   vigintiles (15-19) of all topics. Shards are drawn from the
                   pooled list, so topics keep their share of documents.
    finepdfs       Seeded random FinePDFs eng_Latn files.
    evidence       Dataset and model cards backing the eligibility rule, into
                   evidence/.

Run from the repository root:

    python 00_prepare_data/propensity_settings/download_sources.py dynaword1212 dynaword_new
    python 00_prepare_data/propensity_settings/download_sources.py common_corpus --num-files 40
"""

from __future__ import annotations

import argparse
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

import sources as src


def download(repo: str, files: list[str], local_dir, revision: str | None = None, workers: int = 8) -> None:
    def one(path: str) -> None:
        hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=local_dir)
        print(f"  {repo}: {path}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))


def dynaword_text_files(revision: str) -> dict[str, str]:
    """Subset -> its text parquet file: data/<s>/<s>.parquet at 1.2.12, data/<s>/data.parquet later
    (later versions add a separate metadata.parquet, which holds no text)."""
    files = HfApi().list_repo_files(src.DYNAWORD_REPO, repo_type="dataset", revision=revision)
    return {
        f.split("/")[1]: f
        for f in files
        if f.startswith("data/") and f.count("/") == 2 and f.endswith(".parquet")
        and not f.endswith("/metadata.parquet")
    }


def dynaword_files(revision: str, subsets) -> list[str]:
    available = dynaword_text_files(revision)
    missing = set(subsets) - available.keys()
    if missing:
        raise RuntimeError(f"Dynaword @ {revision} lacks {sorted(missing)}")
    return [available[s] for s in sorted(subsets)]


def cmd_dynaword1212(args) -> None:
    subsets = set(dynaword_text_files(src.DYNAWORD_1212_REVISION))
    trained = sorted(subsets - set(src.DFM_EXCLUDED))
    print(f"Dynaword 1.2.12: {len(subsets)} subsets, {len(trained)} in DFM training")
    download(src.DYNAWORD_REPO, dynaword_files(src.DYNAWORD_1212_REVISION, trained),
             src.RAW_DIR / "dynaword_1212", src.DYNAWORD_1212_REVISION, args.workers)


def cmd_dynaword_new(args) -> None:
    subsets = (*src.DYNAWORD_ADDED_AFTER_1212, *src.DFM_EXCLUDED)
    download(src.DYNAWORD_REPO, dynaword_files("main", subsets), src.RAW_DIR / "dynaword_main", "main", args.workers)


def seeded_choice(files: list[str], n: int, name: str) -> list[str]:
    return sorted(random.Random(f"{src.SEED}-{name}").sample(sorted(files), min(n, len(files))))


def cmd_common_corpus(args) -> None:
    files = [f for f in HfApi().list_repo_files("PleIAs/common_corpus", repo_type="dataset") if f.endswith(".parquet")]
    chosen = seeded_choice(files, args.num_files, "common_corpus")
    print(f"Common Corpus: {len(chosen)} of {len(files)} shards")
    download("PleIAs/common_corpus", chosen, src.RAW_DIR / "common_corpus", workers=args.workers)


def cmd_hplt(args) -> None:
    api = HfApi()

    def list_bin(path: str) -> list[str]:
        return [e.path for e in api.list_repo_tree(src.DOLMA35_REPO, path_in_repo=path, repo_type="dataset")
                if e.path.endswith(".jsonl.zst")]

    bins = [f"hplt/{t}/{v}" for t in src.HPLT_TOPICS for v in src.HPLT_VIGINTILES]
    with ThreadPoolExecutor(16) as pool:
        files = [f for listed in pool.map(list_bin, bins) for f in listed]
    chosen = seeded_choice(files, args.num_files, "hplt")
    print(f"HPLT top vigintiles: {len(chosen)} of {len(files)} shards")
    download(src.DOLMA35_REPO, chosen, src.RAW_DIR / "hplt", workers=args.workers)


def cmd_finepdfs(args) -> None:
    files = [e.path for e in HfApi().list_repo_tree(src.FINEPDFS_REPO, path_in_repo="data/eng_Latn/train",
                                                     repo_type="dataset") if e.path.endswith(".parquet")]
    chosen = seeded_choice(files, args.num_files, "finepdfs")
    print(f"FinePDFs eng_Latn: {len(chosen)} of {len(files)} files")
    download(src.FINEPDFS_REPO, chosen, src.RAW_DIR / "finepdfs", workers=args.workers)


EVIDENCE = {
    "dfm-decoder-open-v0-7b-pt.md": "https://huggingface.co/danish-foundation-models/dfm-decoder-open-v0-7b-pt/raw/main/README.md",
    "danish-dynaword_CHANGELOG.md": "https://huggingface.co/datasets/danish-foundation-models/danish-dynaword/raw/main/CHANGELOG.md",
    "comma-v0.1-2t.md": "https://huggingface.co/common-pile/comma-v0.1-2t/raw/main/README.md",
    "common_corpus.md": "https://huggingface.co/datasets/PleIAs/common_corpus/raw/main/README.md",
    "Olmo-3-1125-32B.md": "https://huggingface.co/allenai/Olmo-3-1125-32B/raw/main/README.md",
    "dolma3_pool.md": "https://huggingface.co/datasets/allenai/dolma3_pool/raw/main/README.md",
    "dolma3_mix-5.5T-1125.md": "https://huggingface.co/datasets/allenai/dolma3_mix-5.5T-1125/raw/main/README.md",
    "dolma3_dolmino_mix-100B-1125.md": "https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125/raw/main/README.md",
    "dolma3_longmino_mix-100B-1125.md": "https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125/raw/main/README.md",
    "dolma3.5_pool.md": "https://huggingface.co/datasets/allenai/dolma3.5_pool/raw/main/README.md",
    "finepdfs.md": "https://huggingface.co/datasets/HuggingFaceFW/finepdfs/raw/main/README.md",
}


def cmd_evidence(args) -> None:
    src.EVIDENCE_DIR.mkdir(exist_ok=True)
    for name, url in EVIDENCE.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            (src.EVIDENCE_DIR / name).write_bytes(response.read())
        print(f"  {name}")


COMMANDS = {
    "dynaword1212": cmd_dynaword1212,
    "dynaword_new": cmd_dynaword_new,
    "common_corpus": cmd_common_corpus,
    "hplt": cmd_hplt,
    "finepdfs": cmd_finepdfs,
    "evidence": cmd_evidence,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commands", nargs="+", choices=list(COMMANDS))
    parser.add_argument("--num-files", type=int, default=24,
                        help="Files to draw for common_corpus / hplt / finepdfs (default 24).")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (default 8).")
    args = parser.parse_args()
    src.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for command in args.commands:
        print(f"=== {command} ===", flush=True)
        COMMANDS[command](args)


if __name__ == "__main__":
    main()

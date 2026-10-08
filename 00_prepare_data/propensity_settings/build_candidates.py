#!/usr/bin/env python3
"""Candidate sentences for the unseen prompt sets, one per document.

Two steps per corpus (dw = Dynaword, cp = Common Pile, d3 = Dolma 3; sources
and eligibility in sources.py):

    pool       Draw up to --pool-size random eligible documents per source
               (seeded; uniform over each source's documents in the
               downloaded files) -> work/pool_<corpus>.jsonl. Filters: Common
               Corpus language == English and collection eligible (science
               collections only with a publication date after the Comma
               training data); HPLT records whose main language is English.
    sentences  Keep documents with a FineWeb-Edu int score >= --min-quality
               (English; quality_score.py adds the scores; Danish sources are
               curated and not scored) and pick one whole sentence per
               document, of Tatoeba length: target lengths are drawn from the
               token lengths of the corpus's generic (Tatoeba) prompt set, and
               each goes to the next unused document (random order) with a
               sentence within length_tolerance(target) tokens of it; a random
               such sentence is taken (prose-like, see prose_sentences in
               scripts/embedding_similarity.py, and in the
               corpus language, see in_language). Targets no document can meet
               are skipped. Duplicate texts are dropped, and up to
               ceil(FINAL_SIZE / n_sources * OVERSAMPLE) are kept per source
               -> candidates_<corpus>.jsonl.

Candidates get the fields of the *_2k.jsonl samples (id, text, source, lang,
n_tokens in Llama-2 tokens without BOS) plus domain (= source), subsource,
doc_id, date, url, quality and target_tokens.

Run from the repository root:

    python 00_prepare_data/propensity_settings/build_candidates.py pool --corpora dw cp d3
    # quality_score.py (GPU) for cp and d3
    python 00_prepare_data/propensity_settings/build_candidates.py sentences --corpora dw cp d3
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard

import sources as src

REPO_ROOT = src.DATA_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from embedding_similarity import prose_sentences  # noqa: E402

WORK_DIR = src.DATA_DIR / "work"
PROMPT_DATA_DIR = REPO_ROOT / "memorization_experiment" / "data"
# Shortest sentence considered, in words: Tatoeba prompts go down to 3-4 tokens.
MIN_WORDS = 3
TOKENIZER_MODEL = "meta-llama/Llama-2-7b-hf"
# Documents are stored up to this many characters: enough for the quality
# classifier (first 512 tokens) and for picking a sentence.
MAX_DOC_CHARS = 100_000
# Very frequent function words. A sentence is taken to be in its language when
# at least MIN_FUNCTION_SHARE of its words are in the list (Danish also must not
# contain the Icelandic/Faroese letters þ and ð). This drops sentences in
# other languages (e.g. Latin, Icelandic, German or English letters in
# kb_historical_letters, English abstracts in tidsskrift-dk), and also
# noun-only fragments such as name lists and headlines.
FUNCTION_WORDS = {
    "eng": set(
        "the of and to in is that for it was on with as by at be this are from or an not have which he she they "
        "we you i his her their its has had were been will would can could there but if all also one more than "
        "when who what so no into about these those may such other only some any do does did out up most".split()
    ),
    "dan": set(
        "og i at er det en til af på som med for de den ikke har var der et om fra kan sig blev jeg du han hun vi "
        "man så men eller skal vil være have sin sit sine efter over under ved hvor hvad når også kun alle nu da "
        "her meget mere end hvis dem ham hende min mit mine vores deres denne dette disse hvordan hvorfor".split()
    ),
}
MIN_FUNCTION_SHARE = 0.15
WORD = re.compile(r"[^\W\d_]+")


def in_language(sentence: str, lang: str) -> bool:
    words = WORD.findall(sentence.lower())
    share = sum(w in FUNCTION_WORDS[lang] for w in words) / max(1, len(words))
    return share >= MIN_FUNCTION_SHARE and not (lang == "dan" and re.search("[þðÞÐ]", sentence))


def parse_date(value) -> str:
    """Normalize a date-ish value to YYYY[-MM[-DD]], or '' if unknown."""
    match = re.search(r"\d{4}(-\d{2}(-\d{2})?)?", str(value or ""))
    return match.group(0) if match else ""


# ----------------------------------------------------------------------------
# pool
# ----------------------------------------------------------------------------


def doc(source: str, doc_id, text: str, subsource: str = "", date: str = "", url: str = "", **extra) -> dict:
    return {"source": source, "subsource": subsource, "doc_id": str(doc_id), "date": date, "url": url,
            "text": text[:MAX_DOC_CHARS], **extra}


def pool_dynaword(pool_size: int) -> list[dict]:
    docs = []
    for subset in (*src.DYNAWORD_ADDED_AFTER_1212, *src.DFM_EXCLUDED):
        table = pq.read_table(src.RAW_DIR / "dynaword_main" / "data" / subset / "data.parquet",
                              columns=["id", "text", "created"])
        # Python lists rather than table.take: large subsets overflow pyarrow's 32-bit string offsets.
        ids, texts, created = (table.column(c).to_pylist() for c in ("id", "text", "created"))
        rows = [i for i, t in enumerate(texts) if t and t.strip()]
        chosen = sorted(random.Random(f"{src.SEED}-pool-{subset}").sample(rows, min(pool_size, len(rows))))
        docs += [doc(subset, ids[i], texts[i], date=str(created[i] or "")) for i in chosen]
        print(f"  {subset}: {len(chosen)} of {len(rows)} docs", flush=True)
    return docs


def wikipedia_docs(corpus: str, pool_size: int) -> list[dict]:
    path = src.RAW_DIR / "wikipedia" / f"{corpus}_docs.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    picked = rows[:pool_size]  # already in seeded random order
    print(f"  wikipedia_new: {len(picked)} of {len(rows)} articles", flush=True)
    # The content date is "created after": page id >= the id threshold of the date.
    return [doc("wikipedia_new", r["id"], r["text"], subsource=r["lang"], date=f">{r['created_after']}",
                url=r["url"], title=r["title"]) for r in picked]


def common_corpus_shard(path: Path) -> list[dict]:
    """Eligible English rows of one Common Corpus shard."""
    table = pq.read_table(path, columns=["identifier", "collection", "language", "date", "title"])
    wanted = set(src.COMMON_CORPUS_ELIGIBLE) | set(src.COMMON_CORPUS_DATED)
    mask = pc.and_(
        pc.equal(table.column("language"), "English"),
        pc.is_in(table.column("collection"), value_set=pa.array(sorted(wanted))),
    )
    idx = pc.indices_nonzero(mask).to_pylist()
    if not idx:
        return []
    meta = table.take(idx).to_pylist()
    rows = []
    for i, m in zip(idx, meta):
        date = parse_date(m["date"])
        if m["collection"] in src.COMMON_CORPUS_DATED and not date[:10] > src.COMMA_DATASET_DATE:
            continue
        # Shard names repeat across common_corpus_<n>/ folders, so key by folder too.
        rows.append({"shard": f"{path.parent.name}/{path.name}", "row": i, "collection": m["collection"], "identifier": m["identifier"],
                     "date": date, "title": m["title"] or ""})
    return rows


def pool_common_corpus(pool_size: int, workers: int) -> list[dict]:
    shards = sorted((src.RAW_DIR / "common_corpus").rglob("*.parquet"))
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for shard_rows in pool.map(common_corpus_shard, shards) for r in shard_rows]
    by_collection = defaultdict(list)
    for r in rows:
        by_collection[r["collection"]].append(r)
    chosen = []
    for collection, items in sorted(by_collection.items()):
        picked = random.Random(f"{src.SEED}-pool-cc-{collection}").sample(items, min(pool_size, len(items)))
        chosen += picked
        print(f"  {collection}: {len(picked)} of {len(items)} eligible English docs in {len(shards)} shards", flush=True)
    by_shard = defaultdict(list)
    for r in chosen:
        by_shard[r["shard"]].append(r)
    paths = {f"{p.parent.name}/{p.name}": p for p in shards}
    docs = []
    for shard, items in sorted(by_shard.items()):
        texts = pq.read_table(paths[shard], columns=["text"]).column("text")
        for r in items:
            docs.append(doc(r["collection"], r["identifier"], texts[r["row"]].as_py() or "", date=r["date"],
                            title=r["title"]))
    return docs


def hplt_shard(args: tuple[Path, int]) -> list[dict]:
    """Uniform sample (reservoir) of up to k English records of one HPLT shard."""
    path, k = args
    rng = random.Random(f"{src.SEED}-pool-hplt-{path.name}-{path.parent.name}-{path.parent.parent.name}")
    sample, seen = [], 0
    with open(path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            record = json.loads(line)
            lang = record.get("lang") or []
            if not record.get("text") or not lang or lang[0] != "eng_Latn":
                continue
            seen += 1
            item = doc("hplt", record.get("id"), record["text"],
                       subsource=(record.get("metadata", {}).get("wo_topic") or "").removeprefix("__label__"),
                       date=str(record.get("ts") or ""), url=record.get("u") or "")
            if len(sample) < k:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < k:
                    sample[j] = item
    return sample


def pool_hplt(pool_size: int, workers: int) -> list[dict]:
    shards = sorted((src.RAW_DIR / "hplt").rglob("*.jsonl.zst"))
    per_shard = math.ceil(pool_size / len(shards))
    with ProcessPoolExecutor(min(workers, len(shards))) as pool:
        docs = [d for shard_docs in pool.map(hplt_shard, [(s, per_shard) for s in shards]) for d in shard_docs]
    docs = random.Random(f"{src.SEED}-pool-hplt").sample(docs, min(pool_size, len(docs)))
    print(f"  hplt: {len(docs)} docs from {len(shards)} shards", flush=True)
    return docs


def finepdfs_file(args: tuple[Path, int]) -> list[dict]:
    path, k = args
    parquet = pq.ParquetFile(path)
    n_rows = parquet.metadata.num_rows
    wanted = sorted(random.Random(f"{src.SEED}-pool-finepdfs-{path.name}").sample(range(n_rows), min(k, n_rows)))
    # Read only the row groups holding the sampled rows (files are several GB).
    docs, start, w = [], 0, 0
    for rg in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(rg).num_rows
        local = []
        while w < len(wanted) and wanted[w] < start + size:
            local.append(wanted[w] - start)
            w += 1
        if local:
            rows = parquet.read_row_group(rg, columns=["id", "text", "dump", "url"]).take(local).to_pylist()
            docs += [doc("finepdfs", r["id"], r["text"] or "", subsource=r["dump"], url=r["url"] or "") for r in rows]
        start += size
    return docs


def pool_finepdfs(pool_size: int, workers: int) -> list[dict]:
    files = sorted((src.RAW_DIR / "finepdfs").rglob("*.parquet"))
    per_file = math.ceil(pool_size / len(files))
    with ProcessPoolExecutor(min(workers, len(files))) as pool:
        docs = [d for file_docs in pool.map(finepdfs_file, [(f, per_file) for f in files]) for d in file_docs]
    docs = [d for d in docs if d["text"].strip()]
    docs = random.Random(f"{src.SEED}-pool-finepdfs").sample(docs, min(pool_size, len(docs)))
    print(f"  finepdfs: {len(docs)} docs from {len(files)} files", flush=True)
    return docs


def build_pool(corpus: str, args: argparse.Namespace) -> None:
    print(f"=== pool {corpus} ===", flush=True)
    if corpus == "dw":
        docs = pool_dynaword(args.pool_size) + wikipedia_docs(corpus, args.pool_size)
    elif corpus == "cp":
        docs = pool_common_corpus(args.pool_size, args.workers) + wikipedia_docs(corpus, args.pool_size)
    else:
        docs = (pool_hplt(args.pool_size, args.workers) + pool_finepdfs(args.pool_size, args.workers)
                + wikipedia_docs(corpus, args.pool_size))
    write_jsonl(WORK_DIR / f"pool_{corpus}.jsonl", docs)
    print(f"{corpus}: {len(docs)} docs; by source {dict(Counter(d['source'] for d in docs))}")


# ----------------------------------------------------------------------------
# sentences
# ----------------------------------------------------------------------------


def length_tolerance(target: int) -> int:
    """Allowed |sentence length - target| in tokens: 1 up to 10 tokens, then 10%."""
    return max(1, round(0.1 * target))


def generic_lengths(corpus: str) -> list[int]:
    """Token lengths of the corpus's generic (Tatoeba) prompt set, which the candidates match."""
    path = PROMPT_DATA_DIR / src.CORPUS_DATASET[corpus] / "generic" / "generic_prompts.jsonl"
    return [row["n_tokens"] for row in read_jsonl(path)]


class SentencePool:
    """The language-checked prose sentences of a source's documents, with token lengths, tokenized lazily."""

    def __init__(self, docs: list[dict], lang: str, enc):
        self.docs, self.lang, self.enc = docs, lang, enc
        self.sentences: dict[int, list[tuple[str, int]]] = {}
        self.used: set[int] = set()

    def of(self, i: int) -> list[tuple[str, int]]:
        if i not in self.sentences:
            texts = [t for t in prose_sentences(self.docs[i]["text"], min_words=MIN_WORDS)
                     if in_language(t, self.lang)]
            lengths = [len(ids) for ids in self.enc(texts, add_special_tokens=False).input_ids] if texts else []
            self.sentences[i] = list(zip(texts, lengths))
        return self.sentences[i]

    def take(self, target: int, rng: random.Random, exclude: set[str]) -> tuple[int, str, int] | None:
        """The first unused document with a sentence of about `target` tokens, and a random such sentence."""
        tolerance = length_tolerance(target)
        for i in range(len(self.docs)):
            if i in self.used:
                continue
            fits = [(t, n) for t, n in self.of(i) if abs(n - target) <= tolerance and t not in exclude]
            if fits:
                self.used.add(i)
                sentence, n_tokens = rng.choice(fits)
                return i, sentence, n_tokens
        return None


def build_sentences(corpus: str, args: argparse.Namespace, enc) -> None:
    print(f"=== sentences {corpus} ===", flush=True)
    lang = src.CORPUS_LANG[corpus]
    pool_path = WORK_DIR / f"pool_{corpus}.jsonl"
    scores_path = WORK_DIR / f"quality_{corpus}.jsonl"
    docs = read_jsonl(pool_path)
    scored = lang == "eng"
    if scored:
        scores = {(r["source"], r["doc_id"]): r for r in read_jsonl(scores_path)}
        for d in docs:
            d["quality"] = scores[(d["source"], d["doc_id"])]["quality"]
            d["quality_int"] = scores[(d["source"], d["doc_id"])]["quality_int"]
    by_source = defaultdict(list)
    seen_docs = set()
    for d in docs:
        # Common Corpus repeats some documents (same identifier) across shards; one sentence per document.
        if (d["source"], d["doc_id"]) in seen_docs:
            continue
        seen_docs.add((d["source"], d["doc_id"]))
        by_source[d["source"]].append(d)
    sources = sorted(by_source)
    target = math.ceil(src.FINAL_SIZE / len(sources) * src.OVERSAMPLE)
    reference = generic_lengths(corpus)
    seen_texts: set[str] = set()
    candidates, report = [], {}
    for source in sources:
        items = by_source[source]
        rng = random.Random(f"{src.SEED}-sentences-{corpus}-{source}")
        rng.shuffle(items)
        passed = [d for d in items if not scored or d["quality_int"] >= args.min_quality]
        pool = SentencePool(passed, lang, enc)
        kept = unmatched = 0
        # Stop after many targets in a row find no document: the source is exhausted.
        misses_in_a_row = 0
        while kept < target and misses_in_a_row < 50 and len(pool.used) < len(passed):
            target_tokens = rng.choice(reference)
            picked = pool.take(target_tokens, rng, seen_texts)
            if picked is None:
                unmatched += 1
                misses_in_a_row += 1
                continue
            misses_in_a_row = 0
            i, sentence, n_tokens = picked
            d = passed[i]
            seen_texts.add(sentence)
            candidates.append({
                "id": f"{corpus}-{source}-{d['doc_id']}",
                "text": sentence,
                "source": source,
                "domain": source,
                "subsource": d["subsource"],
                "lang": lang,
                "n_tokens": n_tokens,
                "target_tokens": target_tokens,
                "doc_id": d["doc_id"],
                "date": d["date"],
                "url": d["url"],
                "quality": d.get("quality"),
                "quality_int": d.get("quality_int"),
            })
            kept += 1
        report[source] = {"pool": len(items), "quality_pass": len(passed), "docs_used": len(pool.used),
                          "docs_tokenized": len(pool.sentences), "unmatched_targets": unmatched,
                          "candidates": kept, "target": target}
        print(f"  {source}: {report[source]}", flush=True)
    write_jsonl(src.DATA_DIR / f"candidates_{corpus}.jsonl", candidates)
    (WORK_DIR / f"candidates_report_{corpus}.json").write_text(json.dumps(report, indent=2))
    print(f"{corpus}: {len(candidates)} candidates")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("pool", "sentences"))
    parser.add_argument("--corpora", nargs="+", choices=list(src.CORPUS_SOURCES), default=list(src.CORPUS_SOURCES))
    parser.add_argument("--pool-size", type=int, default=4000, help="Max documents drawn per source (default 4000).")
    parser.add_argument("--min-quality", type=int, default=2,
                        help="Minimum FineWeb-Edu int score (quality_score.py) for English documents (default 2).")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.step == "pool":
        for corpus in args.corpora:
            build_pool(corpus, args)
    else:
        from transformers import AutoTokenizer

        enc = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, add_bos_token=False, add_eos_token=False)
        for corpus in args.corpora:
            build_sentences(corpus, args, enc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Embedding similarity of prompt sets to random training docs, compared with Tatoeba.

Tests whether a prompt set (e.g. the specific, unseen-source prompt sets) is
closer to the training corpora than the Tatoeba sentences behind the generic
prompt sets. English sets are compared with Common Pile and Dolma3, Danish
sets with Dynaword.

Steps (each cached under --output-dir, so they can run on different machines):

1. Sample: draw --num-docs random documents from each index with SimpleTrace's
   engine wrapper (uniform over document ids, skipping docs shorter than
   --min-doc-tokens Llama-2 tokens). The next sampled docs give a baseline set
   of --num-train-sentences "training sentences" (one per doc, disjoint from
   the --num-docs docs): in-distribution sentences to calibrate the scores.
   Needs the indexes and infini_gram, not torch (--sample-only stops here).
2. Embed: prompts and training sentences as queries (with --instruction), docs
   as documents (no instruction, truncated to --doc-max-tokens), with
   Qwen3-Embedding (last-token pooling, L2-normalized) in vLLM.
3. Analyze, per (prompt set, corpus):
   - max_sim: highest cosine to any sampled doc (the main metric),
   - topk_mean: mean of the --top-k highest cosines (less noisy than max),
   - mean_sim: mean cosine to all sampled docs,
   - for every --compare pair A:B (default: every set against the Tatoeba
     sample of its language), the A minus B difference with a bootstrap 95%
     CI, and the AUC P(max_sim of a random A prompt > that of a random B prompt),
   - the same metrics per Llama-2 token-length bin (sentence length changes
     cosine scores on its own),
   - the source of each prompt's nearest doc (Dynaword `source`, Common Pile
     subset, Dolma3 source and WebOrganizer topic), compared with the source
     mix of the sample: enrichment = share among nearest docs / share in sample.

Outputs: <output-dir>/docs/ (sampled docs, training sentences),
<output-dir>/embeddings/<model>/ (cached .npy), and
<output-dir>/<model>/ (per_prompt_<set>_<corpus>.jsonl, summary.{json,md}).

Examples (from the repository root). Sampling needs infini_gram (no torch);
embedding needs vLLM (or transformers with --backend transformers), so the two
steps can run in different Python environments:

    # 1. Sample docs (CPU is fine)
    python scripts/embedding_similarity.py --sample-only

    # 2. Embed and analyze on a GPU
    python scripts/embedding_similarity.py

    # Prompt-free encoder as a robustness check (outputs under bge-m3/)
    python scripts/embedding_similarity.py --model BAAI/bge-m3 --instruction ''

    # Other prompt sets, compared pairwise (summary written to unseen.{json,md}),
    # against DFM's training data (dynaword1212)
    python scripts/embedding_similarity.py --corpora dynaword1212 \\
        --extra-sets unseen_dw=memorization_experiment/data/dynaword2/specific/specific_prompts.jsonl:dan:unseen \\
        --sets tatoeba_dan unseen_dw --compare unseen_dw:tatoeba_dan --summary-name unseen

    # CPU smoke test of step 2: small model, 200 prompts, 500 docs
    python scripts/embedding_similarity.py --backend transformers \\
        --model Qwen/Qwen3-Embedding-0.6B --limit 200 --limit-docs 500
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "00_prepare_data" / "propensity_settings"
DEFAULT_OUTPUT_DIR = DATA_DIR / "embedding_similarity"
# Root of the large data (indexes/, raw/), outside the repository by default.
INDEXES_ROOT = str(Path(os.environ.get("PROPME_DATA_ROOT", REPO_ROOT / "propme_data")) / "indexes")
# The separate Dolma3 split indexes, as in scripts/test_simpletrace_dolma3.py
# and 00_prepare_data/propensity_settings/run_tracing.py.
DOLMA3_SPLIT_INDEX_DIRS = (
    *(f"{INDEXES_ROOT}/dolma3_split{i}_index" for i in range(1, 14)),
    f"{INDEXES_ROOT}/dolma3_split13_bis_index",
    *(f"{INDEXES_ROOT}/dolma3_split{i}_index" for i in range(14, 26)),
    f"{INDEXES_ROOT}/dolma3_split26_index_6shards",
)
CORPORA = {
    "dynaword": (f"{INDEXES_ROOT}/dynaword_index",),
    "commonpile": (f"{INDEXES_ROOT}/commonpile_index/common_pile_train_index",),
    "dolma3": DOLMA3_SPLIT_INDEX_DIRS,
    # Dynaword 1.2.12 minus the subsets DFM skipped: DFM's actual training data
    # (00_prepare_data/propensity_settings/build_dynaword1212.py). Not a default corpus.
    "dynaword1212": (f"{INDEXES_ROOT}/dynaword1212_index",),
}
DEFAULT_CORPORA = ("dynaword", "commonpile", "dolma3")
CORPORA_BY_LANG = {"eng": ("commonpile", "dolma3"), "dan": ("dynaword", "dynaword1212")}
# name -> (file, lang, kind); by default every set is compared with the "tatoeba"
# set of its language. Files are relative to DATA_DIR (absolute paths work too);
# --extra-sets adds more.
PROMPT_SETS = {
    "tatoeba_eng": ("tatoeba_eng_2k.jsonl", "eng", "tatoeba"),
    "tatoeba_dan": ("tatoeba_dan_2k.jsonl", "dan", "tatoeba"),
}
DEFAULT_INSTRUCTION = (
    "Given a sentence, retrieve documents that match its topic, genre and writing style"
)
LENGTH_BINS = ((1, 10), (11, 20), (21, 30), (31, 45), (46, 10**9))
# Sentence ends followed by a capitalized word, so "stk. 2" or "e.g. this" stay whole.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“„(]?[A-ZÆØÅ])")


# ----------------------------------------------------------------------------
# Step 1: sampling docs from the indexes
# ----------------------------------------------------------------------------


def available_cpus() -> int:
    """CPUs this process may use, honoring the cgroup quota (os.cpu_count does not)."""
    cpus = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return cpus


def parse_source(raw_metadata) -> dict:
    """Doc id, source and subsource from the raw InfiniGram metadata.

    The raw record is {"path": <file>, "linenum": ..., "metadata": {...}}.
    Common Pile and Dolma3 files sit in one directory per source, which gives
    the source: Common Pile `wikiteam/wikiteam.chunk.26.jsonl.gz` -> `wikiteam`;
    Dolma3 `common_crawl-finance_and_business-0018/shard_....jsonl.zst` ->
    source `common_crawl`, subsource (WebOrganizer topic or language)
    `finance_and_business`. Dolma3's own metadata["source"] is not usable
    (URLs, crawl names). Dynaword is a single file and stores the source in
    metadata["source"].
    """
    if isinstance(raw_metadata, str):
        try:
            raw = json.loads(raw_metadata)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw_metadata)
    else:
        raw = raw_metadata or {}
    inner = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    path = str(raw.get("path", ""))
    source, subsource = str(inner.get("source") or ""), ""
    if "/" in path:
        parts = path.split("/")[0].split("-")
        if len(parts) > 1 and re.fullmatch(r"(part)?\d+", parts[-1]):
            parts = parts[:-1]
        source, subsource = parts[0], "-".join(parts[1:])
    return {"doc_id": str(inner.get("id", "")), "source": source or "unknown", "subsource": subsource}


def pick_sentence(text: str, rng: random.Random, min_words: int = 4, max_words: int = 60) -> str | None:
    """A random prose-like sentence of a doc (capitalized, ending in .!?, mostly letters)."""
    candidates = prose_sentences(text, min_words, max_words)
    return rng.choice(candidates) if candidates else None


def prose_sentences(text: str, min_words: int = 4, max_words: int = 60) -> list[str]:
    """The prose-like sentences of a doc, in order (see pick_sentence)."""
    candidates = []
    for paragraph in text.split("\n"):
        for sentence in SENTENCE_SPLIT.split(paragraph.strip()):
            sentence = sentence.strip()
            if not min_words <= len(sentence.split()) <= max_words:
                continue
            if not sentence[0].isupper() or sentence[-1] not in ".!?":
                continue
            last_word = sentence.split()[-1][:-1]
            if len(last_word) < 2 or "." in last_word:  # an abbreviation, e.g. "v." or "N.C."
                continue
            if sum(c.isalpha() or c.isspace() for c in sentence) / len(sentence) < 0.85:
                continue
            candidates.append(sentence)
    return candidates


def sample_corpus(corpus: str, args: argparse.Namespace, docs_path: Path, sentences_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "03_tracing"))
    from simple_trace import PageTableBoundedEngine  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, add_bos_token=False, add_eos_token=False
    )
    index_dirs = CORPORA[corpus]
    missing = [d for d in index_dirs if not Path(d).is_dir()]
    if missing:
        raise FileNotFoundError(f"Index directories not found: {missing}")
    engine = PageTableBoundedEngine(
        list(index_dirs) if len(index_dirs) > 1 else index_dirs[0],
        tokenizer.eos_token_id,
        1e9,
    )
    total = engine.engine.get_total_doc_cnt()
    rng = random.Random(f"{args.seed}-{corpus}")
    # Sampling without replacement from range() does not materialize it, so
    # this is cheap even for Dolma3's billions of docs.
    candidates = rng.sample(range(total), k=min(total, 5 * (args.num_docs + args.num_train_sentences)))
    fetch_tokens = 2 * args.doc_max_tokens  # Llama-2 tokens; roomy for --doc-max-tokens Qwen tokens

    def fetch(doc_ix: int) -> dict:
        doc = engine.get_doc_by_ix(doc_ix=doc_ix, max_disp_len=fetch_tokens)
        return {
            "doc_ix": doc_ix,
            "doc_len": doc.get("doc_len"),
            **parse_source(doc.get("metadata")),
            "text": tokenizer.decode(doc.get("token_ids", [])),
        }

    docs, sentences, skipped = [], [], 0
    sentence_rng = random.Random(f"{args.seed}-{corpus}-sentences")
    progress = tqdm(total=args.num_docs + args.num_train_sentences, desc=f"Sampling {corpus}", unit="doc")
    with ThreadPoolExecutor(args.sample_threads) as pool:
        for start in range(0, len(candidates), 1024):
            for doc in pool.map(fetch, candidates[start:start + 1024]):
                if (doc["doc_len"] or 0) < args.min_doc_tokens or not doc["text"].strip():
                    skipped += 1
                elif len(docs) < args.num_docs:
                    docs.append(doc)
                    progress.update()
                elif len(sentences) < args.num_train_sentences:
                    sentence = pick_sentence(doc["text"], sentence_rng)
                    if sentence is None:
                        skipped += 1
                        continue
                    sentences.append({
                        "id": f"{corpus}-{doc['doc_ix']}",
                        "text": sentence,
                        "n_tokens": len(tokenizer.encode(sentence)),
                        **{k: doc[k] for k in ("doc_ix", "doc_id", "source", "subsource")},
                    })
                    progress.update()
            if len(docs) >= args.num_docs and len(sentences) >= args.num_train_sentences:
                break
    progress.close()
    if len(docs) < args.num_docs or len(sentences) < args.num_train_sentences:
        raise RuntimeError(
            f"{corpus}: only {len(docs)} docs and {len(sentences)} training sentences "
            f"from {len(candidates)} candidates."
        )
    print(f"{corpus}: {total:,} docs in the index; skipped {skipped} short or sentence-less docs.")
    write_jsonl(docs_path, docs)
    write_jsonl(sentences_path, sentences)


# ----------------------------------------------------------------------------
# Step 2: embeddings
# ----------------------------------------------------------------------------


class Embedder:
    """Sentence-transformers-style embedding model run with vLLM (or transformers on CPU).

    Pooling follows the model's own configuration unless --pooling is given
    (Qwen3-Embedding: last token, bge-m3: CLS). Texts are tokenized here and
    truncated between the special tokens the tokenizer adds (Qwen3-Embedding
    appends <|endoftext|> and pools it; XLM-R models wrap texts in <s> ... </s>),
    so a truncated doc keeps them, and both backends get the same token ids.
    """

    def __init__(self, args: argparse.Namespace):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        with_special = self.tokenizer("a").input_ids
        plain = self.tokenizer("a", add_special_tokens=False).input_ids
        start = next(
            (i for i in range(len(with_special) - len(plain) + 1) if with_special[i:i + len(plain)] == plain),
            None,
        )
        if start is None:
            raise RuntimeError(f"Cannot locate the text tokens {plain} in {with_special} ({args.model}, 'a').")
        self.prefix_ids = with_special[:start]
        self.suffix_ids = with_special[start + len(plain):]
        self.pooling = args.pooling
        self.backend = args.backend
        if self.backend == "vllm":
            from vllm import LLM
            from vllm.config import PoolerConfig

            # With chunked prefill (vLLM's default) a run hung with the GPU
            # idle on its last few docs. No input exceeds max_model_len, so
            # chunking is never needed, and the texts share no prefixes worth
            # caching: both are off.
            max_model_len = max(args.doc_max_tokens, args.query_max_tokens) + 16
            self.llm = LLM(
                model=args.model,
                runner="pooling",
                pooler_config=PoolerConfig(seq_pooling_type=args.pooling.upper()) if args.pooling else None,
                dtype="bfloat16" if args.dtype == "auto" else args.dtype,
                max_model_len=max_model_len,
                enable_chunked_prefill=False,
                enable_prefix_caching=False,
                max_num_batched_tokens=max(32768, max_model_len),
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed,
            )
        else:
            import torch
            from transformers import AutoModel

            self.torch = torch
            self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cpu":
                torch.set_num_threads(available_cpus())
            dtype = args.dtype
            if dtype == "auto":
                dtype = "bfloat16" if self.device.startswith("cuda") else "float32"
            print(f"Loading {args.model} on {self.device} ({dtype}, {torch.get_num_threads()} CPU threads)...")
            self.model = AutoModel.from_pretrained(
                args.model, dtype=getattr(torch, dtype), attn_implementation="sdpa"
            ).to(self.device).eval()
            self.batch_size = args.batch_size

    def __call__(self, texts: list[str], max_tokens: int, desc: str) -> np.ndarray:
        keep = max_tokens - len(self.prefix_ids) - len(self.suffix_ids)
        ids = [
            self.prefix_ids + plain[:keep] + self.suffix_ids
            for plain in self.tokenizer(texts, add_special_tokens=False).input_ids
        ]
        print(f"{desc}: {len(ids)} texts, {sum(map(len, ids)):,} tokens.")
        if self.backend == "vllm":
            outputs = self.llm.embed([{"prompt_token_ids": x} for x in ids])
            emb = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        else:
            emb = self._embed_transformers(ids, desc)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _embed_transformers(self, ids: list[list[int]], desc: str) -> np.ndarray:
        torch = self.torch
        pad_id = self.tokenizer.pad_token_id
        # Longest first: batches pad less, and an out-of-memory error shows up at once.
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.zeros((len(ids), self.model.config.hidden_size), dtype=np.float32)
        for start in tqdm(range(0, len(order), self.batch_size), desc=desc, unit="batch"):
            idx = order[start:start + self.batch_size]
            width = max(len(ids[i]) for i in idx)
            # Left padding puts every sequence's last token at position -1.
            input_ids = torch.tensor([[pad_id] * (width - len(ids[i])) + ids[i] for i in idx])
            mask = torch.tensor([[0] * (width - len(ids[i])) + [1] * len(ids[i]) for i in idx])
            mask = mask.to(self.device)
            with torch.inference_mode():
                hidden = self.model(input_ids=input_ids.to(self.device), attention_mask=mask).last_hidden_state.float()
            if self.pooling in (None, "last"):
                pooled = hidden[:, -1]
            elif self.pooling == "cls":
                pooled = hidden[torch.arange(len(idx)), width - mask.sum(dim=1)]  # first real token
            else:
                pooled = (hidden * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            out[idx] = pooled.cpu().numpy()
        return out


def cached_embeddings(
    embedder_factory, cache_dir: Path, name: str, texts: list[str], max_tokens: int, prefix: str
) -> np.ndarray:
    """Embed `prefix + text` for each text, cached by model, settings and texts."""
    key = hashlib.sha1(
        json.dumps([max_tokens, prefix, texts], ensure_ascii=False).encode()
    ).hexdigest()[:12]
    path = cache_dir / f"{name}_{key}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    emb = embedder_factory()([prefix + t for t in texts], max_tokens, desc=f"Embedding {name}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb.astype(np.float16))
    return emb


# ----------------------------------------------------------------------------
# Step 3: analysis
# ----------------------------------------------------------------------------


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    diffs = np.array([
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(n_boot)
    ])
    return [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b), from ranks (Mann-Whitney U / (n_a * n_b))."""
    ranks = np.empty(len(a) + len(b))
    ranks[np.argsort(np.concatenate([a, b]), kind="stable")] = np.arange(1, len(a) + len(b) + 1)
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def length_bin(n_tokens: int) -> str:
    for lo, hi in LENGTH_BINS:
        if lo <= n_tokens <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "unknown"


def analyze_pair(
    set_name: str, corpus: str, prompts: list[dict], q: np.ndarray,
    docs: list[dict], d: np.ndarray, top_k: int,
) -> tuple[list[dict], dict]:
    sims = q @ d.T
    k = min(top_k, sims.shape[1])
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
    top_sims = np.take_along_axis(sims, top, axis=1)
    mean_sim = q @ d.mean(axis=0)

    rows = []
    for i, prompt in enumerate(prompts):
        nearest = docs[top[i, 0]]
        rows.append({
            "set": set_name,
            "corpus": corpus,
            **{key: prompt[key] for key in ("id", "text", "n_tokens", "domain", "topic", "source") if key in prompt},
            "max_sim": round(float(top_sims[i, 0]), 4),
            "topk_mean": round(float(top_sims[i].mean()), 4),
            "mean_sim": round(float(mean_sim[i]), 4),
            "nearest_source": nearest["source"],
            "nearest_subsource": nearest["subsource"],
            "nearest_text": nearest["text"][:300],
            "top_docs": [
                {
                    "doc_ix": docs[j]["doc_ix"], "doc_id": docs[j]["doc_id"],
                    "source": docs[j]["source"], "subsource": docs[j]["subsource"],
                    "sim": round(float(s), 4),
                }
                for j, s in zip(top[i], top_sims[i])
            ],
        })
    metrics = {"max_sim": top_sims[:, 0], "topk_mean": top_sims.mean(axis=1), "mean_sim": mean_sim}
    return rows, metrics


def source_table(rows: list[dict], docs: list[dict], key: str) -> list[dict]:
    """Nearest-doc share of each source next to its share in the doc sample."""
    nearest = Counter(row[f"nearest_{key}"] for row in rows)
    sample = Counter(doc[key] for doc in docs)
    table = []
    for source, count in nearest.most_common():
        share, sample_share = count / len(rows), sample[source] / len(docs)
        table.append({
            key: source or "-",
            "nearest_count": count,
            "nearest_share": round(share, 4),
            "sample_share": round(sample_share, 4),
            "enrichment": round(share / sample_share, 2) if sample_share else None,
        })
    return table


def describe(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    return {
        "mean": round(float(values.mean()), 4),
        "ci95": bootstrap_mean_ci(values, rng, n_boot),
        "median": round(float(np.median(values)), 4),
        "std": round(float(values.std()), 4),
    }


def summarize_markdown(summary: dict) -> str:
    cfg = summary["config"]
    title = "prompt sets vs Tatoeba" if cfg.get("compare") is None else ", ".join(
        pair.replace(":", " vs ") for pair in cfg["compare"])
    lines = [
        f"# Embedding similarity: {title} to training docs",
        "",
        f"Model `{cfg['model']}`; {cfg['num_docs']} random docs per corpus (first {cfg['doc_max_tokens']} "
        f"tokens embedded); top-k = {cfg['top_k']}; query instruction: `{cfg['instruction'] or '(none)'}`.",
        "`train` rows are sentences from other random docs of the same corpus (an in-distribution reference).",
        "",
    ]
    for corpus, block in summary["corpora"].items():
        lines += [f"## {corpus}", "", "| set | n | max_sim mean [95% CI] | median | topk_mean | mean_sim |",
                  "|---|---|---|---|---|---|"]
        for set_name, s in block["sets"].items():
            m = s["max_sim"]
            lines.append(
                f"| {set_name} | {s['n']} | {m['mean']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] | "
                f"{m['median']:.4f} | {s['topk_mean']['mean']:.4f} | {s['mean_sim']['mean']:.4f} |"
            )
        for comp in block["comparisons"]:
            lines += ["", f"**{comp['a']} - {comp['b']}**", "", "| metric | diff [95% CI] | AUC |", "|---|---|---|"]
            for metric, c in comp["metrics"].items():
                lines.append(f"| {metric} | {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['auc']:.3f} |")
        bins = [length_bin(lo) for lo, _ in LENGTH_BINS]
        lines += ["", "max_sim by Llama-2 token length (mean, n):", "",
                  "| set | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
        for set_name, s in block["sets"].items():
            cells = [
                f"{s['by_length'][b]['max_sim_mean']:.4f} ({s['by_length'][b]['n']})" if b in s["by_length"] else "-"
                for b in bins
            ]
            lines.append(f"| {set_name} | " + " | ".join(cells) + " |")
        for set_name, s in block["sets"].items():
            if "by_domain" in s:
                lines += ["", f"{set_name} max_sim by domain (domain or source): " + ", ".join(
                    f"{dom} {v['max_sim_mean']:.4f} (n={v['n']})" for dom, v in s["by_domain"].items())]
        for key in ("source", "subsource"):
            if not any(s[f"nearest_{key}"] and s[f"nearest_{key}"][0][key] != "-" for s in block["sets"].values()):
                continue
            lines += ["", f"Nearest-doc {key} (share among nearest docs / share in sample = enrichment), top 8:", ""]
            for set_name, s in block["sets"].items():
                cells = [
                    f"{r[key]} {r['nearest_share']:.1%}/{r['sample_share']:.1%}={r['enrichment']}x"
                    for r in s[f"nearest_{key}"][:8]
                ]
                lines.append(f"- {set_name}: " + "; ".join(cells))
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=None,
                        help=f"Prompt sets to embed: {', '.join(PROMPT_SETS)} or names from --extra-sets "
                             "(default: the Tatoeba samples).")
    parser.add_argument("--extra-sets", nargs="+", default=[], metavar="NAME=PATH:LANG:KIND",
                        help="More prompt sets, as name=path:lang:kind (path relative to the repository root, "
                             "lang eng or dan). Each row needs text and n_tokens.")
    parser.add_argument("--compare", nargs="+", default=None, metavar="A:B",
                        help="Set pairs to compare (A - B difference, CI, AUC) in every corpus where both are "
                             "embedded. Default: every other set vs the Tatoeba set of its language.")
    parser.add_argument("--summary-name", default="summary",
                        help="Stem of the summary files in the run folder (default summary).")
    parser.add_argument("--corpora", nargs="+", choices=list(CORPORA), default=list(DEFAULT_CORPORA))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-docs", type=int, default=10_000, help="Random docs per corpus (default 10000).")
    parser.add_argument("--num-train-sentences", type=int, default=2_000,
                        help="Baseline sentences per corpus, from other random docs (default 2000).")
    parser.add_argument("--min-doc-tokens", type=int, default=32,
                        help="Skip docs shorter than this many Llama-2 tokens (default 32).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-model", default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer of the indexes (default meta-llama/Llama-2-7b-hf).")
    parser.add_argument("--sample-threads", type=int, default=16,
                        help="Threads fetching docs from the index (default 16).")
    parser.add_argument("--sample-only", action="store_true", help="Only sample docs (no torch needed).")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                        help="Task instruction for the query side (prompts and training sentences); "
                             "'' embeds queries without one. Docs never get one.")
    parser.add_argument("--doc-max-tokens", type=int, default=512,
                        help="Embed the first N model tokens of each doc (default 512).")
    parser.add_argument("--query-max-tokens", type=int, default=256)
    parser.add_argument("--pooling", choices=("last", "cls", "mean"), default=None,
                        help="Override the model's pooling. vLLM default: the model's sentence-transformers "
                             "config (Qwen3-Embedding: last, bge-m3: cls); transformers default: last.")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm",
                        help="vllm (GPU, default) or transformers (e.g. CPU smoke tests).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM only.")
    parser.add_argument("--batch-size", type=int, default=16, help="transformers only; vLLM batches itself.")
    parser.add_argument("--device", default=None, help="transformers only. Default: cuda if available, else cpu.")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto",
                        help="auto = bfloat16 (transformers on CPU: float32).")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: use only the first N prompts / training sentences per set.")
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="Smoke test: use only the first N sampled docs per corpus.")
    return parser


def add_extra_sets(specs: list[str]) -> None:
    for spec in specs:
        name, rest = spec.split("=", 1)
        path, lang, kind = rest.rsplit(":", 2)
        if lang not in CORPORA_BY_LANG:
            raise ValueError(f"--extra-sets {spec}: lang must be one of {list(CORPORA_BY_LANG)}")
        PROMPT_SETS[name] = (str((REPO_ROOT / path).resolve()), lang, kind)


def main() -> None:
    args = build_arg_parser().parse_args()
    add_extra_sets(args.extra_sets)
    args.sets = args.sets or [s for s in PROMPT_SETS if not any(s == spec.split("=", 1)[0] for spec in args.extra_sets)]
    unknown = [s for s in args.sets if s not in PROMPT_SETS]
    if unknown:
        raise SystemExit(f"Unknown prompt sets {unknown}; known: {list(PROMPT_SETS)}")
    corpora = [c for c in args.corpora if any(PROMPT_SETS[s][1] in l for s in args.sets
                                                for l, cs in CORPORA_BY_LANG.items() if c in cs)]
    suffix = f"n{args.num_docs}_t{args.num_train_sentences}_min{args.min_doc_tokens}_len{args.doc_max_tokens}_seed{args.seed}"
    docs_paths = {c: args.output_dir / "docs" / f"docs_{c}_{suffix}.jsonl" for c in corpora}
    sentence_paths = {c: args.output_dir / "docs" / f"train_sentences_{c}_{suffix}.jsonl" for c in corpora}

    for corpus in corpora:
        if docs_paths[corpus].exists() and sentence_paths[corpus].exists():
            print(f"{corpus}: using cached sample {docs_paths[corpus]}")
            continue
        start = time.time()
        sample_corpus(corpus, args, docs_paths[corpus], sentence_paths[corpus])
        print(f"{corpus}: sampled in {time.time() - start:.0f}s -> {docs_paths[corpus]}")
    if args.sample_only:
        return

    model_slug = args.model.rstrip("/").split("/")[-1]
    cache_dir = args.output_dir / "embeddings" / model_slug
    run_name = model_slug + (f"_limit{args.limit}" if args.limit else "") + (
        f"_docs{args.limit_docs}" if args.limit_docs else "")
    run_dir = args.output_dir / run_name
    embedder = None

    def get_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = Embedder(args)
        return embedder

    query_prefix = f"Instruct: {args.instruction}\nQuery:" if args.instruction else ""
    prompts = {s: read_jsonl(DATA_DIR / PROMPT_SETS[s][0])[: args.limit] for s in args.sets}
    rng = np.random.default_rng(args.seed)
    summary = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "corpora": {}}

    for corpus in corpora:
        docs = read_jsonl(docs_paths[corpus])
        doc_emb = cached_embeddings(
            get_embedder, cache_dir, f"docs_{corpus}_{suffix}",
            [doc["text"] for doc in docs], args.doc_max_tokens, "",
        )[: args.limit_docs]
        docs = docs[: args.limit_docs]
        lang = next(l for l, cs in CORPORA_BY_LANG.items() if corpus in cs)
        queries = {s: prompts[s] for s in args.sets if PROMPT_SETS[s][1] == lang}
        queries[f"train_{corpus}"] = read_jsonl(sentence_paths[corpus])[: args.limit]

        block = {"num_docs": len(docs), "sets": {}, "comparisons": []}
        metrics_by_set = {}
        for set_name, items in queries.items():
            q_emb = cached_embeddings(
                get_embedder, cache_dir, set_name, [p["text"] for p in items],
                args.query_max_tokens, query_prefix,
            )
            rows, metrics = analyze_pair(set_name, corpus, items, q_emb, docs, doc_emb, args.top_k)
            write_jsonl(run_dir / f"per_prompt_{set_name}_{corpus}.jsonl", rows)
            metrics_by_set[set_name] = metrics

            by_length, by_domain = {}, {}
            for row in rows:
                by_length.setdefault(length_bin(row["n_tokens"]), []).append(row["max_sim"])
                if "domain" in row:
                    by_domain.setdefault(row["domain"], []).append(row["max_sim"])
            block["sets"][set_name] = {
                "n": len(rows),
                **{name: describe(values, rng, args.n_bootstrap) for name, values in metrics.items()},
                "avg_n_tokens": round(float(np.mean([row["n_tokens"] for row in rows])), 2),
                "by_length": {
                    b: {"n": len(v), "max_sim_mean": round(float(np.mean(v)), 4)}
                    for b, v in sorted(by_length.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))
                },
                **({"by_domain": {
                    dom: {"n": len(v), "max_sim_mean": round(float(np.mean(v)), 4)} for dom, v in sorted(by_domain.items())
                }} if by_domain else {}),
                "nearest_source": source_table(rows, docs, "source"),
                "nearest_subsource": source_table(rows, docs, "subsource"),
            }

        if args.compare:
            pairs = [tuple(pair.split(":", 1)) for pair in args.compare]
        else:
            tatoeba = [s for s in queries if s in PROMPT_SETS and PROMPT_SETS[s][2] == "tatoeba"]
            pairs = [(s, t) for t in tatoeba for s in queries if s in PROMPT_SETS and s not in tatoeba]
        for a, b in pairs:
            if a not in metrics_by_set or b not in metrics_by_set:
                continue
            block["comparisons"].append({
                "a": a, "b": b,
                "metrics": {
                    name: {
                        "diff": round(float(metrics_by_set[a][name].mean() - metrics_by_set[b][name].mean()), 4),
                        "ci95": bootstrap_diff_ci(metrics_by_set[a][name], metrics_by_set[b][name], rng, args.n_bootstrap),
                        "auc": round(auc(metrics_by_set[a][name], metrics_by_set[b][name]), 4),
                    }
                    for name in ("max_sim", "topk_mean", "mean_sim")
                },
            })
        summary["corpora"][corpus] = block

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{args.summary_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    markdown = summarize_markdown(summary)
    (run_dir / f"{args.summary_name}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {run_dir}/{args.summary_name}.{{json,md}} and per-prompt files.")


if __name__ == "__main__":
    main()

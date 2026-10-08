"""Tests for the OLMoTrace reproduction.

Run from the repository root (uses 00_prepare_data/dummy_index):

    python -m unittest discover -s olmotrace/tests -v
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OLMOTRACE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(OLMOTRACE_DIR)
sys.path.insert(0, OLMOTRACE_DIR)

import bm25  # noqa: E402
import olmo_trace as ot  # noqa: E402

DUMMY_INDEX = os.path.join(REPO_ROOT, "00_prepare_data", "dummy_index")
DUMMY_DATASET = os.path.join(REPO_ROOT, "00_prepare_data", "dummy_dataset", "dummy.jsonl")
DUMMY_UNIGRAMS = os.path.join(REPO_ROOT, "02_unigram_probs", "unigram_probs_dummy.json")
# A Python with the real rank_bm25 package installed, for the reference test (skipped if unset).
REFERENCE_RANK_BM25_PYTHON = os.environ.get("RANK_BM25_PYTHON", "")


def _dummy_docs() -> list[str]:
    with open(DUMMY_DATASET) as f:
        return [json.loads(line)["text"] for line in f]


def _test_texts() -> list[str]:
    """Hand-written and randomly spliced texts mixing indexed and novel content."""
    docs = _dummy_docs()
    texts = [
        "Space exploration has expanded our understanding of the ocean. "
        "Music can influence mood and emotional health.\n"
        "Regular exercise helps maintain physical and mental well-being.",
        "I think that the sun rises in the east and sets in the west. That is all.",
        "The coffee shop opened early to serve commuters, and the chef experimented with spices.",
        docs[8],
        "zzqx florp blarg.",
    ]
    rng = random.Random(1234)
    separators = [" ", ". ", "\n", ", ", " and ", " 1962 "]
    for _ in range(25):
        parts = []
        for _ in range(rng.randint(2, 4)):
            words = rng.choice(docs).split()
            lo = rng.randint(0, max(0, len(words) - 2))
            hi = rng.randint(lo + 1, len(words))
            parts.append(" ".join(words[lo:hi]))
        texts.append(rng.choice(separators).join(parts))
    return texts


class _EngineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enc = ot.load_tokenizer()
        cls.enc = enc
        cls.token_info = ot.TokenInfo(enc)
        cls.engine = ot.load_engine(DUMMY_INDEX, enc.eos_token_id)
        # Two copies of the same index behave as a two-shard index.
        cls.engine_2shards = ot.load_engine([DUMMY_INDEX, DUMMY_INDEX], enc.eos_token_id)
        cls.unigram_logprobs = ot.load_unigram_logprobs(DUMMY_UNIGRAMS)

    @staticmethod
    def tok_cnts(engine):
        return [engine.engine.get_tok_cnt(s=s) for s in range(engine.engine.get_num_shards())]


class TestTokenClasses(unittest.TestCase):
    def test_begin_of_word_pieces(self):
        for piece in ["▁the", "▁", "▁.", ".", ",", "'", "<0x0A>", "<0xE2>"]:
            self.assertTrue(ot.is_begin_of_word_piece(piece), piece)
        for piece in ["ing", "t", "9", "<0x80>", "<0xBF>"]:
            self.assertFalse(ot.is_begin_of_word_piece(piece), piece)

    def test_delimiters_are_period_and_newline(self):
        enc = ot.load_tokenizer()
        info = ot.TokenInfo(enc)
        pieces = sorted(enc.convert_ids_to_tokens(sorted(info.delimiter_ids)))
        self.assertEqual(pieces, [".", "<0x0A>", "▁."])


    def test_begin_of_word_flags_after_newline(self):
        enc = ot.load_tokenizer()
        info = ot.TokenInfo(enc)
        ids = enc.encode("a\nbcd ing")
        pieces = enc.convert_ids_to_tokens(ids)
        flags = dict(zip(pieces, info.begin_of_word_flags(ids)))
        self.assertEqual(pieces[:3], ["▁a", "<0x0A>", "b"])
        self.assertTrue(flags["b"])


class TestPureFunctions(unittest.TestCase):
    def test_num_spans_to_keep_avoids_float_error(self):
        # float: 0.05 * 60 = 3.0000000000000004, whose ceil would be 4.
        self.assertEqual(ot.num_spans_to_keep(60, 0.05), 3)
        self.assertEqual(ot.num_spans_to_keep(61, 0.05), 4)
        self.assertEqual(ot.num_spans_to_keep(1, 0.05), 1)
        self.assertEqual(ot.num_spans_to_keep(0, 0.05), 0)

    def test_suppress_nonmaximal_spans(self):
        spans = [(0, 5), (1, 5), (2, 7), (3, 6), (4, 9), (8, 9)]
        self.assertEqual(ot.suppress_nonmaximal_spans(spans), [(0, 5), (2, 7), (4, 9)])

    def test_merge_overlapping_spans(self):
        merged = ot.merge_overlapping_spans([(0, 5), (3, 8), (8, 10), (12, 15), (13, 14)])
        self.assertEqual(merged, [(0, 8, [0, 1]), (8, 10, [2]), (12, 15, [3, 4])])

    def test_relevance_levels(self):
        self.assertEqual(ot.relevance_level(0.7), "high")
        self.assertEqual(ot.relevance_level(0.69), "medium")
        self.assertEqual(ot.relevance_level(0.5), "medium")
        self.assertEqual(ot.relevance_level(0.49), "low")

    def test_sample_occurrences(self):
        segments = [(10, 13), (0, 0), (5, 7)]
        rng = random.Random(0)
        self.assertEqual(
            ot.sample_occurrences(segments, 10, rng),
            [(0, 10), (0, 11), (0, 12), (2, 5), (2, 6)],
        )
        big = [(0, 1_000_000), (0, 1_000_000)]
        picked = ot.sample_occurrences(big, 10, rng)
        self.assertEqual(len(picked), 10)
        self.assertEqual(len(set(picked)), 10)
        self.assertTrue(all(0 <= r < 1_000_000 for _, r in picked))

    def test_merge_documents_groups_snippets_by_document(self):
        def snip(doc_ix, text):
            return {"shard": 0, "doc_ix": doc_ix, "doc_len": 9, "id": str(doc_ix),
                    "metadata": {}, "snippet": text, "context": text}
        docs = ot.merge_documents([[snip(1, "a"), snip(2, "b")], [snip(1, "c")]])
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["filtered_span_indices"], [0, 1])
        self.assertEqual([s["snippet"] for s in docs[0]["snippets"]], ["a", "c"])


class TestBM25(unittest.TestCase):
    CORPUS = [
        "the space needle was built for the 1962 world fair in seattle",
        "the world fair of 1962 attracted millions of visitors",
        "a recipe for sourdough bread with a long fermentation",
        "",
        "seattle seattle seattle space",
    ]
    QUERY = "when was the space needle built? it was built for the 1962 world fair"

    def _local_scores(self):
        corpus = [ot.bm25_tokenize(d) for d in self.CORPUS]
        return list(bm25._BM25Okapi(corpus).get_scores(ot.bm25_tokenize(self.QUERY)))

    def test_matches_reference_rank_bm25(self):
        if not REFERENCE_RANK_BM25_PYTHON or not os.path.exists(REFERENCE_RANK_BM25_PYTHON):
            self.skipTest("set RANK_BM25_PYTHON to a Python with rank_bm25 installed")
        script = (
            "import json, sys, re\n"
            "from rank_bm25 import BM25Okapi\n"
            "c, q = json.load(sys.stdin)\n"
            "tok = lambda t: re.findall(r'\\w+', t.lower())\n"
            "print(json.dumps(list(BM25Okapi([tok(d) for d in c]).get_scores(tok(q)))))\n"
        )
        out = subprocess.run(
            [REFERENCE_RANK_BM25_PYTHON, "-c", script],
            input=json.dumps([self.CORPUS, self.QUERY]),
            capture_output=True, text=True, check=True,
        )
        reference = json.loads(out.stdout)
        for ours, ref in zip(self._local_scores(), reference):
            self.assertAlmostEqual(ours, ref, places=12)

    def test_ranks_relevant_documents_first(self):
        scores = self._local_scores()
        self.assertEqual(max(range(len(scores)), key=scores.__getitem__), 0)


class TestLongestPrefix(_EngineTestCase):
    def _brute_force_longest_prefix(self, engine, s):
        n = 0
        while n < len(s) and engine.find(input_ids=s[: n + 1])["cnt"] > 0:
            n += 1
        return n

    def test_single_find_matches_brute_force(self):
        for engine in (self.engine, self.engine_2shards):
            tok_cnts = self.tok_cnts(engine)
            for text in _test_texts():
                ids = self.enc.encode(text)
                for b in range(len(ids)):
                    with self.subTest(text=text[:40], b=b, shards=len(tok_cnts)):
                        self.assertEqual(
                            ot.get_longest_prefix_len(engine, ids[b:], tok_cnts),
                            self._brute_force_longest_prefix(engine, ids[b:]),
                        )

    def test_suffix_array_boundaries(self):
        tok_cnts = self.tok_cnts(self.engine)
        # Sorts after every indexed suffix: neighbours include separator ranks.
        self.assertEqual(ot.get_longest_prefix_len(self.engine, [31999, 31999], tok_cnts), 0)
        # Sorts before every indexed suffix.
        self.assertEqual(ot.get_longest_prefix_len(self.engine, [3, 3], tok_cnts), 0)


class TestMaximalSpans(_EngineTestCase):
    def _brute_force_maximal_spans(self, ids):
        """Enumerate all spans and apply the paper's step-1 definition directly."""
        info = self.token_info
        is_bow = info.begin_of_word_flags(ids)
        L = len(ids)
        valid = []
        for b in range(L):
            if not is_bow[b]:
                continue
            for e in range(b + 1, L + 1):
                if e < L and not is_bow[e]:
                    continue
                if any(t in info.delimiter_ids for t in ids[b:e - 1]):
                    continue
                if self.engine.find(input_ids=ids[b:e])["cnt"] > 0:
                    valid.append((b, e))
        return sorted(
            (b, e) for b, e in valid
            if not any(b2 <= b and e <= e2 and (b2, e2) != (b, e) for b2, e2 in valid)
        )

    def test_algorithm_matches_definition(self):
        tok_cnts = self.tok_cnts(self.engine)
        for text in _test_texts():
            ids = self.enc.encode(text)
            with self.subTest(text=text[:60]):
                self.assertEqual(
                    ot.get_maximal_matching_spans(self.engine, ids, self.token_info, tok_cnts),
                    self._brute_force_maximal_spans(ids),
                )

    def test_parallel_matches_serial(self):
        from concurrent.futures import ThreadPoolExecutor
        tok_cnts = self.tok_cnts(self.engine)
        with ThreadPoolExecutor(8) as ex:
            for text in _test_texts():
                ids = self.enc.encode(text)
                self.assertEqual(
                    ot.get_maximal_matching_spans(self.engine, ids, self.token_info, tok_cnts, ex),
                    ot.get_maximal_matching_spans(self.engine, ids, self.token_info, tok_cnts),
                )

    def test_spans_stop_after_period(self):
        text = "Space exploration has expanded our understanding of the universe. Music can influence mood"
        ids = self.enc.encode(text)
        spans = ot.get_maximal_matching_spans(self.engine, ids, self.token_info, self.tok_cnts(self.engine))
        decoded = [self.enc.decode(ids[b:e]) for b, e in spans]
        self.assertIn("Space exploration has expanded our understanding of the universe.", decoded)
        self.assertIn("Music can influence mood", decoded)


class TestTraceGeneration(_EngineTestCase):
    def _trace(self, text, prompt=None, index=0, config=None):
        return ot.trace_generation(
            text, self.engine, self.token_info, self.unigram_logprobs,
            config or ot.TraceConfig(), prompt=prompt, index=index,
        )

    def test_end_to_end_on_indexed_text(self):
        text = ("Space exploration has expanded our understanding of the universe. "
                "Music can influence mood and emotional well-being.")
        result = self._trace(text, prompt="Tell me about space and music.")
        L = result["num_tokens"]
        self.assertEqual(len(result["filtered_spans"]), ot.num_spans_to_keep(L, 0.05))
        self.assertTrue(result["spans"])
        for sp in result["spans"]:
            self.assertIn(sp["relevance"], ot.RELEVANCE_LEVELS)
            for rank in sp["doc_ranks"]:
                doc = result["documents"][rank]
                self.assertTrue(any(sp["text"] in s["context"] for s in doc["snippets"]))
        scores = [d["bm25_score"] for d in result["documents"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for doc in result["documents"]:
            for snip in doc["snippets"]:
                span = result["filtered_spans"][snip["filtered_span_index"]]
                self.assertIn(span["text"], snip["snippet"])
                self.assertIn(snip["snippet"], snip["context"])

    def test_span_relevance_is_max_over_enclosing_documents(self):
        text = "The sun rises in the east and sets in the west. Clear communication is essential for effective teamwork."
        result = self._trace(text, config=ot.TraceConfig(span_fraction=0.5))
        for sp in result["spans"]:
            best = max(result["documents"][r]["relevance_score"] for r in sp["doc_ranks"])
            self.assertEqual(sp["relevance_score"], best)

    def test_novel_text_has_no_spans(self):
        result = self._trace("zzqx florp blarg quux")
        self.assertEqual(result["spans"], [])
        self.assertEqual(result["documents"], [])

    def test_sampling_is_deterministic_per_seed(self):
        text = "The sun rises in the east and sets in the west."
        cfg = ot.TraceConfig(docs_per_span=1, seed=7)
        a = self._trace(text, config=cfg, index=3)
        b = self._trace(text, config=cfg, index=3)
        self.assertEqual(
            [(d["shard"], d["doc_ix"]) for d in a["documents"]],
            [(d["shard"], d["doc_ix"]) for d in b["documents"]],
        )


class TestCli(unittest.TestCase):
    def test_cli_on_dummy_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.jsonl")
            summary_path = os.path.join(tmp, "summary.json")
            subprocess.run(
                [sys.executable, os.path.join(OLMOTRACE_DIR, "olmo_trace.py"),
                 "--dataset", "dummy",
                 "--index-dir", DUMMY_INDEX,
                 "--unigram-probs-path", DUMMY_UNIGRAMS,
                 "--num-workers", "2",
                 "--docs-per-span", "10",
                 "--results-output", results_path,
                 "--summary-output", summary_path],
                check=True, capture_output=True, text=True, cwd=REPO_ROOT,
            )
            with open(results_path) as f:
                rows = [json.loads(line) for line in f]
            with open(summary_path) as f:
                summary = json.load(f)
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["index"] for r in rows], list(range(8)))
        self.assertEqual(summary["total_generations"], 8)
        self.assertEqual(summary["total_filtered_spans"], sum(len(r["filtered_spans"]) for r in rows))
        self.assertGreater(summary["generations_with_spans"], 0)


if __name__ == "__main__":
    unittest.main()

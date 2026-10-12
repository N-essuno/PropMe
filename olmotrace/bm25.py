"""BM25 scorer used by OLMoTrace step 5.

The paper (footnote 4) uses the implementation in
https://github.com/dorianbrown/rank_bm25. When that package is installed we use
its ``BM25Okapi`` directly. Otherwise we fall back to ``_BM25Okapi`` below, a
re-implementation of rank_bm25 0.2.2's ``BM25Okapi`` (same k1, b, epsilon
defaults, same epsilon floor for negative idf values). The equivalence is
checked in ``olmotrace/tests/test_olmo_trace.py``.
"""

import math

import numpy as np


class _BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.doc_len: list[int] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}

        nd: dict[str, int] = {}  # word -> number of documents containing it
        num_tokens = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_tokens += len(document)
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
            self.corpus_size += 1
        self.avgdl = num_tokens / self.corpus_size

        # idf = log((N - n + 0.5) / (n + 0.5)); negative values are floored to
        # epsilon * average_idf, exactly as rank_bm25 does.
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score


try:
    from rank_bm25 import BM25Okapi  # the implementation cited by the paper
    BM25_BACKEND = "rank_bm25"
except ImportError:
    BM25Okapi = _BM25Okapi
    BM25_BACKEND = "builtin_rank_bm25_port"

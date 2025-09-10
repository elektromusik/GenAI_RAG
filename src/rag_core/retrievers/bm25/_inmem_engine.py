import math
from collections import Counter
from typing import Iterable


class InMemoryBM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b:float = 0.75):
        self.docs = docs                     # id -> text
        self.N = len(docs)
        self.tfs: dict[str, Counter] = {}    # id -> Counter(term -> tf)
        self.doc_len : dict[str, int] = {}
        df: Counter = Counter()

        for doc_id, text in docs.items():
            tokens = self._analyze(text)    # TODO: Real tokenization
            tf = Counter(tokens)
            self.tfs[doc_id] = tf
            self.doc_len[doc_id] = len(tokens)
            for term in tf:
                df[term] += 1

        self.avgdl = sum(self.doc_len.values()) / max(1, self.N)
        self.idf = {t: math.log((self.N - df[t] + 0.5) / (df[t] + 0.5) + 1e-12) for t in df}
        self.k1, self.b = k1, b

    def _analyze(self, text: str) -> list[str]:
        return [tok for tok in text.lower().split()]    # Minimal; replace by true analyzer

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q_terms = self._analyze(query)
        scores: dict[str, float] = {}

        for doc_id, tf in self.tfs.items():
            dl = self.doc_len[doc_id]
            score = 0.0
            for t in q_terms:
                if t not in tf or t not in self.idf:
                    continue
                idf = self.idf[t]
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (frq * (self.k1 + 1)) / (denom + 1e-12)
            if score != 0.0:
                scores[doc_id] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

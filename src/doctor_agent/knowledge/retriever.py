"""CPU-only retriever. Content is owned by knowledge-rag.

v0: minimal BM25 with no external dependencies. Replace or extend it once the KB (data/kb/) is built.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[\w가-힣]+")


def _tok(s: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(s)]


class BM25Retriever:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tfs = [Counter(_tok(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(docs)) if docs else 0.0
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = _tok(query)
        scores = []
        for doc, tf, dl in zip(self.docs, self.tfs, self.lens):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((doc, s))
        return sorted((x for x in scores if x[1] > 0), key=lambda x: -x[1])[:k]

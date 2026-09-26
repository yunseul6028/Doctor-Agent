import re

_NON_WORD = re.compile(r"[^0-9a-z가-힣]")
SIMILAR = 0.7  # char-bigram Jaccard above which two actions count as the same request


def _bigrams(text: str) -> set[str]:
    t = _NON_WORD.sub("", text.lower())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(a: str, b: str) -> float:
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x and y else float(a.strip().lower() == b.strip().lower())

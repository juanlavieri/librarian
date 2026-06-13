"""Deterministic, dependency-free hashing embedder (the default).

Implements the hashing trick: tokens (unigrams + bigrams) are hashed into a
fixed-dimensional vector with sublinear term weighting and L2 normalization.
This is *not* a neural embedding -- it captures lexical/semantic overlap rather
than deep meaning -- but it requires no model, no API key, and no network, so
the Librarian works the instant it is installed. Swap in ``OpenAIEmbedder`` (or
any custom :class:`Embedder`) for production-grade semantics.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9_\-]*\b")


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        words = [w.lower() for w in _TOKEN_RE.findall(text or "")]
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts: dict = {}
        for tok in self._tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sublinear term-frequency weighting damps very frequent tokens.
            vec[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]

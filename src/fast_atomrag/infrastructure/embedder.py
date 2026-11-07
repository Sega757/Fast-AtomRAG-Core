"""Deterministic hashing embedder (word + char n-grams, signed feature hashing).

No model download, identical vectors across processes and restarts (blake2b, not hash()).
Char n-grams give partial robustness to Russian inflection. Replaceable via the Embedder port.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator, Sequence

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    def __init__(self, dim: int = 512, ngram_min: int = 3, ngram_max: int = 5) -> None:
        if dim < 64:
            raise ValueError("dim must be >= 64")
        if not 1 <= ngram_min <= ngram_max <= 8:
            raise ValueError("require 1 <= ngram_min <= ngram_max <= 8")
        self._dim = dim
        self._nmin = ngram_min
        self._nmax = ngram_max
        self._id = f"hash-w1-c{ngram_min}{ngram_max}-v1-d{dim}"

    @property
    def embedder_id(self) -> str:
        return self._id

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> Iterator[str]:
        norm = unicodedata.normalize("NFKC", text).casefold()
        for word in _WORD.findall(norm):
            yield "w:" + word
            padded = f"<{word}>"
            for n in range(self._nmin, self._nmax + 1):
                for i in range(len(padded) - n + 1):
                    yield "c:" + padded[i : i + n]

    def embed(self, text: str) -> np.ndarray:
        acc: dict[int, float] = {}
        for feature in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
            idx = h % self._dim
            acc[idx] = acc.get(idx, 0.0) + (1.0 if (h >> 63) & 1 == 0 else -1.0)
        vec = np.zeros(self._dim, dtype=np.float32)
        if acc:
            vec[list(acc)] = list(acc.values())
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row] = self.embed(text)
        return out

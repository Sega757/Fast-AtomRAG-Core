"""Exact cosine index on a contiguous float32 matrix (BLAS matmul + argpartition).

Exact and dependency-free. Replace with an ANN adapter (HNSW) behind the same port when the
measured p95 exceeds budget (see bench/ and docs/ADR.md, ADR-002).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class ExactVectorIndex:
    def __init__(self, dim: int, initial_capacity: int = 1024) -> None:
        if dim <= 0 or initial_capacity <= 0:
            raise ValueError("dim and initial_capacity must be > 0")
        self._dim = dim
        self._matrix = np.zeros((initial_capacity, dim), dtype=np.float32)
        self._ids = np.zeros(initial_capacity, dtype=np.int64)
        self._row: dict[int, int] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _prepare(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self._dim,):
            raise ValueError(f"expected vector of shape ({self._dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector contains non-finite values")
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _queries(self, queries: np.ndarray) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self._dim:
            raise ValueError(f"queries must have shape (m, {self._dim})")
        return q

    def upsert(self, atom_id: int, vector: np.ndarray) -> None:
        vec = self._prepare(vector)
        row = self._row.get(atom_id)
        if row is None:
            if self._size == self._matrix.shape[0]:
                grow = self._matrix.shape[0] * 2
                self._matrix = np.resize(self._matrix, (grow, self._dim))
                self._ids = np.resize(self._ids, grow)
            row = self._size
            self._size += 1
            self._row[atom_id] = row
            self._ids[row] = atom_id
        self._matrix[row] = vec

    def remove(self, atom_id: int) -> bool:
        row = self._row.pop(atom_id, None)
        if row is None:
            return False
        last = self._size - 1
        if row != last:
            self._matrix[row] = self._matrix[last]
            moved = int(self._ids[last])
            self._ids[row] = moved
            self._row[moved] = row
        self._size = last
        return True

    def search(self, queries: np.ndarray, k: int) -> dict[int, float]:
        """Top-k by max cosine over the query vectors."""
        if self._size == 0 or k <= 0:
            return {}
        sims = self._matrix[: self._size] @ self._queries(queries).T
        best = sims.max(axis=1)
        k = min(k, self._size)
        top = np.argpartition(-best, k - 1)[:k]
        return {int(self._ids[i]): float(best[i]) for i in top}

    def score(self, queries: np.ndarray, atom_ids: Iterable[int]) -> dict[int, float]:
        pairs = [(a, self._row[a]) for a in atom_ids if a in self._row]
        if not pairs:
            return {}
        rows = np.fromiter((r for _, r in pairs), dtype=np.int64, count=len(pairs))
        best = (self._matrix[rows] @ self._queries(queries).T).max(axis=1)
        return {a: float(s) for (a, _), s in zip(pairs, best, strict=False)}

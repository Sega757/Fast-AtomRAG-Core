"""Ports owned by the application layer; infrastructure provides the adapters."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..domain.models import Atom


@dataclass(frozen=True, slots=True)
class StoredAtom:
    atom: Atom
    embedder_id: str
    vector: np.ndarray


class Embedder(Protocol):
    @property
    def embedder_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: Sequence[str]) -> np.ndarray: ...


class VectorIndex(Protocol):
    def upsert(self, atom_id: int, vector: np.ndarray) -> None: ...

    def remove(self, atom_id: int) -> bool: ...

    def search(self, queries: np.ndarray, k: int) -> dict[int, float]: ...

    def score(self, queries: np.ndarray, atom_ids: Iterable[int]) -> dict[int, float]: ...

    def __len__(self) -> int: ...


class AtomRepository(Protocol):
    def upsert(self, atom: Atom, embedder_id: str, vector: np.ndarray) -> bool: ...

    def upsert_many(self, items: Sequence[tuple[Atom, np.ndarray]], embedder_id: str) -> None: ...

    def delete(self, atom_id: int) -> bool: ...

    def load_all(self) -> Iterator[StoredAtom]: ...

    def close(self) -> None: ...

from __future__ import annotations

import pytest

from fast_atomrag.application.service import AtomRAGService, SearchConfig
from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.infrastructure.embedder import HashingEmbedder
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository
from fast_atomrag.infrastructure.vector_index import ExactVectorIndex


def make_service(db_path: str, *, dim: int = 256, **config: object) -> AtomRAGService:
    embedder = HashingEmbedder(dim=dim)
    service = AtomRAGService(
        repository=SQLiteAtomRepository(db_path),
        embedder=embedder,
        index=ExactVectorIndex(dim=embedder.dim, initial_capacity=4),
        graph=AtomEntityGraph(),
        config=SearchConfig(**config),  # type: ignore[arg-type]
    )
    service.load()
    return service


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "atoms.db")

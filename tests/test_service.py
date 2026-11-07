"""Use cases with real adapters (SQLite file + numpy index): the critical retrieval path."""

from __future__ import annotations

import pytest

from fast_atomrag.application.errors import AtomNotFoundError, ServiceNotReadyError
from fast_atomrag.application.service import AtomRAGService, SearchConfig, SearchMode
from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.domain.resonance import FusionMethod
from fast_atomrag.domain.router import Retrieval
from fast_atomrag.infrastructure.embedder import HashingEmbedder
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository
from fast_atomrag.infrastructure.vector_index import ExactVectorIndex

from .conftest import make_service

MULTI_HOP = [
    (1, "Компания Альфа купила стартап Бета в 2024 году.", ["Альфа", "Бета"]),
    (2, "Стартап Бета разработал протокол Гамма для шифрования.", ["Бета", "Гамма"]),
]
DISTRACTORS = [
    (100 + i, f"Компания номер {i} купила другую компанию и что-то разработала.", [f"Холдинг{i}"]) for i in range(8)
]


def _fill(service: AtomRAGService, rows) -> None:
    for atom_id, text, entities in rows:
        service.upsert_atom(atom_id, text, entities)


def test_graph_mode_surfaces_second_hop_atom_that_vector_mode_misses(db_path):
    service = make_service(db_path, candidate_k=2)
    _fill(service, MULTI_HOP + DISTRACTORS)
    query = "Что разработала компания, которую купила Альфа?"

    vector = service.search(query, top_k=2, mode=SearchMode.VECTOR)
    graph = service.search(query, top_k=2, mode=SearchMode.GRAPH)

    assert 2 not in [a.atom_id for a in vector.atoms]
    assert [a.atom_id for a in graph.atoms] == [1, 2]
    assert "Альфа" in graph.seed_entities
    assert graph.atoms[1].graph_score > 0 and graph.retrieval is Retrieval.GRAPH

    # The linear formula from the design notes loses the 2nd hop to lexical distractors.
    linear = make_service(db_path + ".linear", candidate_k=2, fusion=FusionMethod.LINEAR)
    _fill(linear, MULTI_HOP + DISTRACTORS)
    assert 2 not in [a.atom_id for a in linear.search(query, top_k=2, mode=SearchMode.GRAPH).atoms]


def test_state_survives_restart_and_delete_is_durable(db_path):
    service = make_service(db_path)
    _fill(service, MULTI_HOP)
    service.delete_atom(1)
    with pytest.raises(AtomNotFoundError):
        service.delete_atom(1)
    service.close()

    restarted = make_service(db_path)
    assert restarted.stats()["atoms"] == 1
    assert restarted.get_atom(2).entities == ("Бета", "Гамма")
    hits = restarted.search("Альфа Бета", top_k=5)
    assert [a.atom_id for a in hits.atoms] == [2]


def test_upsert_is_idempotent_and_replaces_graph_edges(db_path):
    service = make_service(db_path)
    assert service.upsert_atom(7, "Старый факт про X", ["X"]) is True
    assert service.upsert_atom(7, "Новый факт про Y", ["Y"]) is False
    res = service.search("Что известно про X?", mode=SearchMode.GRAPH)
    assert res.seed_entities == ()
    assert service.get_atom(7).text == "Новый факт про Y"


def test_embedder_change_triggers_reembedding_on_load(db_path):
    service = make_service(db_path, dim=256)
    _fill(service, MULTI_HOP)
    service.close()
    migrated = make_service(db_path, dim=128)
    assert migrated.stats()["embedder"].endswith("d128")
    assert migrated.search("протокол Гамма", top_k=1).atoms[0].atom_id == 2


def test_auto_mode_routes_by_complexity(db_path):
    service = make_service(db_path)
    _fill(service, MULTI_HOP + DISTRACTORS)
    simple = service.search("привет", mode=SearchMode.AUTO)
    multi = service.search("Альфа и Бета: что связывает?", mode=SearchMode.AUTO)
    assert simple.route is not None and simple.route.complexity == 0
    assert simple.retrieval is Retrieval.VECTOR and len(simple.atoms) <= 2
    assert multi.route is not None and multi.route.complexity == 2
    assert multi.retrieval is Retrieval.GRAPH


def test_requests_before_load_are_rejected(db_path):
    embedder = HashingEmbedder(dim=128)
    service = AtomRAGService(
        SQLiteAtomRepository(db_path), embedder, ExactVectorIndex(128), AtomEntityGraph(), SearchConfig()
    )
    with pytest.raises(ServiceNotReadyError):
        service.search("x")
    with pytest.raises(ServiceNotReadyError):
        service.upsert_atom(1, "some text", [])

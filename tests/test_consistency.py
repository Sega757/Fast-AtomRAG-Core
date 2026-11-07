"""ADR-010: SQLite pragmas, commit-then-mutate ordering, no phantom edges after re-upsert."""

from __future__ import annotations

import sqlite3

import pytest

from fast_atomrag.application.errors import ServiceNotReadyError, StorageError
from fast_atomrag.application.service import SearchMode
from fast_atomrag.domain.graph import ATOM, ENTITY
from fast_atomrag.domain.ppr import personalized_pagerank
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository

from .conftest import make_service


def test_connection_pragmas(db_path):
    repo = SQLiteAtomRepository(db_path)
    assert repo.pragmas() == {"journal_mode": "wal", "synchronous": 1, "busy_timeout": 5000, "foreign_keys": 1}
    repo.close()


def test_storage_failure_on_upsert_leaves_memory_untouched(db_path):
    service = make_service(db_path)
    service.upsert_atom(1, "Старый текст про Альфа", ["Альфа"])

    real_conn = service._repo._conn
    service._repo._conn = _FailingConnection(real_conn)
    with pytest.raises(StorageError):
        service.upsert_atom(1, "Новый текст про Гамма", ["Гамма"])
    with pytest.raises(StorageError):
        service.upsert_atom(2, "Совсем новый атом", ["Дельта"])
    service._repo._conn = real_conn

    assert service.get_atom(1).text == "Старый текст про Альфа"
    assert service.stats()["atoms"] == 1 and service.stats()["entities"] == 1
    assert service.search("Гамма", mode=SearchMode.GRAPH).seed_keys == ()
    assert service.ready


def test_storage_failure_on_delete_leaves_memory_untouched(db_path):
    service = make_service(db_path)
    service.upsert_atom(1, "Факт про Альфа", ["Альфа"])
    real_conn = service._repo._conn
    service._repo._conn = _FailingConnection(real_conn)
    with pytest.raises(StorageError):
        service.delete_atom(1)
    service._repo._conn = real_conn
    assert service.get_atom(1).entities == ("Альфа",)
    assert [a.atom_id for a in service.search("Альфа").atoms] == [1]


def test_failure_after_commit_invalidates_service_and_reload_recovers(db_path, monkeypatch):
    service = make_service(db_path)

    def explode(*_a, **_k):
        raise MemoryError("simulated")

    monkeypatch.setattr(service._graph, "upsert", explode)
    with pytest.raises(MemoryError):
        service.upsert_atom(1, "Закоммичено, но не применено", ["Альфа"])
    assert not service.ready
    with pytest.raises(ServiceNotReadyError):
        service.search("Альфа")
    monkeypatch.undo()
    assert service.load() == 1  # the source of truth had the commit
    assert service.get_atom(1).text == "Закоммичено, но не применено"


def test_reupsert_replaces_text_and_edges_without_phantom_links(db_path):
    service = make_service(db_path)
    assert service.upsert_atom(5, "Утечка дескрипторов в модуле Ingress", ["утечки дескрипторов", "Ingress"])
    service.upsert_atom(6, "Ingress обслуживает внешний трафик", ["Ingress"])
    assert service.upsert_atom(5, "Лимит пула соединений", ["лимит пула"]) is False
    assert service.upsert_atom(5, "Лимит пула соединений", ["лимит пула"]) is False  # idempotent

    g = service._graph
    assert not g.has_entity(g.key("утечка дескрипторов"))
    assert g.entities_of(5) == frozenset({g.key("лимит пула")})
    assert g.neighbors((ENTITY, g.key("Ingress"))) == [(ATOM, 6)]
    ppr = personalized_pagerank({(ENTITY, g.key("Ingress")): 1.0}, g.neighbors, g.degree, eps=1e-9)
    assert (ATOM, 5) not in ppr

    service.close()
    restarted = make_service(db_path)
    assert restarted.get_atom(5).text == "Лимит пула соединений"
    assert restarted._graph.entities_of(5) == frozenset({restarted._graph.key("лимит пула")})


class _FailingConnection:
    """Proxy whose write statements fail like a broken disk; reads pass through."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, *args: object):
        if sql.lstrip().upper().startswith(("INSERT", "DELETE", "UPDATE")):
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, *args)

    def executemany(self, *_args: object):
        raise sqlite3.OperationalError("disk I/O error")

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc: object):
        return self._real.__exit__(*exc)

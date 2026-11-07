"""HTTP contract: status codes, validation, auth, metrics, no internal leakage."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fast_atomrag.api.app import create_app
from fast_atomrag.infrastructure.settings import Settings

KEY = "test-key-0123456789"


@pytest.fixture
def client(db_path):
    app = create_app(Settings(db_path=db_path, embed_dim=128, api_key=KEY))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _put(c, atom_id, text, entities, key=KEY):
    return c.put(f"/api/v1/atoms/{atom_id}", json={"text": text, "entities": entities}, headers={"X-API-Key": key})


def test_health_and_metrics(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["atoms"] == 0
    body = client.get("/metrics").text
    assert "atomrag_atoms" in body and "atomrag_search_phase_seconds" in body


def test_upsert_get_search_delete_lifecycle(client):
    assert _put(client, 1, "AtomRAG хранит факты в графе AEG.", ["AtomRAG", "AEG"]).status_code == 201
    assert _put(client, 1, "AtomRAG хранит атомы знаний в графе AEG.", ["AtomRAG", "AEG"]).status_code == 200
    assert client.get("/api/v1/atoms/1").json()["entities"] == ["AtomRAG", "AEG"]

    res = client.post("/api/v1/search", json={"query": "Как устроен AEG?", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert data["atoms"][0]["atom_id"] == 1 and data["seed_entities"] == ["AEG"]
    assert set(data["timings_ms"]) == {"analyze_ms", "embed_ms", "seeds_ms", "vector_ms", "graph_ms", "total_ms"}

    auto = client.post("/api/v1/search", json={"query": "AtomRAG и AEG", "mode": "auto"}).json()
    assert auto["route"]["sla_feasible"] is True and auto["route"]["bundle"]

    assert client.delete("/api/v1/atoms/1", headers={"X-API-Key": KEY}).status_code == 204
    assert client.delete("/api/v1/atoms/1", headers={"X-API-Key": KEY}).status_code == 404
    assert client.get("/api/v1/atoms/1").status_code == 404
    assert 'atomrag_requests_total{outcome="created",route="upsert_atom"} 1.0' in client.get("/metrics").text


def test_write_endpoints_require_api_key(client):
    assert _put(client, 1, "valid text", [], key="wrong-key-0000000000").status_code == 401
    assert client.put("/api/v1/atoms/1", json={"text": "valid text"}).status_code == 401
    assert client.delete("/api/v1/atoms/1").status_code == 401


@pytest.mark.parametrize(
    ("path_id", "payload"),
    [
        (1, {"text": "ab"}),
        (-1, {"text": "valid text"}),
        (1, {"text": "valid text", "extra": 1}),
        (1, {"text": "valid text", "entities": ["x"] * 33}),
    ],
)
def test_upsert_validation(client, path_id, payload):
    assert client.put(f"/api/v1/atoms/{path_id}", json=payload, headers={"X-API-Key": KEY}).status_code == 422


def test_search_validation(client):
    assert client.post("/api/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "   "}).status_code == 422


def test_unexpected_errors_do_not_leak_internals(client, monkeypatch):
    service = client.app.state.service

    def boom(*_a, **_k):
        raise RuntimeError("secret internal path /etc/x")

    monkeypatch.setattr(service, "search", boom)
    res = client.post("/api/v1/search", json={"query": "anything"})
    assert res.status_code == 500 and res.json() == {"detail": "internal error"}


def test_ready_and_metrics_expose_router_calibration(client):
    _put(client, 1, "AtomRAG хранит факты в графе AEG.", ["AtomRAG", "AEG"])
    for _ in range(3):
        client.post("/api/v1/search", json={"query": "Как устроен AEG?"})
    router = client.get("/health/ready").json()["router"]
    assert router["modes"]["graph"]["sample_count"] == 3 and router["sla_fallbacks"] == 0
    assert router["modes"]["graph"]["calibrated"] is False
    body = client.get("/metrics").text
    assert 'atomrag_retrieval_samples{mode="graph"} 3.0' in body
    assert "atomrag_retrieval_p95_seconds" in body and "atomrag_router_sla_fallbacks" in body


def test_storage_failure_maps_to_503_without_details(client, monkeypatch):
    from fast_atomrag.application.errors import StorageError

    service = client.app.state.service

    def broken(*_a, **_k):
        raise StorageError("storage operation failed: OperationalError")

    monkeypatch.setattr(service._repo, "upsert", broken)
    res = _put(client, 9, "valid text here", [])
    assert res.status_code == 503 and res.json() == {"detail": "storage unavailable"}
    assert client.get("/api/v1/atoms/9").status_code == 404

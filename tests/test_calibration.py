"""ADR-009: rolling latency tracker and router self-calibration."""

from __future__ import annotations

import pytest

from fast_atomrag.application.latency import LatencyTracker
from fast_atomrag.application.service import SearchMode
from fast_atomrag.domain.router import DEFAULT_RETRIEVAL_S, Retrieval

from .conftest import make_service


def test_tracker_uses_defaults_until_min_samples_then_rolling_p95():
    t = LatencyTracker(DEFAULT_RETRIEVAL_S, window=256, min_samples=10)
    for _ in range(9):
        t.record(Retrieval.GRAPH, 0.5)
    assert t.effective_s()[Retrieval.GRAPH] == DEFAULT_RETRIEVAL_S[Retrieval.GRAPH]
    t.record(Retrieval.GRAPH, 0.5)
    assert t.effective_s()[Retrieval.GRAPH] == pytest.approx(0.5)
    snap = t.snapshot()["modes"]["graph"]
    assert snap["calibrated"] and snap["sample_count"] == 10 and snap["p95_ms"] == pytest.approx(500)


def test_tracker_window_is_bounded_and_forgets_old_samples():
    t = LatencyTracker(DEFAULT_RETRIEVAL_S, window=256, min_samples=10)
    for _ in range(256):
        t.record(Retrieval.VECTOR, 1.0)
    for _ in range(256):
        t.record(Retrieval.VECTOR, 0.01)
    assert t.snapshot()["modes"]["vector"]["sample_count"] == 256
    assert t.effective_s()[Retrieval.VECTOR] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        t.record(Retrieval.VECTOR, -1)
    with pytest.raises(ValueError):
        LatencyTracker(DEFAULT_RETRIEVAL_S, window=5, min_samples=10)


def test_router_drops_graph_bundle_when_measured_p95_breaks_sla(db_path):
    service = make_service(db_path)
    service.upsert_atom(1, "Альфа связана с Бета", ["Альфа", "Бета"])
    query = "Альфа и Бета: что связывает?"

    cold = service.search(query, mode=SearchMode.AUTO)
    assert cold.route is not None and cold.route.bundle.name == "b3_graph_k5"

    for _ in range(20):  # simulate a slow graph phase: 0.5 s p95 -> 1.60 + 0.5 > 2.0 s SLA
        service.latency.record(Retrieval.GRAPH, 0.5)
    hot = service.search(query, mode=SearchMode.AUTO)
    assert hot.route is not None and hot.route.bundle.retrieval is Retrieval.VECTOR
    assert hot.route.latencies_s["b3_graph_k5"] > 2.0


def test_sla_fallback_counter_and_real_samples(db_path):
    service = make_service(db_path, latency_min_samples=1)
    service.upsert_atom(1, "Альфа связана с Бета", ["Альфа", "Бета"])
    for mode in Retrieval:
        for _ in range(5):
            service.latency.record(mode, 3.0)  # everything slower than SLA
    res = service.search("Альфа и Бета", mode=SearchMode.AUTO)
    assert res.route is not None and res.route.sla_feasible is False
    assert service.latency.snapshot()["sla_fallbacks"] == 1

    before = service.latency.snapshot()["modes"]["graph"]["sample_count"]
    service.search("Альфа", mode=SearchMode.GRAPH)
    assert service.latency.snapshot()["modes"]["graph"]["sample_count"] == before + 1

"""Prometheus metrics, one registry per app instance (no global state between apps/tests)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "atomrag_requests_total",
            "HTTP requests by route and outcome",
            ["route", "outcome"],
            registry=self.registry,
        )
        self.search_phase = Histogram(
            "atomrag_search_phase_seconds",
            "Search latency by phase",
            ["phase"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.route_choice = Counter(
            "atomrag_route_total",
            "Router decisions by bundle",
            ["bundle", "sla_feasible"],
            registry=self.registry,
        )
        self.retrieval_p50 = Gauge(
            "atomrag_retrieval_p50_seconds",
            "Rolling p50 of in-service search latency",
            ["mode"],
            registry=self.registry,
        )
        self.retrieval_p95 = Gauge(
            "atomrag_retrieval_p95_seconds",
            "Rolling p95 used by the router once calibrated",
            ["mode"],
            registry=self.registry,
        )
        self.retrieval_samples = Gauge(
            "atomrag_retrieval_samples",
            "Samples in the rolling latency window",
            ["mode"],
            registry=self.registry,
        )
        self.sla_fallbacks = Gauge(
            "atomrag_router_sla_fallbacks", "Router decisions where no bundle fit the SLA", registry=self.registry
        )
        self.atoms = Gauge("atomrag_atoms", "Atoms in the in-memory index", registry=self.registry)
        self.entities = Gauge("atomrag_entities", "Entities in the in-memory graph", registry=self.registry)

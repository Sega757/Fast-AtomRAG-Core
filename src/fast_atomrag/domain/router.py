"""Cost-aware routing (CA-RAG style).

U_b = wQ*Q_b - wL*L_b/L_ref - wC*C_b/C_ref,  subject to L_b <= SLA, where
L_b = generation_s(b) + retrieval_s(mode(b)).

retrieval_s is supplied by the caller: the measured rolling p95 once enough samples exist,
a conservative default before that (ADR-009). generation_s and the quality priors Q_b are
ASSUMPTIONS: generation runs outside this service and is not observed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .errors import RoutingError


class Retrieval(StrEnum):
    NONE = "none"
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class Bundle:
    name: str
    retrieval: Retrieval
    top_k: int
    generation_s: float
    cost_usd: float


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    w_quality: float = 1.0
    w_latency: float = 0.6
    w_cost: float = 0.2
    sla_s: float = 2.0
    latency_ref_s: float = 22.3
    cost_ref_usd: float = 0.05


@dataclass(frozen=True, slots=True)
class RouteDecision:
    bundle: Bundle
    complexity: int
    utility: float
    sla_feasible: bool
    utilities: Mapping[str, float]
    latencies_s: Mapping[str, float]


# generation_s / cost: ASSUMPTION (design notes, end-to-end minus retrieval).
DEFAULT_BUNDLES: tuple[Bundle, ...] = (
    Bundle("b0_direct", Retrieval.NONE, 0, 0.45, 0.001),
    Bundle("b1_vector_k2", Retrieval.VECTOR, 2, 1.05, 0.004),
    Bundle("b2_vector_k5", Retrieval.VECTOR, 5, 1.75, 0.009),
    Bundle("b3_graph_k5", Retrieval.GRAPH, 5, 1.60, 0.010),
)

# Conservative cold-start retrieval latency per mode (seconds) until enough samples exist.
DEFAULT_RETRIEVAL_S: Mapping[Retrieval, float] = {
    Retrieval.NONE: 0.0,
    Retrieval.VECTOR: 0.05,
    Retrieval.GRAPH: 0.10,
}

# Quality priors per complexity level (0 simple, 1 single-hop, 2 multi-hop). ASSUMPTION.
DEFAULT_QUALITY: Mapping[int, Mapping[str, float]] = {
    0: {"b0_direct": 0.82, "b1_vector_k2": 0.85, "b2_vector_k5": 0.78, "b3_graph_k5": 0.74},
    1: {"b0_direct": 0.40, "b1_vector_k2": 0.80, "b2_vector_k5": 0.83, "b3_graph_k5": 0.81},
    2: {"b0_direct": 0.15, "b1_vector_k2": 0.42, "b2_vector_k5": 0.68, "b3_graph_k5": 0.88},
}


def complexity_level(sub_query_count: int, known_entity_count: int, token_count: int) -> int:
    """Structural heuristic (UNVERIFIED): multi-hop if several sub-queries or linked entities."""
    if sub_query_count >= 2 or known_entity_count >= 2:
        return 2
    if known_entity_count == 0 and token_count <= 5:
        return 0
    return 1


def select_bundle(
    bundles: Sequence[Bundle],
    quality: Mapping[str, float],
    weights: UtilityWeights,
    complexity: int,
    retrieval_s: Mapping[Retrieval, float],
) -> RouteDecision:
    if not bundles:
        raise RoutingError("bundle catalog is empty")
    if weights.latency_ref_s <= 0 or weights.cost_ref_usd <= 0:
        raise RoutingError("reference latency and cost must be > 0")

    utilities: dict[str, float] = {}
    latencies: dict[str, float] = {}
    feasible: list[tuple[float, Bundle]] = []
    for bundle in bundles:
        if bundle.name not in quality:
            raise RoutingError(f"no quality prior for bundle {bundle.name!r}")
        if bundle.retrieval not in retrieval_s:
            raise RoutingError(f"no retrieval latency for mode {bundle.retrieval.value!r}")
        latency = bundle.generation_s + retrieval_s[bundle.retrieval]
        u = (
            weights.w_quality * quality[bundle.name]
            - weights.w_latency * latency / weights.latency_ref_s
            - weights.w_cost * bundle.cost_usd / weights.cost_ref_usd
        )
        utilities[bundle.name] = round(u, 6)
        latencies[bundle.name] = round(latency, 6)
        if latency <= weights.sla_s:
            feasible.append((u, bundle))

    if feasible:
        utility, best = max(feasible, key=lambda item: (item[0], -latencies[item[1].name]))
        return RouteDecision(best, complexity, utility, True, utilities, latencies)
    # Graceful degradation: nothing fits the SLA -> fastest bundle, flagged.
    fastest = min(bundles, key=lambda b: latencies[b.name])
    return RouteDecision(fastest, complexity, utilities[fastest.name], False, utilities, latencies)

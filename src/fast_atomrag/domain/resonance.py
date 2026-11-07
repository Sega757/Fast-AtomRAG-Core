"""Resonance fusion of the semantic (cosine) and topological (PPR) signals.

LINEAR: S = alpha * cos + (1 - alpha) * ppr / max(ppr)      (formula from the design notes)
RRF:    S = alpha / (k0 + rank_cos) + (1 - alpha) / (k0 + rank_ppr)   (default, see ADR-004)

RRF is scale-free: a second-hop atom with small absolute PPR mass still competes with
lexically similar distractors, which LINEAR fails on (tests/test_service.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .errors import InvalidQueryError


class FusionMethod(StrEnum):
    LINEAR = "linear"
    RRF = "rrf"


@dataclass(frozen=True, slots=True)
class ScoredAtom:
    atom_id: int
    score: float
    vector_score: float
    graph_score: float


def _ranks(scores: Mapping[int, float]) -> dict[int, int]:
    ordered = sorted(scores, key=lambda a: (-scores[a], a))
    return {atom_id: rank for rank, atom_id in enumerate(ordered, start=1)}


def fuse(
    vector_scores: Mapping[int, float],
    graph_mass: Mapping[int, float],
    alpha: float,
    top_k: int,
    method: FusionMethod = FusionMethod.RRF,
    rrf_k: int = 60,
) -> list[ScoredAtom]:
    """Rank the union of candidates; ties break by atom_id for deterministic output.

    Reported vector_score is cosine clipped to [0, 1]; graph_score is max-normalized PPR mass.
    """
    if not 0.0 <= alpha <= 1.0:
        raise InvalidQueryError("alpha must be in [0, 1]")
    if top_k <= 0:
        raise InvalidQueryError("top_k must be > 0")
    if rrf_k < 1:
        raise InvalidQueryError("rrf_k must be >= 1")

    candidates = set(vector_scores) | set(graph_mass)
    if not candidates:
        return []
    peak = max((graph_mass.get(c, 0.0) for c in candidates), default=0.0)
    cos = {c: min(1.0, max(0.0, vector_scores.get(c, 0.0))) for c in candidates}
    graph = {c: (graph_mass.get(c, 0.0) / peak if peak > 0 else 0.0) for c in candidates}

    if method is FusionMethod.LINEAR:
        fused = {c: alpha * cos[c] + (1.0 - alpha) * graph[c] for c in candidates}
    else:
        v_rank = _ranks(cos)
        g_rank = _ranks({c: g for c, g in graph.items() if g > 0})
        fused = {
            c: alpha / (rrf_k + v_rank[c]) + ((1.0 - alpha) / (rrf_k + g_rank[c]) if c in g_rank else 0.0)
            for c in candidates
        }

    scored = [ScoredAtom(c, fused[c], cos[c], graph[c]) for c in candidates]
    scored.sort(key=lambda s: (-s.score, s.atom_id))
    return scored[:top_k]

"""Local Personalized PageRank via the forward-push algorithm (Andersen-Chung-Lang).

Work is bounded by the residual threshold, not by graph size: only the neighbourhood
of the seed nodes is touched. This replaces a global power iteration over the graph.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping

from .errors import InvalidQueryError


def personalized_pagerank(
    seeds: Mapping[Hashable, float],
    neighbors: Callable[[Hashable], list[Hashable]],
    degree: Callable[[Hashable], int],
    restart: float = 0.15,
    eps: float = 1e-4,
    max_pushes: int = 20_000,
) -> dict[Hashable, float]:
    """Approximate PPR vector p with p[u] >= true_ppr[u] - eps * deg(u) (when not truncated).

    `restart` is the teleport probability back to the seed distribution.
    `max_pushes` caps latency on pathological hubs; truncation keeps p a lower bound.
    """
    if not 0.0 < restart < 1.0:
        raise InvalidQueryError("restart must be in (0, 1)")
    if eps <= 0.0:
        raise InvalidQueryError("eps must be > 0")
    if max_pushes <= 0:
        raise InvalidQueryError("max_pushes must be > 0")

    total = sum(w for w in seeds.values() if w > 0)
    if total <= 0:
        return {}

    residual: dict[Hashable, float] = {n: w / total for n, w in seeds.items() if w > 0}
    estimate: dict[Hashable, float] = {}
    queue: deque[Hashable] = deque(residual)
    queued: set[Hashable] = set(residual)
    pushes = 0

    while queue and pushes < max_pushes:
        node = queue.popleft()
        queued.discard(node)
        mass = residual.get(node, 0.0)
        deg = degree(node)
        if deg == 0:
            # Dangling node: the walk can only restart from here; keep its mass as estimate.
            if mass > 0:
                estimate[node] = estimate.get(node, 0.0) + mass
                residual[node] = 0.0
            continue
        if mass < eps * deg:
            continue
        pushes += 1
        estimate[node] = estimate.get(node, 0.0) + restart * mass
        residual[node] = 0.0
        share = (1.0 - restart) * mass / deg
        for nxt in neighbors(node):
            new_mass = residual.get(nxt, 0.0) + share
            residual[nxt] = new_mass
            if nxt not in queued and new_mass >= eps * max(degree(nxt), 1):
                queue.append(nxt)
                queued.add(nxt)

    return estimate

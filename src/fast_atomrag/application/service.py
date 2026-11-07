"""Use cases: index atoms, delete atoms, resonance search with self-calibrating routing.

Write ordering (ADR-010): validate -> commit to the repository -> mutate memory, all under one
lock. A repository failure leaves memory untouched. A failure after commit invalidates the
service (not ready) so the orchestrator restarts it and memory is rebuilt from the repository.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ..domain.analysis import analyze_query
from ..domain.errors import InvalidQueryError
from ..domain.graph import ATOM, ENTITY, AtomEntityGraph
from ..domain.models import Atom
from ..domain.ppr import personalized_pagerank
from ..domain.resonance import FusionMethod, fuse
from ..domain.router import (
    DEFAULT_BUNDLES,
    DEFAULT_QUALITY,
    DEFAULT_RETRIEVAL_S,
    Bundle,
    Retrieval,
    RouteDecision,
    UtilityWeights,
    complexity_level,
    select_bundle,
)
from .errors import AtomNotFoundError, ServiceNotReadyError
from .latency import LatencyTracker
from .ports import AtomRepository, Embedder, VectorIndex

logger = logging.getLogger(__name__)

MAX_QUERY_LEN = 1000
MAX_TOP_K = 50


class SearchMode(StrEnum):
    AUTO = "auto"
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    alpha: float = 0.65
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    candidate_k: int = 50
    ppr_restart: float = 0.15
    ppr_eps: float = 1e-4
    ppr_max_pushes: int = 20_000
    bundles: Sequence[Bundle] = DEFAULT_BUNDLES
    quality: Mapping[int, Mapping[str, float]] = field(default_factory=lambda: DEFAULT_QUALITY)
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    retrieval_defaults_s: Mapping[Retrieval, float] = field(default_factory=lambda: DEFAULT_RETRIEVAL_S)
    latency_window: int = 256
    latency_min_samples: int = 10


@dataclass(frozen=True, slots=True)
class RetrievedAtom:
    atom_id: int
    text: str
    entities: tuple[str, ...]
    score: float
    vector_score: float
    graph_score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    mode: SearchMode
    retrieval: Retrieval
    route: RouteDecision | None
    sub_queries: tuple[str, ...]
    seed_entities: tuple[str, ...]
    seed_keys: tuple[str, ...]
    atoms: tuple[RetrievedAtom, ...]
    timings_ms: Mapping[str, float]


class AtomRAGService:
    """Owns the in-memory index/graph. The repository is the source of truth."""

    def __init__(
        self,
        repository: AtomRepository,
        embedder: Embedder,
        index: VectorIndex,
        graph: AtomEntityGraph,
        config: SearchConfig,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._index = index
        self._graph = graph
        self._config = config
        self._clock = clock
        self._lock = threading.RLock()
        self._atoms: dict[int, Atom] = {}
        self._ready = False
        self.latency = LatencyTracker(config.retrieval_defaults_s, config.latency_window, config.latency_min_samples)

    # ---- lifecycle ------------------------------------------------------
    def load(self) -> int:
        """Rebuild memory structures from the repository; re-embed rows from another embedder."""
        with self._lock:
            self._ready = False
            stale: list[Atom] = []
            for stored in self._repo.load_all():
                vec = stored.vector
                if stored.embedder_id != self._embedder.embedder_id or vec.shape != (self._embedder.dim,):
                    stale.append(stored.atom)
                else:
                    self._apply(stored.atom, vec)
            if stale:
                vectors = self._embedder.embed_many([a.text for a in stale])
                self._repo.upsert_many(list(zip(stale, vectors, strict=True)), self._embedder.embedder_id)
                for atom, vec in zip(stale, vectors, strict=True):
                    self._apply(atom, vec)
            self._ready = True
            return len(self._atoms)

    def close(self) -> None:
        with self._lock:
            self._ready = False
            self._repo.close()

    @property
    def ready(self) -> bool:
        return self._ready

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "atoms": len(self._atoms),
                "entities": self._graph.entity_count,
                "embedder": self._embedder.embedder_id,
            }

    # ---- commands -------------------------------------------------------
    def upsert_atom(self, atom_id: int, text: str, entities: Sequence[str]) -> bool:
        """Idempotent upsert. Returns True if created, False if replaced. Old edges are dropped."""
        atom = Atom.create(atom_id, text, entities)
        vector = self._checked_vector(self._embedder.embed(atom.text))
        with self._lock:
            self._require_ready()
            created = self._repo.upsert(atom, self._embedder.embedder_id, vector)  # may raise StorageError
            self._mutate_after_commit(lambda: self._apply(atom, vector))
            return created

    def delete_atom(self, atom_id: int) -> None:
        with self._lock:
            self._require_ready()
            if not self._repo.delete(atom_id):  # may raise StorageError
                raise AtomNotFoundError(atom_id)
            self._mutate_after_commit(lambda: self._drop(atom_id))

    def get_atom(self, atom_id: int) -> Atom:
        with self._lock:
            self._require_ready()
            atom = self._atoms.get(atom_id)
            if atom is None:
                raise AtomNotFoundError(atom_id)
            return atom

    # ---- query ----------------------------------------------------------
    def search(self, query: str, top_k: int = 5, mode: SearchMode = SearchMode.GRAPH) -> SearchResult:
        text = query.strip()
        if not text or len(text) > MAX_QUERY_LEN:
            raise InvalidQueryError(f"query length must be in [1, {MAX_QUERY_LEN}]")
        if not 1 <= top_k <= MAX_TOP_K:
            raise InvalidQueryError(f"top_k must be in [1, {MAX_TOP_K}]")
        cfg = self._config

        t0 = self._clock()
        analysis = analyze_query(text)
        t1 = self._clock()
        texts = [text, *analysis.sub_queries] if len(analysis.sub_queries) > 1 else [text]
        query_vectors = self._embedder.embed_many(texts)
        t2 = self._clock()

        with self._lock:
            self._require_ready()
            seeds = self._seed_entities(text, analysis.explicit_entities)
            t_seeds = self._clock()

            route: RouteDecision | None = None
            retrieval = Retrieval.GRAPH if mode is SearchMode.GRAPH else Retrieval.VECTOR
            k = top_k
            if mode is SearchMode.AUTO:
                level = complexity_level(len(analysis.sub_queries), len(seeds), analysis.token_count)
                route = select_bundle(cfg.bundles, cfg.quality[level], cfg.weights, level, self.latency.effective_s())
                if not route.sla_feasible:
                    self.latency.record_sla_fallback()
                retrieval = route.bundle.retrieval
                k = min(top_k, route.bundle.top_k)

            vector_scores: dict[int, float] = {}
            graph_mass: dict[int, float] = {}
            t3 = t4 = self._clock()
            if retrieval is not Retrieval.NONE and k > 0 and len(self._index) > 0:
                vector_scores = self._index.search(query_vectors, max(cfg.candidate_k, k))
                t3 = self._clock()
                if retrieval is Retrieval.GRAPH and seeds:
                    graph_mass = self._graph_candidates(seeds)
                    extra = [a for a in graph_mass if a not in vector_scores]
                    if extra:
                        vector_scores.update(self._index.score(query_vectors, extra))
                t4 = self._clock()
            alpha = cfg.alpha if retrieval is Retrieval.GRAPH else 1.0
            ranked = fuse(vector_scores, graph_mass, alpha, k, cfg.fusion, cfg.rrf_k) if k > 0 else []
            atoms = tuple(
                RetrievedAtom(
                    atom_id=s.atom_id,
                    text=self._atoms[s.atom_id].text,
                    entities=self._atoms[s.atom_id].entities,
                    score=round(s.score, 6),
                    vector_score=round(s.vector_score, 6),
                    graph_score=round(s.graph_score, 6),
                )
                for s in ranked
            )
            seed_display = tuple(self._graph.display(s) for s in seeds)
        t5 = self._clock()

        if retrieval is not Retrieval.NONE:
            self.latency.record(retrieval, t5 - t0)
        timings = {
            "analyze_ms": t1 - t0,
            "embed_ms": t2 - t1,
            "seeds_ms": t_seeds - t2,
            "vector_ms": t3 - t_seeds,
            "graph_ms": t4 - t3,
            "total_ms": t5 - t0,
        }
        return SearchResult(
            query=text,
            mode=mode,
            retrieval=retrieval,
            route=route,
            sub_queries=analysis.sub_queries,
            seed_entities=seed_display,
            seed_keys=tuple(seeds),
            atoms=atoms,
            timings_ms={name: round(v * 1000, 3) for name, v in timings.items()},
        )

    # ---- internals ------------------------------------------------------
    def _checked_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self._embedder.dim,) or not np.all(np.isfinite(vec)):
            raise ValueError("embedder returned an invalid vector")
        return vec

    def _mutate_after_commit(self, mutation: Callable[[], None]) -> None:
        try:
            mutation()
        except Exception:
            self._ready = False
            logger.exception("memory mutation failed after commit; service invalidated until reload")
            raise

    def _apply(self, atom: Atom, vector: np.ndarray) -> None:
        self._atoms[atom.atom_id] = atom
        self._index.upsert(atom.atom_id, vector)
        self._graph.upsert(atom.atom_id, atom.entities)

    def _drop(self, atom_id: int) -> None:
        self._atoms.pop(atom_id, None)
        self._index.remove(atom_id)
        self._graph.remove(atom_id)

    def _require_ready(self) -> None:
        if not self._ready:
            raise ServiceNotReadyError("index is not loaded")

    def _seed_entities(self, text: str, explicit: Sequence[str]) -> list[str]:
        seeds = self._graph.match_entities(text)
        seen = set(seeds)
        for name in explicit:
            key = self._graph.key(name)
            if key and key not in seen and self._graph.has_entity(key):
                seen.add(key)
                seeds.append(key)
        return seeds

    def _graph_candidates(self, seeds: Sequence[str]) -> dict[int, float]:
        cfg = self._config
        ppr = personalized_pagerank(
            seeds={(ENTITY, s): 1.0 for s in seeds},
            neighbors=self._graph.neighbors,
            degree=self._graph.degree,
            restart=cfg.ppr_restart,
            eps=cfg.ppr_eps,
            max_pushes=cfg.ppr_max_pushes,
        )
        atom_mass = ((node[1], mass) for node, mass in ppr.items() if node[0] == ATOM)
        return dict(heapq.nlargest(cfg.candidate_k, atom_mass, key=lambda item: item[1]))

"""FastAPI transport: parse -> call use case -> map errors. Run with `--factory`."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..application.errors import AtomNotFoundError, ServiceNotReadyError, StorageError
from ..application.service import AtomRAGService, SearchConfig
from ..domain.errors import InvalidAtomError, InvalidQueryError
from ..domain.graph import AtomEntityGraph
from ..domain.models import MAX_ATOM_ID
from ..domain.resonance import FusionMethod
from ..domain.router import UtilityWeights
from ..domain.text import plain_key, stemmed_key
from ..infrastructure.embedder import HashingEmbedder
from ..infrastructure.settings import Settings, load_settings
from ..infrastructure.sqlite_repository import SQLiteAtomRepository
from ..infrastructure.vector_index import ExactVectorIndex
from .metrics import Metrics
from .schemas import AtomIn, AtomOut, RetrievedAtomOut, RouteOut, SearchIn, SearchOut, StatsOut, UpsertOut

logger = logging.getLogger("fast_atomrag")

AtomId = Annotated[int, Path(ge=0, le=MAX_ATOM_ID)]


def get_service(request: Request) -> AtomRAGService:
    service = getattr(request.app.state, "service", None)
    if service is None or not service.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service not ready")
    return service


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected: str | None = request.app.state.settings.api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


Service = Annotated[AtomRAGService, Depends(get_service)]


def build_service(settings: Settings) -> AtomRAGService:
    embedder = HashingEmbedder(dim=settings.embed_dim)
    config = SearchConfig(
        alpha=settings.alpha,
        fusion=FusionMethod(settings.fusion),
        candidate_k=settings.candidate_k,
        ppr_restart=settings.ppr_restart,
        ppr_eps=settings.ppr_eps,
        ppr_max_pushes=settings.ppr_max_pushes,
        weights=UtilityWeights(sla_s=settings.sla_seconds),
    )
    return AtomRAGService(
        repository=SQLiteAtomRepository(settings.db_path),
        embedder=embedder,
        index=ExactVectorIndex(dim=embedder.dim),
        graph=AtomEntityGraph(
            key_fn=stemmed_key if settings.entity_stemming else plain_key,
            fuzzy_threshold=settings.fuzzy_threshold,
        ),
        config=config,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = build_service(settings)
        count = service.load()
        app.state.service = service
        stats = service.stats()
        metrics.atoms.set(count)
        metrics.entities.set(stats["entities"])
        logger.info("index loaded: %s atoms, embedder=%s", count, stats["embedder"])
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="Fast-AtomRAG Core", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    def refresh_gauges(service: AtomRAGService) -> None:
        stats = service.stats()
        metrics.atoms.set(stats["atoms"])
        metrics.entities.set(stats["entities"])

    def count(route: str, outcome: str) -> None:
        metrics.requests.labels(route=route, outcome=outcome).inc()

    @app.exception_handler(InvalidAtomError)
    @app.exception_handler(InvalidQueryError)
    async def invalid_input(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(AtomNotFoundError)
    async def not_found(_: Request, exc: AtomNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ServiceNotReadyError)
    async def not_ready(_: Request, __: ServiceNotReadyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "service not ready"})

    @app.exception_handler(StorageError)
    async def storage_unavailable(request: Request, exc: StorageError) -> JSONResponse:
        logger.error("storage error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=503, content={"detail": "storage unavailable"})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> JSONResponse:
        service = getattr(request.app.state, "service", None)
        if service is None or not service.ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(
            status_code=200,
            content={"status": "ready", **service.stats(), "router": service.latency.snapshot()},
        )

    @app.get("/metrics", tags=["telemetry"])
    def prometheus(request: Request) -> Response:
        service = getattr(request.app.state, "service", None)
        if service is not None:
            snap = service.latency.snapshot()
            for mode, values in snap["modes"].items():
                metrics.retrieval_p50.labels(mode=mode).set(values["p50_ms"] / 1000)
                metrics.retrieval_p95.labels(mode=mode).set(values["p95_ms"] / 1000)
                metrics.retrieval_samples.labels(mode=mode).set(values["sample_count"])
            metrics.sla_fallbacks.set(snap["sla_fallbacks"])
        return Response(content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/stats", response_model=StatsOut, tags=["atoms"])
    def stats(service: Service) -> dict[str, int | str]:
        return service.stats()

    @app.put(
        "/api/v1/atoms/{atom_id}",
        response_model=UpsertOut,
        tags=["atoms"],
        dependencies=[Depends(require_api_key)],
    )
    def upsert_atom(atom_id: AtomId, body: AtomIn, response: Response, service: Service) -> UpsertOut:
        created = service.upsert_atom(atom_id, body.text, body.entities)
        response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        refresh_gauges(service)
        count("upsert_atom", "created" if created else "updated")
        return UpsertOut(atom_id=atom_id, created=created)

    @app.get("/api/v1/atoms/{atom_id}", response_model=AtomOut, tags=["atoms"])
    def get_atom(atom_id: AtomId, service: Service) -> AtomOut:
        atom = service.get_atom(atom_id)
        return AtomOut(atom_id=atom.atom_id, text=atom.text, entities=list(atom.entities))

    @app.delete(
        "/api/v1/atoms/{atom_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["atoms"],
        dependencies=[Depends(require_api_key)],
    )
    def delete_atom(atom_id: AtomId, service: Service) -> Response:
        service.delete_atom(atom_id)
        refresh_gauges(service)
        count("delete_atom", "deleted")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/v1/search", response_model=SearchOut, tags=["retrieval"])
    def search(body: SearchIn, service: Service) -> SearchOut:
        result = service.search(body.query, top_k=body.top_k, mode=body.mode)
        for phase, ms in result.timings_ms.items():
            metrics.search_phase.labels(phase=phase.removesuffix("_ms")).observe(ms / 1000.0)
        route = None
        if result.route is not None:
            decision = result.route
            metrics.route_choice.labels(
                bundle=decision.bundle.name, sla_feasible=str(decision.sla_feasible).lower()
            ).inc()
            route = RouteOut(
                bundle=decision.bundle.name,
                retrieval=decision.bundle.retrieval.value,
                complexity=decision.complexity,
                utility=round(decision.utility, 6),
                sla_feasible=decision.sla_feasible,
                utilities=dict(decision.utilities),
                latencies_s=dict(decision.latencies_s),
            )
        count("search", "ok")
        return SearchOut(
            query=result.query,
            mode=result.mode,
            retrieval=result.retrieval.value,
            route=route,
            sub_queries=list(result.sub_queries),
            seed_entities=list(result.seed_entities),
            atoms=[
                RetrievedAtomOut(
                    atom_id=a.atom_id,
                    text=a.text,
                    entities=list(a.entities),
                    score=a.score,
                    vector_score=a.vector_score,
                    graph_score=a.graph_score,
                )
                for a in result.atoms
            ],
            timings_ms=dict(result.timings_ms),
        )

    return app

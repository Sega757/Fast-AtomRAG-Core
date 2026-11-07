"""HTTP DTOs. Transport validation only; domain invariants live in fast_atomrag.domain."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..application.service import SearchMode

EntityName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]


class AtomIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=3, max_length=4000, description="Self-contained fact")
    entities: list[EntityName] = Field(default_factory=list, max_length=32)


class AtomOut(BaseModel):
    atom_id: int
    text: str
    entities: list[str]


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchMode = SearchMode.GRAPH


class RetrievedAtomOut(BaseModel):
    atom_id: int
    text: str
    entities: list[str]
    score: float
    vector_score: float
    graph_score: float


class RouteOut(BaseModel):
    bundle: str
    retrieval: str
    complexity: int
    utility: float
    sla_feasible: bool
    utilities: dict[str, float]
    latencies_s: dict[str, float]


class SearchOut(BaseModel):
    query: str
    mode: SearchMode
    retrieval: str
    route: RouteOut | None
    sub_queries: list[str]
    seed_entities: list[str]
    atoms: list[RetrievedAtomOut]
    timings_ms: dict[str, float]


class UpsertOut(BaseModel):
    atom_id: int
    created: bool


class StatsOut(BaseModel):
    atoms: int
    entities: int
    embedder: str

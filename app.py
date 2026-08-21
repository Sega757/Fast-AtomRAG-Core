import os
import time
import re
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import networkx as nx
import usearch.index as ui
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# ==========================================
# 0. Prometheus Метрики
# ==========================================

REQ_COUNT = Counter(
    "rag_requests_total",
    "Total RAG search requests",
    ["status"]
)
LATENCY_HISTOGRAM = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG latency in seconds",
    ["phase"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
)
INDEX_SIZE_GAUGE = Gauge(
    "rag_indexed_atoms_count",
    "Total number of atomic units in memory graph/HNSW index"
)

# ==========================================
# 1. Схемы Pydantic (Request / Response)
# ==========================================

class AtomPayload(BaseModel):
    atom_id: int = Field(..., description="Уникальный идентификатор атома")
    text: str = Field(..., min_length=3, description="Текст фактологического атома")
    entities: List[str] = Field(default_factory=list, description="Список связанных сущностей")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Поисковый запрос")
    top_k: int = Field(default=3, ge=1, le=20, description="Количество извлекаемых атомов")

class RetrievedAtomDTO(BaseModel):
    atom_id: int
    text: str
    resonance_score: float
    linked_entities: List[str]

class SearchResponse(BaseModel):
    query: str
    retrieved_atoms: List[RetrievedAtomDTO]
    latency_breakdown_ms: Dict[str, float]
    total_latency_ms: float

# ==========================================
# 2. Изолированный движок Fast-AtomRAG
# ==========================================

class FastQueryDecomposer:
    def __init__(self, onnx_path: Optional[str] = None, dim: int = 384):
        self.dim = dim
        self.session = None
        if onnx_path and os.path.exists(onnx_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            
        self.split_pattern = re.compile(
            r'\b(?:и\s+как|а\s+также|но\s+при\s+этом|после\s+того\s+как|сравни\s+с|'
            r'and\s+how|as\s+well\s+as|whereas|compared\s+to)\b|[;?]',
            flags=re.IGNORECASE
        )
        self.quoted_pattern = re.compile(r'["«\']([^"»\']+)["»\']')
        self.capitalized_pattern = re.compile(r'\b[A-ZА-ЯЁ][a-zа-яё0-9_-]+(?:\s+[A-ZА-ЯЁ][a-zа-яё0-9_-]+)*\b')
        self.stopwords = {"что", "как", "почему", "зачем", "где", "what", "how", "why"}

    def extract_entities(self, text: str) -> List[str]:
        entities = set()
        for m in self.quoted_pattern.findall(text):
            if len(m.strip()) > 1:
                entities.add(m.strip())
        for m in self.capitalized_pattern.findall(text):
            cleaned = m.strip()
            if cleaned.lower() not in self.stopwords and len(cleaned) > 1:
                entities.add(cleaned)
        return list(entities)

    def decompose(self, text: str) -> List[str]:
        parts = self.split_pattern.split(text)
        sub = [p.strip() for p in parts if len(p.strip()) > 3]
        return sub if sub else [text.strip()]

    def encode(self, text: str) -> np.ndarray:
        if self.session is not None:
            max_len = 64
            input_ids = np.zeros((1, max_len), dtype=np.int64)
            attention_mask = np.zeros((1, max_len), dtype=np.int64)
            tokens = text.lower().split()[:max_len]
            for idx, token in enumerate(tokens):
                input_ids[0, idx] = hash(token) % 30522
                attention_mask[0, idx] = 1
            outputs = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            vec = outputs[0][0].mean(axis=0)
        else:
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(self.dim).astype(np.float32)
            
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

class FastAEGResonanceEngine:
    def __init__(self, dim: int = 384, alpha: float = 0.65):
        self.alpha = alpha
        self.atom_index = ui.Index(ndim=dim, metric="cos", dtype="f16")
        self.graph = nx.Graph()
        self.atom_metadata: Dict[int, Dict[str, Any]] = {}

    def index_atom(self, atom_id: int, text: str, vector: np.ndarray, entities: List[str]):
        self.atom_index.add(atom_id, vector)
        self.atom_metadata[atom_id] = {"text": text, "entities": entities}
        self.graph.add_node(atom_id, node_type="atom")
        for ent in entities:
            self.graph.add_node(ent, node_type="entity")
            self.graph.add_edge(atom_id, ent)
        INDEX_SIZE_GAUGE.set(len(self.atom_index))

    def search_resonance(self, query_vector: np.ndarray, seed_entities: List[str], top_k: int = 3) -> List[RetrievedAtomDTO]:
        if len(self.atom_index) == 0:
            return []

        search_k = min(50, len(self.atom_index))
        matches = self.atom_index.search(query_vector, search_k)
        candidate_ids = matches.keys.tolist()
        vector_scores = {cid: max(0.0, 1.0 - dist) for cid, dist in zip(candidate_ids, matches.distances)}

        ppr = {}
        valid_seeds = [e for e in seed_entities if self.graph.has_node(e)]
        if valid_seeds:
            personalization = {n: (1.0 / len(valid_seeds) if n in valid_seeds else 0.0) for n in self.graph.nodes}
            ppr = nx.pagerank(self.graph, alpha=0.85, personalization=personalization, max_iter=20, tol=1e-3)

        scored = []
        for aid in candidate_ids:
            v_score = vector_scores.get(aid, 0.0)
            neighbors = [n for n in self.graph.neighbors(aid) if self.graph.nodes[n].get("node_type") == "entity"]
            g_score = sum(ppr.get(n, 0.0) for n in neighbors) if neighbors else 0.0
            resonance = self.alpha * v_score + (1.0 - self.alpha) * g_score
            scored.append((aid, resonance))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for aid, score in scored[:top_k]:
            meta = self.atom_metadata[aid]
            results.append(RetrievedAtomDTO(
                atom_id=aid,
                text=meta["text"],
                resonance_score=round(float(score), 4),
                linked_entities=meta["entities"]
            ))
        return results

# ==========================================
# 3. Lifespan контекст приложения
# ==========================================

engine_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    onnx_path = os.getenv("ONNX_MODEL_PATH", "/app/models/model.onnx")
    decomposer = FastQueryDecomposer(onnx_path=onnx_path, dim=384)
    engine = FastAEGResonanceEngine(dim=384, alpha=0.65)
    
    # Pre-warm индекса (Seed Data)
    demo_atoms = [
        (1, "AtomRAG использует атомно-сущностный граф (AEG) для гранулярного хранения фактов.", ["AtomRAG", "AEG"]),
        (2, "Adaptive-RAG применяет классификатор сложности для снижения латентности.", ["Adaptive-RAG", "Маршрутизатор"]),
        (3, "Personalized PageRank обеспечивает топологический резонанс на графе.", ["PPR", "AEG"])
    ]
    for aid, txt, ents in demo_atoms:
        vec = decomposer.encode(txt)
        engine.index_atom(aid, txt, vec, ents)
        
    engine_state["decomposer"] = decomposer
    engine_state["engine"] = engine
    engine_state["ready"] = True
    yield
    engine_state.clear()

# ==========================================
# 4. Инициализация FastAPI и Роуты
# ==========================================

app = FastAPI(
    title="FastAtomRAG Inference Core",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health/liveness", tags=["Health"])
async def liveness():
    return {"status": "alive"}

@app.get("/health/readiness", tags=["Health"])
async def readiness():
    if not engine_state.get("ready", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not ready")
    return {"status": "ready", "indexed_atoms": len(engine_state["engine"].atom_index)}

@app.get("/metrics", tags=["Telemetry"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/atoms", tags=["Indexation"], status_code=status.HTTP_201_CREATED)
async def add_atom(payload: AtomPayload):
    decomposer: FastQueryDecomposer = engine_state["decomposer"]
    engine: FastAEGResonanceEngine = engine_state["engine"]
    
    vec = decomposer.encode(payload.text)
    engine.index_atom(payload.atom_id, payload.text, vec, payload.entities)
    return {"status": "indexed", "atom_id": payload.atom_id}

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_resonance(req: SearchRequest):
    t_start = time.perf_counter()
    breakdown = {}
    
    try:
        decomposer: FastQueryDecomposer = engine_state["decomposer"]
        engine: FastAEGResonanceEngine = engine_state["engine"]

        # Фаза 1: Декомпозиция и сущности
        t0 = time.perf_counter()
        entities = decomposer.extract_entities(req.query)
        vec = decomposer.encode(req.query)
        t_decomp = time.perf_counter() - t0
        breakdown["decomposer_ms"] = round(t_decomp * 1000, 2)
        LATENCY_HISTOGRAM.labels(phase="decomposer").observe(t_decomp)

        # Фаза 2: Графово-векторный резонансный поиск
        t1 = time.perf_counter()
        atoms = engine.search_resonance(query_vector=vec, seed_entities=entities, top_k=req.top_k)
        t_graph = time.perf_counter() - t1
        breakdown["resonance_search_ms"] = round(t_graph * 1000, 2)
        LATENCY_HISTOGRAM.labels(phase="graph_resonance").observe(t_graph)

        t_total = time.perf_counter() - t_start
        LATENCY_HISTOGRAM.labels(phase="total_retrieval").observe(t_total)
        REQ_COUNT.labels(status="success").inc()

        return SearchResponse(
            query=req.query,
            retrieved_atoms=atoms,
            latency_breakdown_ms=breakdown,
            total_latency_ms=round(t_total * 1000, 2)
        )
    except Exception as e:
        REQ_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
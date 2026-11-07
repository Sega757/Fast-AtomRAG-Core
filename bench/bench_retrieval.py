"""Retrieval latency benchmark on a synthetic Atom-Entity Graph (in-process, no HTTP).

Usage: python bench/bench_retrieval.py --atoms 10000 --queries 300
Measures what this repo controls (analysis + embedding + vector + PPR + fusion).
LLM generation is out of scope and not measured.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_atomrag.application.service import AtomRAGService, SearchConfig, SearchMode
from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.domain.models import Atom
from fast_atomrag.infrastructure.embedder import HashingEmbedder
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository
from fast_atomrag.infrastructure.vector_index import ExactVectorIndex

SYLLABLES = ["ра", "ко", "ми", "та", "ну", "ле", "зо", "фи", "до", "ве", "гра", "сто", "пли", "мер"]


def word(rng: random.Random) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    entities = [f"Ent{i}-{word(rng)}" for i in range(max(10, args.atoms // 10))]
    weights = [1.0 / (i + 1) for i in range(len(entities))]  # Zipf-like hubs

    embedder = HashingEmbedder(dim=args.dim)
    with tempfile.TemporaryDirectory() as tmp:
        repo = SQLiteAtomRepository(str(Path(tmp) / "bench.db"))
        atoms = []
        for atom_id in range(args.atoms):
            ents = list(dict.fromkeys(rng.choices(entities, weights=weights, k=rng.randint(2, 4))))
            text = " ".join(word(rng) for _ in range(rng.randint(8, 16))) + " " + " ".join(ents)
            atoms.append(Atom.create(atom_id, text, ents))

        t0 = time.perf_counter()
        vectors = embedder.embed_many([a.text for a in atoms])
        embed_s = time.perf_counter() - t0
        repo.upsert_many(list(zip(atoms, vectors, strict=True)), embedder.embedder_id)

        service = AtomRAGService(
            repo, embedder, ExactVectorIndex(args.dim), AtomEntityGraph(), SearchConfig(candidate_k=50)
        )
        t1 = time.perf_counter()
        service.load()
        load_s = time.perf_counter() - t1

        queries = []
        for _ in range(args.queries):
            ents = rng.sample(entities[: max(10, len(entities) // 5)], k=rng.randint(1, 2))
            queries.append(" ".join(word(rng) for _ in range(rng.randint(3, 6))) + " " + " и как ".join(ents))

        print(f"atoms={args.atoms} entities={len(entities)} dim={args.dim}")
        print(f"embed_throughput={args.atoms / embed_s:,.0f} atoms/s  load_from_sqlite={load_s:.2f}s")
        for mode in (SearchMode.VECTOR, SearchMode.GRAPH, SearchMode.AUTO):
            service.search(queries[0], mode=mode)  # warm-up
            totals, graph, seeds = [], [], []
            for q in queries:
                res = service.search(q, top_k=5, mode=mode)
                totals.append(res.timings_ms["total_ms"])
                graph.append(res.timings_ms["graph_ms"])
                seeds.append(res.timings_ms["seeds_ms"])
            print(
                f"mode={mode.value:6s} total_ms p50={statistics.median(totals):.2f} "
                f"p95={percentile(totals, 95):.2f} p99={percentile(totals, 99):.2f} max={max(totals):.2f} "
                f"| seeds_ms p95={percentile(seeds, 95):.3f} graph_ms p95={percentile(graph, 95):.2f}"
            )
        service.close()


if __name__ == "__main__":
    main()

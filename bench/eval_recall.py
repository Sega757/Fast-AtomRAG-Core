"""Multi-hop recall evaluation on bench/recall_corpus.py.

Usage: python bench/eval_recall.py [--repeat 20]
Prints Recall@3, Recall@5, MRR@10, chain-Recall@5, start-entity hit rate and p95 latency
for vector_only, linear_fusion, rrf_hybrid (no stemming) and rrf_hybrid (stemming + trigram).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.recall_corpus import CHAINS, chain_ids, corpus, target_id
from fast_atomrag.application.service import AtomRAGService, SearchConfig, SearchMode
from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.domain.resonance import FusionMethod
from fast_atomrag.domain.text import KeyFn, plain_key, stemmed_key
from fast_atomrag.infrastructure.embedder import HashingEmbedder
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository
from fast_atomrag.infrastructure.vector_index import ExactVectorIndex


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    mode: SearchMode
    fusion: FusionMethod
    key_fn: KeyFn
    fuzzy: float


VARIANTS = (
    Variant("vector_only", SearchMode.VECTOR, FusionMethod.RRF, stemmed_key, 0.75),
    Variant("linear_fusion", SearchMode.GRAPH, FusionMethod.LINEAR, stemmed_key, 0.75),
    Variant("rrf_hybrid_nostem", SearchMode.GRAPH, FusionMethod.RRF, plain_key, 0.0),
    Variant("rrf_hybrid_stem", SearchMode.GRAPH, FusionMethod.RRF, stemmed_key, 0.75),
)


@dataclass(frozen=True, slots=True)
class Scores:
    n_queries: int
    recall_at_3: float
    recall_at_5: float
    mrr_at_10: float
    chain_recall_at_5: float
    seed_hit_rate: float
    p95_ms: float
    recall_at_5_2hop: float
    recall_at_5_3hop: float


def _service(tmp: str, v: Variant) -> AtomRAGService:
    embedder = HashingEmbedder(dim=512)
    service = AtomRAGService(
        SQLiteAtomRepository(str(Path(tmp) / f"{v.name}.db")),
        embedder,
        ExactVectorIndex(512),
        AtomEntityGraph(key_fn=v.key_fn, fuzzy_threshold=v.fuzzy),
        SearchConfig(fusion=v.fusion),
    )
    service.load()
    for atom_id, text, entities in corpus():
        service.upsert_atom(atom_id, text, entities)
    return service


def evaluate(repeat: int = 20) -> dict[str, Scores]:
    results: dict[str, Scores] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for v in VARIANTS:
            service = _service(tmp, v)
            r3 = r5 = mrr = chain5 = seeds = 0.0
            hop_hits: dict[int, list[float]] = {2: [], 3: []}
            latencies: list[float] = []
            for c, chain in enumerate(CHAINS, start=1):
                res = service.search(chain.question, top_k=10, mode=v.mode)
                ranked = [a.atom_id for a in res.atoms]
                target = target_id(c, chain)
                rank = ranked.index(target) + 1 if target in ranked else None
                r3 += rank is not None and rank <= 3
                hit5 = rank is not None and rank <= 5
                r5 += hit5
                hop_hits[len(chain.hops)].append(float(hit5))
                mrr += 1.0 / rank if rank else 0.0
                chain5 += set(chain_ids(c, chain)) <= set(ranked[:5])
                seeds += v.key_fn(chain.start_entity) in res.seed_keys
                for _ in range(repeat):
                    latencies.append(service.search(chain.question, top_k=10, mode=v.mode).timings_ms["total_ms"])
            service.close()
            n = len(CHAINS)
            latencies.sort()
            results[v.name] = Scores(
                n_queries=n,
                recall_at_3=r3 / n,
                recall_at_5=r5 / n,
                mrr_at_10=mrr / n,
                chain_recall_at_5=chain5 / n,
                seed_hit_rate=seeds / n,
                p95_ms=latencies[round(0.95 * (len(latencies) - 1))],
                recall_at_5_2hop=sum(hop_hits[2]) / len(hop_hits[2]),
                recall_at_5_3hop=sum(hop_hits[3]) / len(hop_hits[3]),
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    results = evaluate(args.repeat)
    atoms = len(corpus())
    print(f"corpus: {atoms} atoms, {len(CHAINS)} multi-hop queries")
    print("| variant | Recall@3 | Recall@5 | R@5 2-hop | R@5 3-hop | MRR@10 | chain R@5 | start-entity hit | p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, s in results.items():
        seed_hit = "n/a" if name == "vector_only" else f"{s.seed_hit_rate:.3f}"  # graph channel unused
        print(
            f"| {name} | {s.recall_at_3:.3f} | {s.recall_at_5:.3f} | {s.recall_at_5_2hop:.3f} | "
            f"{s.recall_at_5_3hop:.3f} | {s.mrr_at_10:.3f} | {s.chain_recall_at_5:.3f} | "
            f"{seed_hit} | {s.p95_ms:.2f} |"
        )


if __name__ == "__main__":
    main()

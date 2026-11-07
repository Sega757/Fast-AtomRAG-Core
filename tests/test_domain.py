"""Domain invariants and algorithms: pure, no infrastructure."""

from __future__ import annotations

import pytest

from fast_atomrag.domain.analysis import analyze_query
from fast_atomrag.domain.errors import InvalidAtomError, InvalidQueryError, RoutingError
from fast_atomrag.domain.graph import ATOM, ENTITY, AtomEntityGraph
from fast_atomrag.domain.models import MAX_ATOM_ID, MAX_ENTITIES, Atom, normalize_entity
from fast_atomrag.domain.ppr import personalized_pagerank
from fast_atomrag.domain.resonance import FusionMethod, fuse
from fast_atomrag.domain.router import (
    DEFAULT_BUNDLES,
    DEFAULT_QUALITY,
    DEFAULT_RETRIEVAL_S,
    Bundle,
    Retrieval,
    UtilityWeights,
    complexity_level,
    select_bundle,
)


# ---- Atom invariants (risk: corrupt/unbounded input reaching the index) ----------------
@pytest.mark.parametrize("bad_id", [-1, MAX_ATOM_ID + 1, True, 1.5])
def test_atom_rejects_invalid_ids(bad_id):
    with pytest.raises(InvalidAtomError):
        Atom.create(bad_id, "valid text", [])


def test_atom_dedupes_entities_by_normalized_key():
    atom = Atom.create(1, "  fact text  ", ["AtomRAG", "atomrag", " Atom  RAG "])
    assert atom.text == "fact text"
    assert atom.entities == ("AtomRAG", "Atom RAG")
    assert atom.entity_keys == ("atomrag", "atom rag")


@pytest.mark.parametrize(
    ("text", "entities"),
    [("ab", []), ("ok text", ["   "]), ("ok text", [f"e{i}" for i in range(MAX_ENTITIES + 1)])],
)
def test_atom_rejects_invalid_payloads(text, entities):
    with pytest.raises(InvalidAtomError):
        Atom.create(1, text, entities)


def test_normalize_entity_is_unicode_and_case_insensitive():
    assert normalize_entity("ＡｔｏｍＲＡＧ") == "atomrag"
    assert normalize_entity("Ёлка   Маршрутизатора") == "елка маршрутизатора"


# ---- Graph consistency (risk: stale edges after re-upsert/delete) ----------------------
def test_graph_upsert_replaces_edges_and_remove_cleans_entities():
    g = AtomEntityGraph()
    g.upsert(1, ["A", "B"])
    g.upsert(2, ["B"])
    g.upsert(1, ["C"])
    assert g.has_entity("c") and not g.has_entity("a")
    assert g.degree((ENTITY, "b")) == 1
    assert g.remove(2) is True
    assert not g.has_entity("b")
    assert g.remove(2) is False
    assert g.atom_count == 1 and g.entity_count == 1


def test_graph_matcher_finds_multiword_and_punctuated_entities():
    g = AtomEntityGraph()
    g.upsert(1, ["Коллапс маршрутизатора", "L2A", "Llama-3.1"])
    found = g.match_entities("Как избежать: коллапса маршрутизаторов в L2A? И Llama 3.1.")
    assert found == [g.key("Коллапс маршрутизатора"), "l2a", "llama 3.1"]
    g.remove(1)
    assert g.match_entities("L2A") == []


# ---- PPR correctness (risk: wrong graph scores silently degrade ranking) --------------
def _power_iteration(adj: dict, seeds: dict, restart: float, iters: int = 500) -> dict:
    nodes = list(adj)
    p = {n: seeds.get(n, 0.0) for n in nodes}
    for _ in range(iters):
        nxt = {n: restart * seeds.get(n, 0.0) for n in nodes}
        for u in nodes:
            for v in adj[u]:
                nxt[v] += (1 - restart) * p[u] / len(adj[u])
        p = nxt
    return p


def test_push_ppr_matches_power_iteration():
    g = AtomEntityGraph()
    g.upsert(1, ["x", "y"])
    g.upsert(2, ["y", "z"])
    g.upsert(3, ["z", "x"])
    g.upsert(4, ["z"])
    adj = {}
    for node in [(ATOM, i) for i in range(1, 5)] + [(ENTITY, e) for e in "xyz"]:
        adj[node] = g.neighbors(node)
    seeds = {(ENTITY, "x"): 1.0}
    exact = _power_iteration(adj, seeds, 0.15)
    approx = personalized_pagerank(seeds, g.neighbors, g.degree, restart=0.15, eps=1e-9, max_pushes=10**6)
    for node, value in exact.items():
        assert approx.get(node, 0.0) == pytest.approx(value, abs=1e-6)


def test_push_ppr_is_local_to_seed_component():
    g = AtomEntityGraph()
    g.upsert(1, ["x"])
    g.upsert(2, ["far"])
    ppr = personalized_pagerank({(ENTITY, "x"): 1.0}, g.neighbors, g.degree)
    assert (ATOM, 2) not in ppr and (ATOM, 1) in ppr


def test_push_ppr_truncation_keeps_lower_bound_and_validates_params():
    g = AtomEntityGraph()
    for i in range(50):
        g.upsert(i, ["hub", f"e{i}"])
    full = personalized_pagerank({(ENTITY, "hub"): 1.0}, g.neighbors, g.degree, eps=1e-9, max_pushes=10**6)
    cut = personalized_pagerank({(ENTITY, "hub"): 1.0}, g.neighbors, g.degree, eps=1e-9, max_pushes=5)
    assert all(cut[n] <= full[n] + 1e-12 for n in cut)
    assert sum(cut.values()) < sum(full.values())
    with pytest.raises(InvalidQueryError):
        personalized_pagerank({}, g.neighbors, g.degree, restart=1.0)
    assert personalized_pagerank({}, g.neighbors, g.degree) == {}


# ---- Fusion (risk: one term dominating due to scale mismatch) -------------------------
def test_linear_fuse_normalizes_graph_mass_and_unions_candidates():
    ranked = fuse({1: 0.9, 2: 0.1}, {2: 0.002, 3: 0.001}, alpha=0.5, top_k=3, method=FusionMethod.LINEAR)
    by_id = {s.atom_id: s for s in ranked}
    assert by_id[2].graph_score == pytest.approx(1.0)
    assert by_id[3].graph_score == pytest.approx(0.5) and by_id[3].vector_score == 0.0
    assert [s.atom_id for s in ranked] == [2, 1, 3]


@pytest.mark.parametrize("method", list(FusionMethod))
def test_fuse_alpha_one_is_pure_vector_and_ties_are_deterministic(method):
    ranked = fuse({5: 0.5, 3: 0.5, 9: 0.7}, {3: 1.0}, alpha=1.0, top_k=3, method=method)
    assert [s.atom_id for s in ranked] == [9, 3, 5]
    with pytest.raises(InvalidQueryError):
        fuse({}, {}, alpha=1.5, top_k=1, method=method)


def test_rrf_lifts_graph_reachable_atom_with_small_mass_over_lexical_distractor():
    vector = {10: 0.70, 11: 0.69, 1: 0.60, 2: 0.30}
    mass = {1: 0.33, 2: 0.13}
    linear = fuse(vector, mass, alpha=0.65, top_k=2, method=FusionMethod.LINEAR)
    rrf = fuse(vector, mass, alpha=0.65, top_k=2, method=FusionMethod.RRF)
    assert [s.atom_id for s in linear] == [1, 10]
    assert [s.atom_id for s in rrf] == [1, 2]


# ---- Query analysis (risk: multi-hop queries not decomposed / entities missed) --------
def test_analyze_query_splits_and_extracts_entities():
    a = analyze_query("Сравни «AtomRAG» с Adaptive-RAG, и как это влияет на Llama-3.1?")
    assert len(a.sub_queries) == 2
    assert {"AtomRAG", "Adaptive-RAG", "Llama-3.1"} <= set(a.explicit_entities)


def test_analyze_query_keeps_simple_query_whole():
    a = analyze_query("столица Франции")
    assert a.sub_queries == ("столица Франции",) and a.explicit_entities == ()


# ---- Router (risk: SLA violation or nonsense route) ----------------------------------
def test_router_respects_sla_and_picks_max_utility():
    w = UtilityWeights(sla_s=2.0)
    simple = select_bundle(DEFAULT_BUNDLES, DEFAULT_QUALITY[0], w, 0, DEFAULT_RETRIEVAL_S)
    multi = select_bundle(DEFAULT_BUNDLES, DEFAULT_QUALITY[2], w, 2, DEFAULT_RETRIEVAL_S)
    assert simple.bundle.name == "b1_vector_k2"
    assert multi.bundle.retrieval is Retrieval.GRAPH and multi.sla_feasible


def test_router_degrades_gracefully_when_nothing_fits_sla():
    slow = (Bundle("slow", Retrieval.GRAPH, 5, 22.3, 0.05), Bundle("less_slow", Retrieval.VECTOR, 5, 5.0, 0.01))
    d = select_bundle(slow, {"slow": 0.9, "less_slow": 0.5}, UtilityWeights(sla_s=2.0), 2, DEFAULT_RETRIEVAL_S)
    assert d.bundle.name == "less_slow" and d.sla_feasible is False


def test_router_rejects_inconsistent_catalog():
    with pytest.raises(RoutingError):
        select_bundle(DEFAULT_BUNDLES, {"b0_direct": 1.0}, UtilityWeights(), 0, DEFAULT_RETRIEVAL_S)
    with pytest.raises(RoutingError):
        select_bundle((), {}, UtilityWeights(), 0, DEFAULT_RETRIEVAL_S)


def test_complexity_heuristic():
    assert complexity_level(2, 0, 10) == 2
    assert complexity_level(1, 2, 10) == 2
    assert complexity_level(1, 0, 3) == 0
    assert complexity_level(1, 1, 3) == 1

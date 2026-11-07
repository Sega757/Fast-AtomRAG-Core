"""L4 multi-hop recall on the stress corpus (bench/recall_corpus.py, bench/eval_recall.py)."""

from __future__ import annotations

import pytest

from bench.eval_recall import evaluate
from bench.recall_corpus import CHAINS, STOPWORDS, corpus
from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.domain.text import key_tokens, plain_key, stemmed_key


def _content_stems(text: str) -> set[str]:
    return {t for t in stemmed_key(text).split() if len(t) >= 3 and t not in STOPWORDS}


def test_corpus_shape():
    rows = corpus()
    assert 60 <= len(rows) <= 100
    assert len(CHAINS) == 15
    assert sum(len(c.hops) == 2 for c in CHAINS) >= 1 and sum(len(c.hops) == 3 for c in CHAINS) >= 1
    assert sum(len(c.distractors) for c in CHAINS) == 30
    assert len({r[0] for r in rows}) == len(rows)


@pytest.mark.parametrize("chain", CHAINS, ids=[c.name for c in CHAINS])
def test_target_is_lexically_disjoint_from_question(chain):
    target_text = chain.hops[-1][0]
    assert _content_stems(chain.question) & _content_stems(target_text) == set()


@pytest.mark.parametrize("chain", CHAINS, ids=[c.name for c in CHAINS])
def test_inflected_start_entity_resolves_to_its_graph_node(chain):
    g = AtomEntityGraph()
    for atom_id, (_, entities) in enumerate(chain.hops):
        g.upsert(atom_id, entities)
    assert stemmed_key(chain.start_entity) in g.match_entities(chain.question)


def test_chains_use_different_surface_forms_between_hops():
    """The corpus must actually exercise inflection: adjacent hops never share a plain key."""
    for chain in CHAINS:
        for (_, left), (_, right) in zip(chain.hops, chain.hops[1:], strict=False):
            assert {plain_key(e) for e in left} & {plain_key(e) for e in right} == set(), chain.name
            assert {stemmed_key(e) for e in left} & {stemmed_key(e) for e in right}, chain.name
        assert key_tokens(chain.start_entity)


@pytest.fixture(scope="module")
def scores():
    return evaluate(repeat=3)


def test_rrf_hybrid_beats_vector_and_linear_on_multi_hop(scores):
    rrf, vec, lin = scores["rrf_hybrid_stem"], scores["vector_only"], scores["linear_fusion"]
    assert rrf.recall_at_5 > vec.recall_at_5
    assert rrf.recall_at_5 > lin.recall_at_5
    assert rrf.recall_at_5_2hop > lin.recall_at_5_2hop and rrf.recall_at_5_3hop > lin.recall_at_5_3hop
    assert rrf.mrr_at_10 > vec.mrr_at_10 and rrf.mrr_at_10 > lin.mrr_at_10
    assert rrf.recall_at_5 >= 0.9


def test_stemming_gives_full_start_node_coverage_and_drives_recall(scores):
    assert scores["rrf_hybrid_stem"].seed_hit_rate == 1.0
    assert scores["rrf_hybrid_nostem"].seed_hit_rate < 1.0
    assert scores["rrf_hybrid_stem"].recall_at_5 > scores["rrf_hybrid_nostem"].recall_at_5

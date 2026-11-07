"""ADR-008: stemming, entity-key normalization, fuzzy fallback, overhead budget."""

from __future__ import annotations

import time

import pytest

from fast_atomrag.domain.graph import AtomEntityGraph
from fast_atomrag.domain.stemming import stem_en, stem_ru, stem_token
from fast_atomrag.domain.text import char_trigrams, plain_key, stemmed_key


# Reference outputs of the published Snowball-Russian and Porter algorithms.
@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("коллапса", "коллапс"),
        ("коллапсом", "коллапс"),
        ("маршрутизаторов", "маршрутизатор"),
        ("балансировщика", "балансировщик"),
        ("утечку", "утечк"),
        ("утечки", "утечк"),
        ("дескрипторов", "дескриптор"),
        ("красивейший", "красив"),
        ("бегавшись", "бега"),
        ("соединений", "соединен"),
        ("нормализации", "нормализац"),
    ],
)
def test_snowball_russian(word, stem):
    assert stem_ru(word) == stem


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("generalization", "gener"),
        ("hopping", "hop"),
        ("agreed", "agre"),
        ("feed", "feed"),
        ("controll", "control"),
        ("roll", "roll"),
        ("connections", "connect"),
        ("stampedes", "stamped"),
    ],
)
def test_porter_english(word, stem):
    assert stem_en(word) == stem


def test_stem_token_leaves_codes_and_short_tokens_alone():
    assert stem_token("3.1") == "3.1"
    assert stem_token("l2a") == "l2a"
    assert stem_token("ок") == "ок"


def test_keys_collapse_case_yo_hyphens_and_inflection():
    assert plain_key("  Ёмкость--Пула  ") == "емкость пула"
    assert plain_key("Llama-3.1") == "llama 3.1"
    assert stemmed_key("Коллапса Маршрутизатора") == stemmed_key("коллапс маршрутизатора")
    assert stemmed_key("Cache Stampedes") == stemmed_key("cache stampede")
    assert stemmed_key("утечку дескрипторов") == stemmed_key("Утечка Дескрипторов")


def test_inflected_entities_share_one_graph_node():
    g = AtomEntityGraph()
    g.upsert(1, ["утечки дескрипторов"])
    g.upsert(2, ["Утечка дескрипторов"])
    assert g.entity_count == 1
    assert g.match_entities("Как устранили утечку дескрипторов?") == [g.key("утечка дескрипторов")]


@pytest.mark.parametrize(
    ("entity", "query"),
    [
        ("отказоустойчивость кластера", "что с отказоустойчивасть кластера"),
        ("connection pool exhaustion", "why conection pool exhaustion happens"),
        ("утечка файловых дескрипторов", "утечка файловых дискрипторов в модуле"),
    ],
)
def test_trigram_fallback_catches_typos_above_threshold(entity, query):
    g = AtomEntityGraph(fuzzy_threshold=0.75)
    g.upsert(1, [entity, "unrelated thing"])
    assert g.match_entities(query) == [g.key(entity)]
    assert AtomEntityGraph(fuzzy_threshold=0.0).match_entities(query) == []


def test_trigram_fallback_rejects_below_threshold_and_is_skipped_on_exact_hit():
    g = AtomEntityGraph()
    g.upsert(1, ["kubernetes", "postgresql replication"])
    jac = len(char_trigrams("kubernet") & char_trigrams("kuberneti")) / len(
        char_trigrams("kubernet") | char_trigrams("kuberneti")
    )
    assert jac < 0.75
    assert g.match_entities("kubernetis outage") == []
    assert g.match_entities("kubernetes and postgresq replication") == [g.key("kubernetes")]


def test_fuzzy_index_is_cleaned_on_remove():
    g = AtomEntityGraph()
    g.upsert(1, ["circuit breaker configuration"])
    g.remove(1)
    assert g.match_entities("circuit braker configuration") == []


def test_entity_matching_overhead_under_budget():
    """ADR-008 budget: <= 0.2 ms per query for key normalization + exact matching (warm cache)."""
    g = AtomEntityGraph()
    for i in range(5000):
        g.upsert(i, [f"сущность{i} модуль", "Kubernetes", "коллапс маршрутизатора"])
    query = "Почему произошёл коллапс маршрутизатора при нагрузке на Kubernetes кластер в пятницу?"
    for _ in range(50):
        g.match_entities(query)
    samples = []
    for _ in range(300):
        t0 = time.perf_counter()
        g.match_entities(query)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    assert samples[len(samples) // 2] < 0.0002

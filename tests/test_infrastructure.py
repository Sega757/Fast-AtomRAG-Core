"""Adapters against real dependencies (numpy, SQLite file, subprocesses)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from fast_atomrag.domain.models import Atom
from fast_atomrag.infrastructure.embedder import HashingEmbedder
from fast_atomrag.infrastructure.settings import ConfigError, load_settings
from fast_atomrag.infrastructure.sqlite_repository import SQLiteAtomRepository
from fast_atomrag.infrastructure.vector_index import ExactVectorIndex


# ---- Embedder (risk: vectors differ across workers/restarts -> index silently wrong) -
def test_embedder_is_stable_across_processes():
    code = (
        "from fast_atomrag.infrastructure.embedder import HashingEmbedder;"
        "v=HashingEmbedder(dim=128).embed('AtomRAG резонанс');"
        "print(','.join(f'{x:.6f}' for x in v[:16]))"
    )
    outs = set()
    for seed in ("1", "2"):
        src = os.path.join(os.path.dirname(__file__), "..", "src")
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": os.path.abspath(src)}
        outs.add(
            subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        )
    assert len(outs) == 1


def test_embedder_similarity_tracks_lexical_overlap_including_inflection():
    e = HashingEmbedder(dim=512)
    q = e.embed("коллапс маршрутизатора")
    related = e.embed("Регуляризатор предотвращает коллапс маршрутизатора")
    inflected = e.embed("коллапса маршрутизаторов")
    unrelated = e.embed("Рецепт борща со свёклой")
    assert float(q @ related) > float(q @ unrelated)
    assert float(q @ inflected) > float(q @ unrelated)
    assert np.linalg.norm(e.embed("")) == 0.0
    with pytest.raises(ValueError):
        HashingEmbedder(dim=8)


# ---- Vector index (risk: swap-remove/growth corrupts id<->row mapping) ----------------
def test_index_matches_brute_force_after_upserts_replacements_and_removals():
    rng = np.random.default_rng(7)
    dim = 32
    idx = ExactVectorIndex(dim, initial_capacity=2)
    ref: dict[int, np.ndarray] = {}
    for step in range(300):
        atom_id = int(rng.integers(0, 60))
        if step % 4 == 3:
            assert idx.remove(atom_id) == (atom_id in ref)
            ref.pop(atom_id, None)
        else:
            v = rng.normal(size=dim).astype(np.float32)
            idx.upsert(atom_id, v)
            ref[atom_id] = v / np.linalg.norm(v)
    assert len(idx) == len(ref)
    q = rng.normal(size=dim).astype(np.float32)
    q /= np.linalg.norm(q)
    expected = sorted(ref, key=lambda a: -float(ref[a] @ q))[:5]
    got = idx.search(q, 5)
    assert set(got) == set(expected)
    for atom_id, score in idx.score(q, [*list(ref)[:10], 999]).items():
        assert score == pytest.approx(float(ref[atom_id] @ q), abs=1e-5)


def test_index_validates_vectors_and_multi_query_takes_max():
    idx = ExactVectorIndex(4)
    with pytest.raises(ValueError):
        idx.upsert(1, np.zeros(3))
    with pytest.raises(ValueError):
        idx.upsert(1, np.array([np.nan, 0, 0, 0]))
    idx.upsert(1, np.array([1, 0, 0, 0]))
    idx.upsert(2, np.array([0, 1, 0, 0]))
    res = idx.search(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32), 2)
    assert res == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}
    assert idx.search(np.array([1, 0, 0, 0]), 0) == {}


# ---- SQLite repository (risk: data loss / wrong created flag / schema drift) ----------
def test_repository_roundtrip_persists_across_reopen(db_path):
    repo = SQLiteAtomRepository(db_path)
    atom = Atom.create(42, "Факт про AEG", ["AEG", "АтомРАГ"])
    vec = np.arange(4, dtype=np.float32)
    assert repo.upsert(atom, "emb-1", vec) is True
    assert repo.upsert(atom, "emb-1", vec) is False
    repo.close()

    repo = SQLiteAtomRepository(db_path)
    stored = list(repo.load_all())
    assert len(stored) == 1 and stored[0].atom == atom and stored[0].embedder_id == "emb-1"
    np.testing.assert_array_equal(stored[0].vector, vec)
    assert repo.delete(42) is True and repo.delete(42) is False
    assert repo.count() == 0
    repo.close()


def test_repository_refuses_newer_schema(db_path):
    repo = SQLiteAtomRepository(db_path)
    repo._conn.execute("PRAGMA user_version=99")
    repo.close()
    with pytest.raises(RuntimeError):
        SQLiteAtomRepository(db_path)


# ---- Settings (risk: bad config discovered at request time instead of startup) --------
def test_settings_defaults_and_validation():
    s = load_settings({})
    assert s.alpha == 0.65 and s.api_key is None
    assert load_settings({"ATOMRAG_ALPHA": "0.3"}).alpha == 0.3
    for bad in (
        {"ATOMRAG_ALPHA": "2"},
        {"ATOMRAG_CANDIDATE_K": "x"},
        {"ATOMRAG_API_KEY": "short"},
        {"ATOMRAG_LOG_LEVEL": "LOUD"},
        {"ATOMRAG_FUSION": "magic"},
    ):
        with pytest.raises(ConfigError):
            load_settings(bad)

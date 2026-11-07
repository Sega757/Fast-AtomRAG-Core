"""In-memory bipartite Atom-Entity Graph (AEG) with exact and fuzzy entity matching.

Entity keys come from a pluggable key function (default: stemmed_key, ADR-008), so
"коллапса маршрутизатора" and "коллапс маршрутизатора" are ONE node. If no entity phrase
matches a query exactly, a char-trigram Jaccard fallback (threshold, default 0.75) is used.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .text import KeyFn, char_trigrams, stemmed_key

ATOM = "a"
ENTITY = "e"
Node = tuple[str, int | str]


class AtomEntityGraph:
    """Bipartite graph: atom nodes <-> entity nodes. Not thread-safe; callers serialize access."""

    MAX_MATCH_WORDS = 6

    def __init__(self, key_fn: KeyFn = stemmed_key, fuzzy_threshold: float = 0.75) -> None:
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be in [0, 1] (0 disables the fallback)")
        self._key_fn = key_fn
        self._fuzzy_threshold = fuzzy_threshold
        self._atom_entities: dict[int, frozenset[str]] = {}
        self._entity_atoms: dict[str, set[int]] = {}
        self._display: dict[str, str] = {}
        self._lengths: Counter[int] = Counter()
        self._grams: dict[str, frozenset[str]] = {}
        self._gram_index: dict[str, set[str]] = {}

    def key(self, name: str) -> str:
        return self._key_fn(name)

    # ---- mutation -------------------------------------------------------
    def upsert(self, atom_id: int, entity_displays: Sequence[str]) -> None:
        """Replace all edges of `atom_id` (old edges are removed first: no phantom links)."""
        self.remove(atom_id)
        keys: set[str] = set()
        for display in entity_displays:
            key = self._key_fn(display)
            if not key or key in keys:
                continue
            keys.add(key)
            atoms = self._entity_atoms.get(key)
            if atoms is None:
                self._entity_atoms[key] = {atom_id}
                self._display[key] = display
                self._index_entity(key)
            else:
                atoms.add(atom_id)
        self._atom_entities[atom_id] = frozenset(keys)

    def remove(self, atom_id: int) -> bool:
        keys = self._atom_entities.pop(atom_id, None)
        if keys is None:
            return False
        for key in keys:
            atoms = self._entity_atoms[key]
            atoms.discard(atom_id)
            if not atoms:
                del self._entity_atoms[key]
                del self._display[key]
                self._unindex_entity(key)
        return True

    def _index_entity(self, key: str) -> None:
        self._lengths[len(key.split())] += 1
        if self._fuzzy_threshold > 0:
            grams = char_trigrams(key)
            self._grams[key] = grams
            for g in grams:
                self._gram_index.setdefault(g, set()).add(key)

    def _unindex_entity(self, key: str) -> None:
        n = len(key.split())
        self._lengths[n] -= 1
        if self._lengths[n] <= 0:
            del self._lengths[n]
        grams = self._grams.pop(key, frozenset())
        for g in grams:
            bucket = self._gram_index[g]
            bucket.discard(key)
            if not bucket:
                del self._gram_index[g]

    # ---- queries --------------------------------------------------------
    def neighbors(self, node: Node) -> list[Node]:
        kind, ident = node
        if kind == ATOM:
            return [(ENTITY, k) for k in self._atom_entities.get(ident, ())]  # type: ignore[arg-type]
        return [(ATOM, a) for a in self._entity_atoms.get(ident, ())]  # type: ignore[arg-type]

    def degree(self, node: Node) -> int:
        kind, ident = node
        if kind == ATOM:
            return len(self._atom_entities.get(ident, ()))  # type: ignore[arg-type]
        return len(self._entity_atoms.get(ident, ()))  # type: ignore[arg-type]

    def has_entity(self, key: str) -> bool:
        return key in self._entity_atoms

    def display(self, key: str) -> str:
        return self._display.get(key, key)

    def entities_of(self, atom_id: int) -> frozenset[str]:
        return self._atom_entities.get(atom_id, frozenset())

    def match_entities(self, text: str) -> list[str]:
        """Keys of known entities found in `text`: exact key-phrase match, else fuzzy fallback."""
        if not self._lengths:
            return []
        tokens = self._key_fn(text).split()
        lengths = sorted((n for n in self._lengths if n <= self.MAX_MATCH_WORDS), reverse=True)
        found: list[str] = []
        seen: set[str] = set()
        for start in range(len(tokens)):
            for n in lengths:
                if start + n > len(tokens):
                    continue
                phrase = " ".join(tokens[start : start + n])
                if phrase in self._entity_atoms and phrase not in seen:
                    seen.add(phrase)
                    found.append(phrase)
        if found or self._fuzzy_threshold <= 0:
            return found
        return self._fuzzy_match(tokens, lengths)

    def _fuzzy_match(self, tokens: list[str], lengths: list[int]) -> list[str]:
        """Trigram Jaccard >= threshold with prefix filtering: probe only the rarest
        |A| - ceil(t*|A|) + 1 grams of each window; any qualifying entity must share one."""
        t = self._fuzzy_threshold
        best: dict[str, float] = {}
        for n in lengths:
            for start in range(len(tokens) - n + 1):
                window = char_trigrams(" ".join(tokens[start : start + n]))
                size = len(window)
                need = math.ceil(t * size)
                probe = sorted(window, key=lambda g: len(self._gram_index.get(g, ())))[: size - need + 1]
                candidates: set[str] = set()
                for g in probe:
                    candidates |= self._gram_index.get(g, set())
                for key in candidates:
                    grams = self._grams[key]
                    if not t * size <= len(grams) <= size / t:
                        continue
                    inter = len(window & grams)
                    score = inter / (size + len(grams) - inter)
                    if score >= t and score > best.get(key, 0.0):
                        best[key] = score
        return sorted(best, key=lambda k: (-best[k], k))

    @property
    def atom_count(self) -> int:
        return len(self._atom_entities)

    @property
    def entity_count(self) -> int:
        return len(self._entity_atoms)

"""Core value objects: atoms of knowledge and entity normalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .errors import InvalidAtomError
from .text import plain_key

MAX_ATOM_ID = 2**63 - 1
MIN_TEXT_LEN = 3
MAX_TEXT_LEN = 4000
MAX_ENTITIES = 32
MAX_ENTITY_LEN = 128


def normalize_entity(name: str) -> str:
    """Surface-level dedupe key (no stemming): NFKC, casefold, ё->е, spaces/hyphens collapsed."""
    return plain_key(name)


@dataclass(frozen=True, slots=True)
class Atom:
    """A self-contained fact linked to named entities (a node of the Atom-Entity Graph)."""

    atom_id: int
    text: str
    entities: tuple[str, ...]

    @classmethod
    def create(cls, atom_id: int, text: str, entities: Iterable[str]) -> Atom:
        if isinstance(atom_id, bool) or not isinstance(atom_id, int):
            raise InvalidAtomError("atom_id must be an integer")
        if not 0 <= atom_id <= MAX_ATOM_ID:
            raise InvalidAtomError(f"atom_id must be in [0, {MAX_ATOM_ID}]")
        clean_text = text.strip()
        if not MIN_TEXT_LEN <= len(clean_text) <= MAX_TEXT_LEN:
            raise InvalidAtomError(f"text length must be in [{MIN_TEXT_LEN}, {MAX_TEXT_LEN}]")

        seen: set[str] = set()
        displays: list[str] = []
        for raw in entities:
            display = " ".join(raw.split())
            key = normalize_entity(raw)
            if not key:
                raise InvalidAtomError("entity names must not be blank")
            if len(display) > MAX_ENTITY_LEN:
                raise InvalidAtomError(f"entity names must be at most {MAX_ENTITY_LEN} chars")
            if key in seen:
                continue
            seen.add(key)
            displays.append(display)
        if len(displays) > MAX_ENTITIES:
            raise InvalidAtomError(f"an atom may link at most {MAX_ENTITIES} entities")
        return cls(atom_id=atom_id, text=clean_text, entities=tuple(displays))

    @property
    def entity_keys(self) -> tuple[str, ...]:
        return tuple(normalize_entity(e) for e in self.entities)

"""Rule-based query analysis: sub-query decomposition and explicit entity extraction (<1 ms)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import plain_key as normalize_entity

_SPLIT = re.compile(
    r"[;?]+|\b(?:и\s+как|а\s+также|а\s+затем|после\s+чего|при\s+этом|"
    r"and\s+how|as\s+well\s+as|whereas|compared\s+to|and\s+then)\b",
    re.IGNORECASE | re.UNICODE,
)
_QUOTED = re.compile(r"[\"«“]([^\"»”]{2,128})[\"»”]")
_TOKEN = re.compile(r"\w[\w.+\-]*", re.UNICODE)
_MIN_SUBQUERY_WORDS = 2


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    sub_queries: tuple[str, ...]
    explicit_entities: tuple[str, ...]
    token_count: int


def _looks_like_entity(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return False
    if any(c.isdigit() for c in token):  # L2A, GPT-4o, Llama-3.1
        return True
    if len(letters) >= 2 and all(c.isupper() for c in letters):  # AEG, PPR, RAG
        return True
    return any(c.isupper() for c in token[1:])  # AtomRAG, DeepSeek


def analyze_query(query: str) -> QueryAnalysis:
    text = query.strip()
    parts = [p.strip(" ,.:") for p in _SPLIT.split(text)]
    subs = tuple(p for p in parts if len(_TOKEN.findall(p)) >= _MIN_SUBQUERY_WORDS)
    if not subs:
        subs = (text,)

    seen: set[str] = set()
    entities: list[str] = []
    candidates = [m.group(1).strip() for m in _QUOTED.finditer(text)]
    candidates += [t.rstrip(".+-") for t in _TOKEN.findall(text) if _looks_like_entity(t.rstrip(".+-"))]
    for cand in candidates:
        key = normalize_entity(cand)
        if key and key not in seen:
            seen.add(key)
            entities.append(cand)
    return QueryAnalysis(
        sub_queries=subs,
        explicit_entities=tuple(entities),
        token_count=len(_TOKEN.findall(text)),
    )

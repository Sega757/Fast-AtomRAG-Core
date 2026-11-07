"""Entity-key normalization (ADR-008): one function for graph keys AND query matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from .stemming import stem_token

# Tokens: letter/digit runs, internal dots kept ("3.1"); hyphens, underscores, slashes split.
_TOKEN = re.compile(r"[^\W_]+(?:\.[^\W_]+)*", re.UNICODE)

KeyFn = Callable[[str], str]


def key_tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return _TOKEN.findall(folded)


def plain_key(text: str) -> str:
    """Casefold + ё->е + collapse spaces/hyphens. No stemming."""
    return " ".join(key_tokens(text))


def stemmed_key(text: str) -> str:
    """plain_key + per-token Snowball-RU / Porter-EN stemming."""
    return " ".join(stem_token(t) for t in key_tokens(text))


def char_trigrams(key: str) -> frozenset[str]:
    padded = f" {key} "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))

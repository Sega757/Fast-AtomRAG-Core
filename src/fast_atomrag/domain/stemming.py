"""Pure-stdlib stemmers: Snowball Russian and the original Porter (1980) English algorithm.

Deterministic and dependency-free (ADR-008). Both are cached per token, so the hot path is
a dict lookup after warm-up.
"""

from __future__ import annotations

import re
from functools import lru_cache

# --------------------------------------------------------------------------------------
# Russian Snowball (https://snowballstem.org/algorithms/russian/stemmer.html)
# --------------------------------------------------------------------------------------
_RU_VOWELS = "аеиоуыэюя"


def _alt(endings: str) -> str:
    """Space-separated endings -> regex alternation (longest first)."""
    return "|".join(sorted(endings.split(), key=len, reverse=True))


def _group(after_a_ya: str, free: str) -> str:
    """Group-1 endings must be preceded by "а"/"я" inside RV; group-2 endings need not."""
    return rf"(?:(?<=[ая])(?:{_alt(after_a_ya)})|(?:{_alt(free)}))"


# `re.search(...$)` returns the leftmost match, i.e. the LONGEST matching ending.
_PERFECTIVE_GERUND = re.compile(_group("в вши вшись", "ив ивши ившись ыв ывши ывшись") + "$")
_REFLEXIVE = re.compile(r"(?:ся|сь)$")
_ADJECTIVE = _alt("ее ие ые ое ими ыми ей ий ый ой ем им ым ом его ого ему ому их ых ую юю ая яя ою ею")
_PARTICIPLE = _group("ем нн вш ющ щ", "ивш ывш ующ")
_ADJECTIVAL = re.compile(rf"{_PARTICIPLE}?(?:{_ADJECTIVE})$")
_VERB = re.compile(
    _group(
        "ла на ете йте ли й л ем н ло но ет ют ны ть ешь нно",
        "ила ыла ена ейте уйте ите или ыли ей уй ил ыл им ым ен ило ыло ено ят ует уют ит ыт ены ить ыть ишь ую ю",
    )
    + "$"
)
_NOUN_ENDINGS = (
    "а ев ов ие ье е иями ями ами еи ии и ией ей ой ий й иям ям ием ем ам ом о у ах иях ях ы ь ию ью ю ия ья я"
)
_NOUN = re.compile(rf"(?:{_alt(_NOUN_ENDINGS)})$")
_DERIVATIONAL = re.compile(r"(?:ость|ост)$")
_SUPERLATIVE = re.compile(r"(?:ейше|ейш)$")


def _ru_regions(word: str) -> tuple[int, int]:
    """Return (rv, r2) start offsets."""
    rv = len(word)
    for i, ch in enumerate(word):
        if ch in _RU_VOWELS:
            rv = i + 1
            break

    def next_region(start: int) -> int:
        for i in range(start + 1, len(word)):
            if word[i] not in _RU_VOWELS and word[i - 1] in _RU_VOWELS:
                return i + 1
        return len(word)

    r1 = next_region(0)
    r2 = next_region(r1)
    return rv, r2


def _cut(rx: re.Pattern[str], s: str) -> tuple[str, bool]:
    m = rx.search(s)
    return (s[: m.start()], True) if m else (s, False)


@lru_cache(maxsize=65536)
def stem_ru(word: str) -> str:
    word = word.replace("ё", "е")
    rv, r2 = _ru_regions(word)
    head, rv_part = word[:rv], word[rv:]

    # Step 1
    rv_part, done = _cut(_PERFECTIVE_GERUND, rv_part)
    if not done:
        rv_part, _ = _cut(_REFLEXIVE, rv_part)
        for rx in (_ADJECTIVAL, _VERB, _NOUN):
            rv_part, done = _cut(rx, rv_part)
            if done:
                break
    # Step 2
    if rv_part.endswith("и"):
        rv_part = rv_part[:-1]
    # Step 3: derivational ending inside R2
    m = _DERIVATIONAL.search(rv_part)
    if m and rv + m.start() >= r2:
        rv_part = rv_part[: m.start()]
    # Step 4
    if rv_part.endswith("нн"):
        rv_part = rv_part[:-1]
    else:
        rv_part, cut = _cut(_SUPERLATIVE, rv_part)
        if (cut and rv_part.endswith("нн")) or rv_part.endswith("ь"):
            rv_part = rv_part[:-1]
    return head + rv_part


# --------------------------------------------------------------------------------------
# English Porter (1980)
# --------------------------------------------------------------------------------------
def _is_cons(w: str, i: int) -> bool:
    ch = w[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == 0 or not _is_cons(w, i - 1)
    return True


def _measure(stem: str) -> int:
    m, i, n = 0, 0, len(stem)
    while i < n and _is_cons(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_cons(stem, i):
            i += 1
        if i >= n:
            break
        while i < n and _is_cons(stem, i):
            i += 1
        m += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    return (
        len(w) >= 3
        and _is_cons(w, len(w) - 3)
        and not _is_cons(w, len(w) - 2)
        and _is_cons(w, len(w) - 1)
        and w[-1] not in "wxy"
    )


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
)
_STEP3 = (("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""))
_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ion",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def _replace_longest(w: str, rules: tuple[tuple[str, str], ...], min_m: int) -> str:
    best = max((r for r in rules if w.endswith(r[0])), key=lambda r: len(r[0]), default=None)
    if best is None:
        return w
    stem = w[: -len(best[0])]
    return stem + best[1] if _measure(stem) > min_m else w


@lru_cache(maxsize=65536)
def stem_en(word: str) -> str:
    w = word
    if len(w) <= 2:
        return w
    # Step 1a
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # Step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suffix in ("ed", "ing"):
            if w.endswith(suffix) and _has_vowel(w[: -len(suffix)]):
                w = w[: -len(suffix)]
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif _double_cons(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _cvc(w):
                    w += "e"
                break
    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # Steps 2-3
    w = _replace_longest(w, _STEP2, 0)
    w = _replace_longest(w, _STEP3, 0)
    # Step 4
    best = max((s for s in _STEP4 if w.endswith(s)), key=len, default=None)
    if best is not None:
        stem = w[: -len(best)]
        if _measure(stem) > 1 and (best != "ion" or stem.endswith(("s", "t"))):
            w = stem
    # Step 5a
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            w = stem
    # Step 5b
    if _measure(w) > 1 and _double_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w


_CYRILLIC = re.compile(r"[а-я]")
_LATIN_ONLY = re.compile(r"[a-z]+")


def stem_token(token: str) -> str:
    """Stem a lowercase token; tokens with digits or mixed scripts are returned unchanged."""
    if len(token) <= 2 or any(ch.isdigit() for ch in token):
        return token
    if _LATIN_ONLY.fullmatch(token):
        return stem_en(token)
    if _CYRILLIC.search(token) and not re.search(r"[a-z]", token):
        return stem_ru(token)
    return token

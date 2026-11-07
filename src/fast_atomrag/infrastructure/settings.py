"""Environment configuration, validated at startup (fail-fast)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class ConfigError(ValueError):
    """Invalid or inconsistent configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: str = "./data/atoms.db"
    api_key: str | None = None
    embed_dim: int = 512
    alpha: float = 0.65
    fusion: str = "rrf"
    entity_stemming: bool = True
    fuzzy_threshold: float = 0.75
    candidate_k: int = 50
    ppr_restart: float = 0.15
    ppr_eps: float = 1e-4
    ppr_max_pushes: int = 20_000
    sla_seconds: float = 2.0
    log_level: str = "INFO"


def _num(env: Mapping[str, str], name: str, default: float, lo: float, hi: float, cast: type) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name}={value} must be in [{lo}, {hi}]")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    api_key = env.get("ATOMRAG_API_KEY") or None
    if api_key is not None and len(api_key) < 16:
        raise ConfigError("ATOMRAG_API_KEY must be at least 16 characters")
    log_level = env.get("ATOMRAG_LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError(f"ATOMRAG_LOG_LEVEL={log_level!r} is invalid")
    fusion = env.get("ATOMRAG_FUSION", "rrf").lower()
    if fusion not in {"rrf", "linear"}:
        raise ConfigError(f"ATOMRAG_FUSION={fusion!r} must be 'rrf' or 'linear'")
    stemming_raw = env.get("ATOMRAG_ENTITY_STEMMING", "true").strip().lower()
    if stemming_raw not in {"true", "false", "1", "0"}:
        raise ConfigError("ATOMRAG_ENTITY_STEMMING must be true/false")
    return Settings(
        db_path=env.get("ATOMRAG_DB_PATH", Settings.db_path),
        api_key=api_key,
        embed_dim=int(_num(env, "ATOMRAG_EMBED_DIM", 512, 64, 4096, int)),
        alpha=_num(env, "ATOMRAG_ALPHA", 0.65, 0.0, 1.0, float),
        fusion=fusion,
        entity_stemming=stemming_raw in {"true", "1"},
        fuzzy_threshold=_num(env, "ATOMRAG_FUZZY_THRESHOLD", 0.75, 0.0, 1.0, float),
        candidate_k=int(_num(env, "ATOMRAG_CANDIDATE_K", 50, 1, 1000, int)),
        ppr_restart=_num(env, "ATOMRAG_PPR_RESTART", 0.15, 0.01, 0.99, float),
        ppr_eps=_num(env, "ATOMRAG_PPR_EPS", 1e-4, 1e-8, 1e-1, float),
        ppr_max_pushes=int(_num(env, "ATOMRAG_PPR_MAX_PUSHES", 20_000, 100, 10_000_000, int)),
        sla_seconds=_num(env, "ATOMRAG_SLA_SECONDS", 2.0, 0.05, 120.0, float),
        log_level=log_level,
    )

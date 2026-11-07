"""SQLite atom store: the durable source of truth. In-memory index is derived from it (ADR-007/010)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np

from ..application.errors import StorageError
from ..application.ports import StoredAtom
from ..domain.models import Atom

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000
T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS atoms (
    atom_id     INTEGER PRIMARY KEY,
    text        TEXT    NOT NULL,
    entities    TEXT    NOT NULL,
    embedder_id TEXT    NOT NULL,
    vector      BLOB    NOT NULL
);
"""

_UPSERT = """
INSERT INTO atoms (atom_id, text, entities, embedder_id, vector)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(atom_id) DO UPDATE SET
    text = excluded.text,
    entities = excluded.entities,
    embedder_id = excluded.embedder_id,
    vector = excluded.vector
"""


class SQLiteAtomRepository:
    """Not thread-safe by itself; the service serializes access with its lock.

    Every write runs in one transaction (`with conn:` -> COMMIT or ROLLBACK). sqlite3 errors are
    translated to StorageError, which the service treats as "nothing changed".
    """

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=BUSY_TIMEOUT_MS / 1000)
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if path != ":memory:" and str(mode).lower() != "wal":
            raise StorageError(f"could not enable WAL journal mode (got {mode!r})")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema v{version} is newer than supported v{SCHEMA_VERSION}")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def pragmas(self) -> dict[str, object]:
        names = ("journal_mode", "synchronous", "busy_timeout", "foreign_keys")
        return {n: self._conn.execute(f"PRAGMA {n}").fetchone()[0] for n in names}

    @staticmethod
    def _row(atom: Atom, embedder_id: str, vector: np.ndarray) -> tuple[int, str, str, str, bytes]:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        return atom.atom_id, atom.text, json.dumps(list(atom.entities), ensure_ascii=False), embedder_id, blob

    def _tx(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        try:
            with self._conn:
                return fn(self._conn)
        except sqlite3.Error as exc:
            raise StorageError(f"storage operation failed: {exc.__class__.__name__}") from exc

    def upsert(self, atom: Atom, embedder_id: str, vector: np.ndarray) -> bool:
        def op(conn: sqlite3.Connection) -> bool:
            exists = conn.execute("SELECT 1 FROM atoms WHERE atom_id = ?", (atom.atom_id,)).fetchone()
            conn.execute(_UPSERT, self._row(atom, embedder_id, vector))
            return exists is None

        return self._tx(op)

    def upsert_many(self, items: Sequence[tuple[Atom, np.ndarray]], embedder_id: str) -> None:
        rows = [self._row(a, embedder_id, v) for a, v in items]
        self._tx(lambda conn: conn.executemany(_UPSERT, rows))

    def delete(self, atom_id: int) -> bool:
        return self._tx(lambda conn: conn.execute("DELETE FROM atoms WHERE atom_id = ?", (atom_id,)).rowcount > 0)

    def load_all(self) -> Iterator[StoredAtom]:
        try:
            cursor = self._conn.execute(
                "SELECT atom_id, text, entities, embedder_id, vector FROM atoms ORDER BY atom_id"
            )
            for atom_id, text, entities, embedder_id, blob in cursor:
                atom = Atom.create(atom_id, text, json.loads(entities))
                yield StoredAtom(atom, embedder_id, np.frombuffer(blob, dtype=np.float32).copy())
        except sqlite3.Error as exc:
            raise StorageError(f"storage read failed: {exc.__class__.__name__}") from exc

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

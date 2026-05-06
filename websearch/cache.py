"""SQLite-backed disk cache for HTTP responses.

Keyed by (method, url, body_hash). Stores status, headers, body, and a
fetch timestamp. Callers decide TTL via the `max_age` parameter on get().
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path(os.environ.get("WEBSEARCH_CACHE_DIR", Path.home() / ".cache" / "websearch"))
DEFAULT_DB = DEFAULT_DIR / "cache.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    method      TEXT NOT NULL,
    url         TEXT NOT NULL,
    final_url   TEXT NOT NULL,
    status      INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    body        BLOB NOT NULL,
    fetched_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_url ON responses(url);
"""


@dataclass
class CachedResponse:
    method: str
    url: str
    final_url: str
    status: int
    content_type: str
    body: str
    fetched_at: float
    age_seconds: float


def _key(method: str, url: str, data: Optional[dict] = None) -> str:
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(b"\x00")
    h.update(url.encode())
    if data:
        h.update(b"\x00")
        h.update(json.dumps(data, sort_keys=True).encode())
    return h.hexdigest()


class Cache:
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the cache can be used from worker threads
        # in fetch_many. We pair it with an explicit Lock for writes.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL improves concurrent read/write behavior.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def get(
        self,
        method: str,
        url: str,
        data: Optional[dict] = None,
        max_age: Optional[float] = None,
    ) -> Optional[CachedResponse]:
        """Return a cached response if it exists and is fresh enough.

        max_age: maximum age in seconds. None means no expiry check.
        """
        key = _key(method, url, data)
        with self._lock:
            row = self._conn.execute(
                "SELECT method, url, final_url, status, content_type, body, fetched_at "
                "FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        fetched_at = row[6]
        age = time.time() - fetched_at
        if max_age is not None and age > max_age:
            return None
        return CachedResponse(
            method=row[0],
            url=row[1],
            final_url=row[2],
            status=row[3],
            content_type=row[4],
            body=row[5].decode("utf-8", errors="replace"),
            fetched_at=fetched_at,
            age_seconds=age,
        )

    def put(
        self,
        method: str,
        url: str,
        final_url: str,
        status: int,
        content_type: str,
        body: str,
        data: Optional[dict] = None,
    ) -> None:
        key = _key(method, url, data)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, method, url, final_url, status, content_type, body, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    method.upper(),
                    url,
                    final_url,
                    status,
                    content_type,
                    body.encode("utf-8"),
                    time.time(),
                ),
            )
            self._conn.commit()

    def clear(self, older_than: Optional[float] = None) -> int:
        """Delete entries older than `older_than` seconds. Returns count deleted."""
        with self._lock:
            if older_than is None:
                cur = self._conn.execute("DELETE FROM responses")
            else:
                cutoff = time.time() - older_than
                cur = self._conn.execute(
                    "DELETE FROM responses WHERE fetched_at < ?", (cutoff,)
                )
            self._conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), MIN(fetched_at), MAX(fetched_at), SUM(LENGTH(body)) FROM responses"
            ).fetchone()
        return {
            "count": row[0] or 0,
            "oldest": row[1],
            "newest": row[2],
            "total_body_bytes": row[3] or 0,
            "path": str(self.path),
        }


_default: Optional[Cache] = None


def default() -> Cache:
    global _default
    if _default is None:
        _default = Cache()
    return _default

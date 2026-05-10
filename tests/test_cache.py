"""Tests for the SQLite-backed response cache."""
import time

import pytest

from websearch.cache import Cache


@pytest.fixture
def cache(tmp_path):
    return Cache(path=tmp_path / "cache.db")


def test_put_get_roundtrip(cache):
    cache.put("GET", "https://x.test/a", "https://x.test/a", 200, "text/html", "<p>hi</p>")
    got = cache.get("GET", "https://x.test/a")
    assert got is not None
    assert got.body == "<p>hi</p>"
    assert got.status == 200
    assert got.content_type == "text/html"
    assert got.age_seconds >= 0


def test_get_miss_returns_none(cache):
    assert cache.get("GET", "https://x.test/missing") is None


def test_max_age_expiry(cache):
    cache.put("GET", "https://x.test/b", "https://x.test/b", 200, "text/html", "body")
    # Within window
    assert cache.get("GET", "https://x.test/b", max_age=60) is not None
    # Force expiry by asking for an impossibly tight window
    time.sleep(0.05)
    assert cache.get("GET", "https://x.test/b", max_age=0.001) is None


def test_clear_all(cache):
    cache.put("GET", "https://x.test/c", "https://x.test/c", 200, "text/html", "x")
    cache.put("GET", "https://x.test/d", "https://x.test/d", 200, "text/html", "y")
    n = cache.clear()
    assert n == 2
    assert cache.get("GET", "https://x.test/c") is None


def test_clear_older_than(cache):
    cache.put("GET", "https://x.test/e", "https://x.test/e", 200, "text/html", "z")
    # Nothing is older than 1 hour yet
    assert cache.clear(older_than=3600) == 0
    # Everything is older than 0 seconds
    assert cache.clear(older_than=0) == 1


def test_idx_url_was_dropped_on_open(cache):
    """The legacy idx_url index must not exist after opening — it was dropped
    because all lookups go through the primary key."""
    rows = cache._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "idx_url" not in names


def test_stats_reports_count_and_path(cache):
    cache.put("GET", "https://x.test/f", "https://x.test/f", 200, "text/html", "body")
    s = cache.stats()
    assert s["count"] == 1
    assert s["total_body_bytes"] >= len("body")
    assert "cache.db" in s["path"]

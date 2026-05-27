"""Tests for the SearXNG backend and MCP server scaffolding."""
import importlib

import pytest

from websearch import core


def test_since_to_searxng_range():
    assert core._since_to_searxng_range("d") == "day"
    assert core._since_to_searxng_range("w") == "week"
    assert core._since_to_searxng_range("m") == "month"
    assert core._since_to_searxng_range("y") == "year"
    # Precise dates / unknown buckets -> no coarse range (enforce-since handles it)
    assert core._since_to_searxng_range("2026-01-01") is None
    assert core._since_to_searxng_range(None) is None


class _FakeResp:
    def __init__(self, json_data, content_type="application/json"):
        self._json = json_data
        self.headers = {"Content-Type": content_type}
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_search_searxng_parses_json(monkeypatch):
    payload = {
        "results": [
            {"url": "https://a.example/1", "title": "First", "content": "snip one"},
            {"url": "https://b.example/2", "title": "Second", "content": "snip two"},
            {"title": "no url — skipped", "content": "x"},
        ]
    }
    monkeypatch.setattr(core, "_do_request", lambda *a, **k: _FakeResp(payload))
    results = core.search_searxng("anything", "http://searx.local")
    assert [r.url for r in results] == ["https://a.example/1", "https://b.example/2"]
    assert results[0].title == "First"
    assert results[0].snippet == "snip one"


def test_search_searxng_non_json_raises(monkeypatch):
    monkeypatch.setattr(
        core, "_do_request", lambda *a, **k: _FakeResp("<html/>", content_type="text/html")
    )
    with pytest.raises(RuntimeError, match="json"):
        core.search_searxng("q", "http://searx.local")


def test_local_host_detection():
    assert core._local_host("http://localhost:8888")
    assert core._local_host("http://127.0.0.1:8888/")
    assert not core._local_host("https://searx.example.com")


def test_searxng_start_cmd(tmp_path):
    # An existing file is treated as a compose file.
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n")
    cmd = core._searxng_start_cmd("/usr/local/bin/docker", str(compose))
    assert cmd == ["/usr/local/bin/docker", "compose", "-f", str(compose), "up", "-d"]
    # A non-path is treated as a container name.
    cmd = core._searxng_start_cmd("/usr/local/bin/docker", "searxng")
    assert cmd == ["/usr/local/bin/docker", "start", "searxng"]


def test_ensure_searxng_already_up_is_noop(monkeypatch):
    monkeypatch.setattr(core, "searxng_reachable", lambda *a, **k: True)
    # Reachable -> True without ever touching docker (would raise if it did).
    monkeypatch.setattr(core, "_find_docker", lambda: (_ for _ in ()).throw(AssertionError))
    assert core.ensure_searxng("http://localhost:8888", "searxng") is True


def test_ensure_searxng_no_autostart_returns_false(monkeypatch):
    monkeypatch.setattr(core, "searxng_reachable", lambda *a, **k: False)
    # Down + no autostart spec -> False (caller falls back to DDG/Bing).
    assert core.ensure_searxng("http://localhost:8888", None) is False
    # Down + autostart but remote URL -> never autostart a remote instance.
    assert core.ensure_searxng("https://searx.example.com", "searxng") is False


def test_mcp_server_module_shape():
    # The module imports cleanly (only stdlib + core). build_server/run exist;
    # build_server raises a helpful ImportError when `mcp` is not installed.
    mod = importlib.import_module("websearch.mcp_server")
    assert callable(mod.build_server)
    assert callable(mod.run)
    if importlib.util.find_spec("mcp") is None:
        with pytest.raises(ImportError, match="pipx inject"):
            mod.build_server()
    else:
        assert mod.build_server() is not None

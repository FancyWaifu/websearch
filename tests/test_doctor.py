"""Tests for doctor — version parsing, age computation, optional-dep presence."""
from __future__ import annotations

import datetime as _dt
import subprocess
from unittest import mock

import pytest

from websearch.doctor import (
    _have_yt_dlp_ejs,
    _yt_dlp_probe,
    _yt_dlp_version_and_age,
)
from websearch.transcripts import _yt_dlp_runtime_args


def _fake_run(stdout: str, rc: int = 0):
    """Return a callable that mimics subprocess.run returning fixed stdout/rc.
    Class body doesn't close over outer scope, so build a SimpleNamespace."""
    from types import SimpleNamespace
    result = SimpleNamespace(
        returncode=rc,
        stdout=stdout.encode("utf-8"),
        stderr=b"",
    )
    return lambda *a, **kw: result


def test_version_and_age_returns_none_when_path_missing():
    v, age = _yt_dlp_version_and_age(None)
    assert v is None and age is None


def test_version_and_age_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr("websearch.doctor.subprocess.run", _fake_run("", rc=2))
    v, age = _yt_dlp_version_and_age("/fake/yt-dlp")
    assert v is None and age is None


def test_version_and_age_parses_standard_version():
    """Real yt-dlp prints `2026.03.17\n`; verify we parse the date out."""
    with mock.patch("websearch.doctor.subprocess.run", _fake_run("2026.03.17\n")):
        v, age = _yt_dlp_version_and_age("/fake/yt-dlp")
    assert v == "2026.03.17"
    # Compute expected age relative to today so the test doesn't rot.
    today = _dt.date.today()
    expected = (today - _dt.date(2026, 3, 17)).days
    assert age == expected


def test_version_and_age_handles_unparseable_version():
    """Some yt-dlp builds (e.g. nightly forks) print non-YYYY.MM.DD strings.
    Return the raw version but no age — better than crashing."""
    with mock.patch("websearch.doctor.subprocess.run", _fake_run("nightly-abc123\n")):
        v, age = _yt_dlp_version_and_age("/fake/yt-dlp")
    assert v == "nightly-abc123"
    assert age is None


def test_version_and_age_handles_invalid_date():
    """A version-shaped string that doesn't form a real date (e.g. month 13)
    should also degrade gracefully."""
    with mock.patch("websearch.doctor.subprocess.run", _fake_run("2026.13.17\n")):
        v, age = _yt_dlp_version_and_age("/fake/yt-dlp")
    assert v == "2026.13.17"
    assert age is None


def test_version_and_age_handles_subprocess_exception(monkeypatch):
    def boom(*a, **kw):
        raise OSError("permission denied")
    monkeypatch.setattr("websearch.doctor.subprocess.run", boom)
    v, age = _yt_dlp_version_and_age("/fake/yt-dlp")
    assert v is None and age is None


def test_have_yt_dlp_ejs_returns_bool():
    """Just verify the check doesn't crash and returns a bool — the actual
    True/False depends on what's installed in the test env."""
    assert isinstance(_have_yt_dlp_ejs(), bool)


# ---------- yt_dlp_runtime_args + probe ----------

def test_runtime_args_defaults_to_node(monkeypatch):
    monkeypatch.delenv("WEBSEARCH_YT_JS_RUNTIME", raising=False)
    assert _yt_dlp_runtime_args() == ["--js-runtimes", "node"]


def test_runtime_args_respects_env_override(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_YT_JS_RUNTIME", "deno")
    assert _yt_dlp_runtime_args() == ["--js-runtimes", "deno"]


def test_runtime_args_empty_env_opts_out(monkeypatch):
    """Empty string means 'fall back to yt-dlp's own default' — no flag passed."""
    monkeypatch.setenv("WEBSEARCH_YT_JS_RUNTIME", "")
    assert _yt_dlp_runtime_args() == []


def test_yt_dlp_probe_returns_empty_when_path_missing():
    assert _yt_dlp_probe(None) == {}


def test_yt_dlp_probe_parses_debug_lines(monkeypatch):
    """Probe runs `yt-dlp --verbose --simulate <bad-url>` and scrapes the
    [debug] Optional libraries + [debug] JS runtimes lines from combined
    stdout+stderr."""
    fake_stderr = b"""[debug] Loading plugins...
[debug] Optional libraries: brotli-1.1.0, mutagen-1.47.0, yt_dlp_ejs-0.8.0
[debug] JS runtimes: node-24.7.0
ERROR: Unable to download: ...
"""
    monkeypatch.setattr(
        "websearch.doctor.subprocess.run",
        lambda *a, **kw: _fake_run("", rc=1)(*a, **kw).__class__(
            returncode=1, stdout=b"", stderr=fake_stderr,
        ),
    )
    out = _yt_dlp_probe("/fake/yt-dlp")
    assert out["yt_dlp_ejs"] is True
    assert out["js_runtimes"] == "node-24.7.0"
    assert "yt_dlp_ejs" in out.get("optional_libs", "")
    assert out["runtime_requested"] == "node"


def test_yt_dlp_probe_detects_missing_ejs(monkeypatch):
    fake_stderr = b"""[debug] Optional libraries: brotli-1.1.0, mutagen-1.47.0
[debug] JS runtimes: none
"""
    from types import SimpleNamespace
    monkeypatch.setattr(
        "websearch.doctor.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout=b"", stderr=fake_stderr,
        ),
    )
    out = _yt_dlp_probe("/fake/yt-dlp")
    assert out["yt_dlp_ejs"] is False
    assert out["js_runtimes"] == "none"


def test_yt_dlp_probe_uses_env_runtime(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_YT_JS_RUNTIME", "bun")
    captured = []

    def fake_run(cmd, **kw):
        captured.append(list(cmd))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")

    monkeypatch.setattr("websearch.doctor.subprocess.run", fake_run)
    _yt_dlp_probe("/fake/yt-dlp")
    assert "--js-runtimes" in captured[0]
    idx = captured[0].index("--js-runtimes")
    assert captured[0][idx + 1] == "bun"

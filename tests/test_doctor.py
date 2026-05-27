"""Tests for doctor — version parsing, age computation, optional-dep presence."""
from __future__ import annotations

import datetime as _dt
import subprocess
from unittest import mock

import pytest

from websearch.doctor import _yt_dlp_version_and_age, _have_yt_dlp_ejs


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

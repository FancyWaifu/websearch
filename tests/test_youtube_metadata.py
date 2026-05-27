"""Tests for YouTube metadata enrichment and Whisper plumbing.

Pure-function coverage of the header formatting. Whisper subprocess paths
are exercised via mocks where useful and skipped where they'd require the
optional faster-whisper dep.
"""
from __future__ import annotations

from unittest import mock

import pytest

from websearch.transcripts import (
    _format_chapters,
    _format_duration,
    _format_upload_date,
    _format_video_header,
    _format_views,
    _video_id,
    _yt_dlp_error_hint,
    fetch_metadata,
)


# ---------- Small helpers ----------

@pytest.mark.parametrize(
    "seconds,want",
    [
        (None, ""),
        (0, ""),       # 0 treated as missing
        (45, "45s"),
        (90, "1m30s"),
        (3600, "1h00m"),
        (24180, "6h43m"),
        (3661, "1h01m"),
    ],
)
def test_format_duration(seconds, want):
    assert _format_duration(seconds) == want


@pytest.mark.parametrize(
    "n,want",
    [
        (None, ""),
        (0, ""),
        (5, "5"),
        (999, "999"),
        (1500, "1.5K"),
        (1_234_567, "1.2M"),
        (45_000_000, "45.0M"),
    ],
)
def test_format_views(n, want):
    assert _format_views(n) == want


@pytest.mark.parametrize(
    "raw,want",
    [
        (None, ""),
        ("", ""),
        ("20240115", "2024-01-15"),
        ("19991231", "1999-12-31"),
        ("2024-01-15", "2024-01-15"),  # already formatted -> passthrough
        ("garbage", "garbage"),
    ],
)
def test_format_upload_date(raw, want):
    assert _format_upload_date(raw) == want


# ---------- Chapters ----------

def test_format_chapters_renders_mm_ss_for_short_videos():
    chapters = [
        {"start_time": 0, "title": "Intro"},
        {"start_time": 90, "title": "Setup"},
        {"start_time": 360, "title": "The main argument"},
    ]
    out = _format_chapters(chapters)
    assert "00:00  Intro" in out
    assert "01:30  Setup" in out
    assert "06:00  The main argument" in out


def test_format_chapters_renders_h_mm_ss_for_long_videos():
    chapters = [{"start_time": 3700, "title": "Late chapter"}]
    out = _format_chapters(chapters)
    assert "1:01:40  Late chapter" in out


def test_format_chapters_truncates_long_lists():
    chapters = [{"start_time": i * 10, "title": f"Ch {i}"} for i in range(50)]
    out = _format_chapters(chapters, limit=5)
    assert out.count("\n") == 5  # 5 chapter lines + 1 "more" line - 1 (no trailing nl)
    assert "(45 more)" in out


def test_format_chapters_handles_empty_and_missing():
    assert _format_chapters(None) == ""
    assert _format_chapters([]) == ""
    # Chapters with no title get skipped.
    out = _format_chapters([{"start_time": 0}, {"start_time": 30, "title": "real"}])
    assert "real" in out
    assert "00:00" not in out  # the title-less one was dropped


# ---------- Header assembly ----------

def test_format_video_header_assembles_full_card():
    meta = {
        "title": "Intro to LLMs",
        "channel": "Andrej Karpathy",
        "duration": 3720,
        "upload_date": "20231122",
        "view_count": 2_500_000,
        "description": "A talk about large language models.\nWith some details.",
        "chapters": [
            {"start_time": 0, "title": "Intro"},
            {"start_time": 600, "title": "Architecture"},
        ],
    }
    out = _format_video_header(meta)
    assert out.startswith("# YouTube: Intro to LLMs\n")
    assert "Channel: Andrej Karpathy" in out
    assert "Duration: 1h02m" in out
    assert "Uploaded: 2023-11-22" in out
    assert "Views: 2.5M" in out
    assert "_A talk about large language models." in out
    assert "Chapters:" in out
    assert "00:00  Intro" in out
    assert "10:00  Architecture" in out
    assert out.rstrip().endswith("--- Transcript ---")


def test_format_video_header_skips_missing_fields():
    meta = {"title": "Just a title"}
    out = _format_video_header(meta)
    assert "# YouTube: Just a title" in out
    assert "Channel" not in out
    assert "Duration" not in out
    assert "--- Transcript ---" in out  # always emits the separator


def test_format_video_header_empty_returns_empty():
    assert _format_video_header({}) == ""


def test_format_video_header_truncates_long_description():
    desc = "A" * 1000
    meta = {"title": "x", "description": desc}
    out = _format_video_header(meta)
    # 240-char limit + "..." marker
    assert "..." in out
    assert "A" * 240 in out
    assert "A" * 241 not in out


# ---------- Video ID parsing ----------

@pytest.mark.parametrize(
    "url,want",
    [
        ("https://www.youtube.com/watch?v=zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://youtu.be/zjkBMFhNj_g", "zjkBMFhNj_g"),
        ("https://youtube.com/shorts/AbCdEfGhIjK", "AbCdEfGhIjK"),
        ("https://www.youtube.com/embed/AbCdEfGhIjK", "AbCdEfGhIjK"),
        ("https://example.com/notayoutubeurl", None),
        ("https://www.youtube.com/feed/subscriptions", None),
    ],
)
def test_video_id_extraction(url, want):
    assert _video_id(url) == want


# ---------- fetch_metadata subprocess wiring ----------

def _stub_oembed(monkeypatch, returns: dict) -> None:
    """Patch the private oembed helper to return a fixed dict — avoids real
    network on tests that only care about the yt-dlp branch."""
    monkeypatch.setattr(
        "websearch.transcripts._fetch_metadata_via_oembed",
        lambda url, timeout: returns,
    )


def test_fetch_metadata_returns_empty_when_both_paths_fail(monkeypatch):
    monkeypatch.setattr("websearch.transcripts.have_yt_dlp", lambda: False)
    _stub_oembed(monkeypatch, {})
    assert fetch_metadata("https://youtu.be/zjkBMFhNj_g") == {}


def test_fetch_metadata_falls_back_to_oembed_when_yt_dlp_missing(monkeypatch):
    monkeypatch.setattr("websearch.transcripts.have_yt_dlp", lambda: False)
    _stub_oembed(monkeypatch, {"title": "T", "channel": "C", "_source": "oembed"})
    meta = fetch_metadata("https://youtu.be/zjkBMFhNj_g")
    assert meta["title"] == "T"
    assert meta["channel"] == "C"


def test_fetch_metadata_parses_yt_dlp_json(monkeypatch):
    monkeypatch.setattr("websearch.transcripts.have_yt_dlp", lambda: True)
    fake_json = (
        b'{"title": "T", "channel": "C", "duration": 60, "upload_date": "20240101",'
        b' "view_count": 100, "description": "d", "chapters": []}\n'
    )

    class _R:
        returncode = 0
        stdout = fake_json
        stderr = b""

    monkeypatch.setattr("websearch.transcripts.subprocess.run",
                        lambda *a, **kw: _R())
    # If yt-dlp returns something, the oembed fallback must NOT be called —
    # blow up if it is so the test catches regressions.
    monkeypatch.setattr(
        "websearch.transcripts._fetch_metadata_via_oembed",
        lambda *a, **kw: pytest.fail("oembed should not run when yt-dlp succeeds"),
    )
    meta = fetch_metadata("https://youtu.be/zjkBMFhNj_g")
    assert meta["title"] == "T"
    assert meta["channel"] == "C"
    assert meta["view_count"] == 100


# ---------- yt-dlp error hint surfacing ----------

def test_yt_dlp_error_hint_empty_input_returns_empty():
    assert _yt_dlp_error_hint("") == ""
    assert _yt_dlp_error_hint("totally unrelated error") == ""


def test_yt_dlp_error_hint_sign_in_to_confirm():
    err = "ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies..."
    out = _yt_dlp_error_hint(err)
    assert "WEBSEARCH_YT_COOKIES_FROM" in out
    assert "firefox" in out


def test_yt_dlp_error_hint_suppresses_signin_when_cookies_set():
    """If cookies are already configured, telling the user to set cookies is
    counterproductive — they did, and yt-dlp is still failing for a different
    reason. Only fire the OTHER hints in that case."""
    err = "Sign in to confirm — also n challenge solving failed"
    with_cookies = _yt_dlp_error_hint(err, cookies_set=True)
    assert "WEBSEARCH_YT_COOKIES_FROM" not in with_cookies
    assert "yt-dlp-ejs" in with_cookies


def test_yt_dlp_error_hint_n_challenge():
    err = "WARNING: [youtube] x: n challenge solving failed"
    out = _yt_dlp_error_hint(err)
    assert "yt-dlp-ejs" in out
    assert "EJS" in out


def test_yt_dlp_error_hint_po_token():
    err = "tv_simply client formats require a GVS PO Token"
    out = _yt_dlp_error_hint(err)
    assert "PO Token" in out
    assert "bgutil-ytdlp-pot-provider" in out or "PO-Token-Guide" in out


def test_yt_dlp_error_hint_format_unavailable():
    err = "ERROR: Requested format is not available"
    out = _yt_dlp_error_hint(err)
    assert "yt-dlp-ejs" in out or "yt-dlp[default]" in out


def test_yt_dlp_error_hint_concatenates_multiple_matches():
    err = ("Sign in to confirm — n challenge solving failed — "
           "GVS PO Token required")
    out = _yt_dlp_error_hint(err)
    # All three hints should appear; the joiner is " — "
    assert out.startswith(" — ")
    assert "WEBSEARCH_YT_COOKIES_FROM" in out
    assert "yt-dlp-ejs" in out
    assert "PO Token" in out


def test_fetch_metadata_falls_back_to_oembed_on_yt_dlp_failure(monkeypatch):
    monkeypatch.setattr("websearch.transcripts.have_yt_dlp", lambda: True)

    class _R:
        returncode = 1
        stdout = b""
        stderr = b"Sign in to confirm"

    monkeypatch.setattr("websearch.transcripts.subprocess.run",
                        lambda *a, **kw: _R())
    _stub_oembed(monkeypatch, {"title": "from oembed", "channel": "ch"})
    meta = fetch_metadata("https://youtu.be/zjkBMFhNj_g")
    assert meta["title"] == "from oembed"

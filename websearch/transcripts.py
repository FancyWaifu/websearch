"""YouTube transcript fetch.

The built-in HTML fetcher returns ~zero usable text for a YouTube page —
the transcript is loaded as captions, not as readable body content. This
module detects YouTube URLs and pulls captions through a two-tier path:

1. `youtube-transcript-api` (optional pip dep) — calls YouTube's transcript
   JSON endpoint directly. Doesn't trip the bot-check that started rejecting
   yt-dlp in early 2026 (`Sign in to confirm you're not a bot`).
2. `yt-dlp` — fallback when the API lib isn't installed or doesn't have a
   transcript for the video. Honors $WEBSEARCH_YT_COOKIES_FROM (values:
   safari / chrome / firefox / edge) to pass browser cookies and clear
   YouTube's bot challenge.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

YT_HOST_RE = re.compile(r"(?:^|\.)(youtube\.com|youtu\.be)$", re.IGNORECASE)


def is_youtube_url(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return bool(YT_HOST_RE.search(host))


def have_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


_VTT_CUE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*$",
    re.MULTILINE,
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")
_VTT_HEADER_RE = re.compile(r"^WEBVTT.*?(?=\n\n|\Z)", re.DOTALL | re.MULTILINE)


def _vtt_to_text(vtt: str) -> str:
    """Strip VTT cues/timestamps/tags and dedupe adjacent identical lines.

    YouTube auto-subs duplicate lines as they roll on screen — dedup makes the
    output 3-5× shorter without losing content.
    """
    body = _VTT_HEADER_RE.sub("", vtt)
    body = _VTT_CUE_RE.sub("", body)
    body = _VTT_TAG_RE.sub("", body)
    lines = [ln.strip() for ln in body.splitlines()]
    out: list[str] = []
    prev = ""
    for ln in lines:
        if not ln or ln == prev:
            continue
        if ln.isdigit():  # cue index
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out)


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def _video_id(url: str) -> Optional[str]:
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


def have_youtube_transcript_api() -> bool:
    try:
        import youtube_transcript_api  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_via_api(url: str, lang: str) -> tuple[str, Optional[str]]:
    """Try youtube-transcript-api first. Returns ("", err) if the lib isn't
    installed or the video has no transcript reachable that way."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:
        return "", "youtube-transcript-api not installed"
    vid = _video_id(url)
    if not vid:
        return "", "could not extract a YouTube video ID from the URL"
    try:
        # Library API has shifted across versions (0.6→1.x). Try both shapes.
        try:
            entries = YouTubeTranscriptApi.get_transcript(vid, languages=[lang, "en"])
        except AttributeError:
            api = YouTubeTranscriptApi()
            entries = api.fetch(vid, languages=[lang, "en"]).to_raw_data()
    except Exception as e:  # noqa: BLE001
        return "", f"youtube-transcript-api: {e}"
    parts = [(e.get("text", "") or "").strip() for e in entries]
    text = "\n".join(p for p in parts if p)
    return (text, None) if text else ("", "youtube-transcript-api: empty transcript")


def fetch_transcript(url: str, timeout: int = 60, lang: str = "en") -> tuple[str, Optional[str]]:
    """Return (text, error). On error, text is "" and error explains why.

    Tries youtube-transcript-api first (no auth, no bot-check), falls back
    to yt-dlp + browser-cookies if configured.
    """
    text, api_err = _fetch_via_api(url, lang)
    if text:
        return text, None

    if not have_yt_dlp():
        return "", (
            f"{api_err}; yt-dlp not installed either — "
            "`pipx inject websearch youtube-transcript-api` for the API path, "
            "or `brew install yt-dlp` for the fallback."
        )

    with tempfile.TemporaryDirectory() as tmp:
        out_tpl = str(Path(tmp) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-lang", lang,
            "--sub-format", "vtt",
            "-o", out_tpl,
        ]
        # Browser-cookies path is the supported workaround for YouTube's
        # bot challenge — yt-dlp can't fetch transcripts as anonymous from
        # most ASNs as of early 2026.
        cookies_from = os.environ.get("WEBSEARCH_YT_COOKIES_FROM")
        if cookies_from:
            cmd += ["--cookies-from-browser", cookies_from]
        cmd += ["--", url]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return "", f"yt-dlp timed out after {timeout}s (api: {api_err})"
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            hint = ""
            if "Sign in to confirm" in err and not cookies_from:
                hint = (" — set $WEBSEARCH_YT_COOKIES_FROM=safari (or "
                        "chrome/firefox/edge) so yt-dlp can pass browser "
                        "cookies, or install youtube-transcript-api")
            return "", f"yt-dlp rc={r.returncode}: {err[:300]}{hint} (api: {api_err})"
        vtts = sorted(Path(tmp).glob("*.vtt"))
        if not vtts:
            return "", (
                f"no subtitles produced (video may have none / be private / "
                f"geo-locked) (api: {api_err})"
            )
        raw = vtts[0].read_text(encoding="utf-8", errors="replace")
        return _vtt_to_text(raw), None

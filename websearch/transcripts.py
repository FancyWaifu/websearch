"""YouTube transcript fetch via yt-dlp.

The built-in HTML fetcher returns ~zero usable text for a YouTube page —
the transcript is loaded as captions, not as readable body content. This
module detects YouTube URLs and shells out to yt-dlp to grab auto-subs
without downloading the video, then strips the VTT to plain prose.
"""
from __future__ import annotations

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


def fetch_transcript(url: str, timeout: int = 60, lang: str = "en") -> tuple[str, Optional[str]]:
    """Return (text, error). On error, text is "" and error explains why."""
    if not have_yt_dlp():
        return "", (
            "yt-dlp not installed — `brew install yt-dlp` (or `pipx install yt-dlp`) "
            "to enable YouTube transcript fetch."
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
            url,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return "", f"yt-dlp timed out after {timeout}s"
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            return "", f"yt-dlp rc={r.returncode}: {err[:300]}"
        vtts = sorted(Path(tmp).glob("*.vtt"))
        if not vtts:
            return "", "no subtitles produced (video may have none / be private / geo-locked)"
        raw = vtts[0].read_text(encoding="utf-8", errors="replace")
        return _vtt_to_text(raw), None

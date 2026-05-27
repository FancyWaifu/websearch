"""YouTube transcript + metadata + (optional) Whisper fallback.

The built-in HTML fetcher returns ~zero usable text for a YouTube page —
the transcript is loaded as captions, not as readable body content. This
module detects YouTube URLs and pulls captions through a tiered path:

1. `youtube-transcript-api` (optional pip dep) — calls YouTube's transcript
   JSON endpoint directly. Doesn't trip the bot-check that started rejecting
   yt-dlp in early 2026 (`Sign in to confirm you're not a bot`).
2. `yt-dlp` — fallback when the API lib isn't installed or doesn't have a
   transcript for the video. Honors $WEBSEARCH_YT_COOKIES_FROM (values:
   safari / chrome / firefox / edge) to pass browser cookies and clear
   YouTube's bot challenge.
3. `faster-whisper` (opt-in via `use_whisper=True`) — for captionless
   videos. Downloads the worst-quality audio stream via yt-dlp (it gets
   resampled to 16kHz mono regardless) and transcribes locally.

Successful transcripts are prepended with a markdown header carrying
title, channel, duration, upload date, view count, and chapter markers
so the reader gets a "card" they can navigate without playing the video.
Metadata is fetched separately via `yt-dlp --dump-json --skip-download`;
failure to get metadata downgrades silently — never blocks a transcript.
"""
from __future__ import annotations

import json
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


def _format_duration(seconds: Optional[float]) -> str:
    """24180 -> '6h43m'. Returns '' if seconds is missing."""
    if not seconds:
        return ""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _format_views(n: Optional[int]) -> str:
    """1234567 -> '1.2M'. Returns '' if n is missing."""
    if not n:
        return ""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _format_upload_date(s: Optional[str]) -> str:
    """'20240115' -> '2024-01-15'. Returns the input if it doesn't parse."""
    if not s or len(s) != 8 or not s.isdigit():
        return s or ""
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _format_chapters(chapters: Optional[list], limit: int = 30) -> str:
    """Render yt-dlp chapter list as `MM:SS Title` lines. Empty → ''."""
    if not chapters:
        return ""
    out: list[str] = []
    for ch in chapters[:limit]:
        start = ch.get("start_time") or 0
        title = (ch.get("title") or "").strip()
        if not title:
            continue
        s = int(start)
        if s >= 3600:
            ts = f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
        else:
            ts = f"{s//60:02d}:{s%60:02d}"
        out.append(f"  {ts}  {title}")
    if len(chapters) > limit:
        out.append(f"  ... ({len(chapters) - limit} more)")
    return "\n".join(out)


def _fetch_metadata_via_yt_dlp(url: str, timeout: int) -> dict:
    """Try yt-dlp --dump-json. Bot-checked without cookies on most ASNs in
    2026, so this often returns {}; callers fall back to oembed."""
    if not have_yt_dlp():
        return {}
    cmd = ["yt-dlp", "--quiet", "--no-warnings", "--skip-download",
           "--dump-json", *_yt_dlp_runtime_args()]
    cookies_from = os.environ.get("WEBSEARCH_YT_COOKIES_FROM")
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    cmd += ["--", url]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {}
    if r.returncode != 0 or not r.stdout:
        return {}
    try:
        # --dump-json emits one JSON object per video; --skip-download keeps
        # it to a single object for a single URL.
        return json.loads(r.stdout.decode("utf-8", errors="replace").splitlines()[0])
    except Exception:  # noqa: BLE001
        return {}


def _fetch_metadata_via_oembed(url: str, timeout: int) -> dict:
    """No-auth fallback: YouTube's public oembed endpoint.

    Returns a thinner dict (title + channel only — no duration, chapters,
    view count, upload date, or description) but works without cookies and
    against the same bot-checked endpoints that block yt-dlp. We normalize
    the field names to match what `_format_video_header` expects.
    """
    import urllib.parse
    import urllib.request

    api = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({
        "url": url, "format": "json",
    })
    try:
        with urllib.request.urlopen(api, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return {}
    return {
        "title": data.get("title") or "",
        "channel": data.get("author_name") or "",
        # Mark as oembed-sourced so callers can tell it's the thin variant
        # if they ever need to (no behavioural use yet).
        "_source": "oembed",
    }


def fetch_metadata(url: str, timeout: int = 15) -> dict:
    """Pull video metadata. Tries yt-dlp first (rich: duration/chapters/
    views/upload date), then YouTube's oembed endpoint as a no-auth
    fallback (thin: title + channel only).

    Returns {} when both sources fail — metadata is pure enrichment, never
    a hard requirement for the transcript path.
    """
    meta = _fetch_metadata_via_yt_dlp(url, timeout)
    if meta:
        return meta
    return _fetch_metadata_via_oembed(url, timeout)


_DESC_MAX_CHARS = 240


def _format_video_header(meta: dict) -> str:
    """Render the metadata dict as a markdown header block. Empty meta → ''."""
    if not meta:
        return ""
    title = (meta.get("title") or "").strip()
    channel = (meta.get("channel") or meta.get("uploader") or "").strip()
    duration = _format_duration(meta.get("duration"))
    uploaded = _format_upload_date(meta.get("upload_date"))
    views = _format_views(meta.get("view_count"))
    description = (meta.get("description") or "").strip()
    chapters_md = _format_chapters(meta.get("chapters"))

    lines: list[str] = []
    if title:
        lines.append(f"# YouTube: {title}")
    facts = []
    if channel:
        facts.append(f"Channel: {channel}")
    if duration:
        facts.append(f"Duration: {duration}")
    if uploaded:
        facts.append(f"Uploaded: {uploaded}")
    if views:
        facts.append(f"Views: {views}")
    if facts:
        lines.append(" | ".join(facts))
    if description:
        truncated = description[:_DESC_MAX_CHARS]
        if len(description) > _DESC_MAX_CHARS:
            truncated += "..."
        # Collapse internal newlines so the description stays in one block.
        truncated = re.sub(r"\s*\n\s*", " ", truncated)
        lines.append("")
        lines.append(f"_{truncated}_")
    if chapters_md:
        lines.append("")
        lines.append("Chapters:")
        lines.append(chapters_md)
    lines.append("")
    lines.append("--- Transcript ---")
    return "\n".join(lines) + "\n"


def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def have_faster_whisper() -> bool:
    """Public alias used by doctor + cli."""
    return _have_faster_whisper()


def _yt_dlp_runtime_args() -> list[str]:
    """yt-dlp defaults to enabling only `deno` as a JS runtime, but Node is
    far more commonly installed. Without an enabled runtime, YouTube's
    n-challenge can't be solved and ALL downloadable formats are filtered
    out — yt-dlp returns "Requested format is not available". Explicitly
    enable a runtime; default to `node` since it's the common case.

    Override via $WEBSEARCH_YT_JS_RUNTIME (e.g. `deno`, `bun`). Empty
    string opts out entirely (keeps yt-dlp's default behavior).
    """
    runtime = os.environ.get("WEBSEARCH_YT_JS_RUNTIME", "node")
    return ["--js-runtimes", runtime] if runtime else []


def _fetch_via_whisper(url: str, timeout: int = 600) -> tuple[str, Optional[str]]:
    """Download audio via yt-dlp + transcribe locally with faster-whisper.

    Opt-in only (gated by --whisper). Uses the lowest-bitrate audio stream
    since faster-whisper resamples to 16kHz mono anyway. Model size comes
    from $WEBSEARCH_WHISPER_MODEL (default 'small'); larger models give
    better accuracy at significant CPU/time cost.
    """
    if not have_yt_dlp():
        return "", "whisper: yt-dlp not installed (needed to download audio)"
    if not _have_faster_whisper():
        return "", ("whisper: faster-whisper not installed — "
                    "`pipx inject websearch faster-whisper`")
    with tempfile.TemporaryDirectory() as tmp:
        out_tpl = str(Path(tmp) / "%(id)s.%(ext)s")
        # Whisper resamples to 16kHz mono internally, so audio quality
        # doesn't matter — but format AVAILABILITY does. `bestaudio/worst`
        # is yt-dlp's documented "always finds something" spec; pure
        # `worstaudio` returns "Requested format is not available" on some
        # videos. `--audio-quality 9` (lowest VBR mp3) then keeps the
        # post-extract file small regardless of the source bitrate.
        cmd = [
            "yt-dlp", "--quiet", "--no-warnings",
            *_yt_dlp_runtime_args(),
            "-f", "bestaudio/worst",
            "-x", "--audio-format", "mp3", "--audio-quality", "9",
            "-o", out_tpl,
        ]
        cookies_from = os.environ.get("WEBSEARCH_YT_COOKIES_FROM")
        if cookies_from:
            cmd += ["--cookies-from-browser", cookies_from]
        cmd += ["--", url]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout // 2,
                               check=False)
        except subprocess.TimeoutExpired:
            return "", f"whisper: yt-dlp audio download timed out (>{timeout//2}s)"
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            hint = _yt_dlp_error_hint(err, cookies_set=bool(cookies_from))
            return "", (
                f"whisper: yt-dlp audio download failed: {err[:300]}{hint}"
            )
        audios = sorted(Path(tmp).glob("*.mp3"))
        if not audios:
            audios = sorted(Path(tmp).glob("*"))  # whatever yt-dlp produced
        if not audios:
            return "", "whisper: yt-dlp produced no audio file"

        from faster_whisper import WhisperModel  # type: ignore
        model_size = os.environ.get("WEBSEARCH_WHISPER_MODEL", "small")
        try:
            # CPU is the only universally-available compute target; the
            # int8 quantization runs cleanly on Apple Silicon and modest
            # x86 boxes alike. Users with CUDA can override via env.
            device = os.environ.get("WEBSEARCH_WHISPER_DEVICE", "cpu")
            compute = os.environ.get("WEBSEARCH_WHISPER_COMPUTE", "int8")
            model = WhisperModel(model_size, device=device, compute_type=compute)
        except Exception as e:  # noqa: BLE001
            return "", f"whisper: model load failed ({model_size!r}): {e}"
        try:
            segments, _info = model.transcribe(str(audios[0]), beam_size=1)
            parts = [(seg.text or "").strip() for seg in segments]
        except Exception as e:  # noqa: BLE001
            return "", f"whisper: transcription failed: {e}"
        text = "\n".join(p for p in parts if p)
        return (text, None) if text else ("", "whisper: empty transcript")


def fetch_transcript(
    url: str,
    timeout: int = 60,
    lang: str = "en",
    use_whisper: bool = False,
    include_metadata: bool = True,
) -> tuple[str, Optional[str]]:
    """Return (text, error). On error, text is "" and error explains why.

    Tiered path: youtube-transcript-api (no auth) -> yt-dlp + browser
    cookies -> faster-whisper (only if `use_whisper=True`). Successful
    output is prepended with a markdown header carrying title/channel/
    duration/upload date/views/chapters when `include_metadata=True`.
    """
    def _wrap(transcript: str) -> str:
        if not include_metadata or not transcript:
            return transcript
        header = _format_video_header(fetch_metadata(url))
        return f"{header}\n{transcript}" if header else transcript

    text, api_err = _fetch_via_api(url, lang)
    if text:
        return _wrap(text), None

    yt_err: Optional[str] = None
    if have_yt_dlp():
        text, yt_err = _fetch_via_yt_dlp_subs(url, timeout, lang)
        if text:
            return _wrap(text), None

    if use_whisper:
        text, w_err = _fetch_via_whisper(url, timeout=max(timeout, 600))
        if text:
            return _wrap(text), None
        return "", _combine_errors(api_err, yt_err, w_err)

    if not have_yt_dlp():
        return "", (
            f"{api_err}; yt-dlp not installed either — "
            "`pipx inject websearch youtube-transcript-api` for the API path, "
            "or `brew install yt-dlp` for the fallback."
        )
    return "", _combine_errors(api_err, yt_err, None)


def _fetch_via_yt_dlp_subs(
    url: str, timeout: int, lang: str
) -> tuple[str, Optional[str]]:
    """yt-dlp --write-auto-subs path. Returns (text, error)."""
    with tempfile.TemporaryDirectory() as tmp:
        out_tpl = str(Path(tmp) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            *_yt_dlp_runtime_args(),
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
            return "", f"yt-dlp timed out after {timeout}s"
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            hint = _yt_dlp_error_hint(err, cookies_set=bool(cookies_from))
            return "", f"yt-dlp rc={r.returncode}: {err[:300]}{hint}"
        vtts = sorted(Path(tmp).glob("*.vtt"))
        if not vtts:
            return "", ("no subtitles produced (video may have none / be "
                        "private / geo-locked; --whisper would transcribe "
                        "from audio)")
        raw = vtts[0].read_text(encoding="utf-8", errors="replace")
        return _vtt_to_text(raw), None


def _combine_errors(*errs: Optional[str]) -> str:
    """Join non-empty error strings with `; `."""
    return "; ".join(e for e in errs if e)


# YouTube's mid-2026 anti-yt-dlp escalation surfaced several new failure
# modes whose default error text is unhelpful ("Requested format is not
# available" doesn't tell you to install yt-dlp-ejs). Each entry pairs a
# stderr fragment with an actionable hint; multiple matches concatenate so
# the user sees the whole picture in one error.
_YT_DLP_HINT_PATTERNS = (
    (
        "Sign in to confirm",
        "set $WEBSEARCH_YT_COOKIES_FROM=firefox (or chrome/edge — Safari "
        "cookies are sandboxed on macOS and need Full Disk Access to read)",
    ),
    (
        "n challenge solving failed",
        "install the n-challenge solver: `pip install yt-dlp-ejs` and make "
        "sure node/deno is on PATH (yt-dlp/wiki/EJS)",
    ),
    (
        "GVS PO Token",
        "yt-dlp needs a PO Token: install `bgutil-ytdlp-pot-provider` or "
        "follow yt-dlp/wiki/PO-Token-Guide",
    ),
    (
        "Requested format is not available",
        "YouTube isn't serving downloadable formats to this client; usually "
        "fixed by `pip install -U 'yt-dlp[default]' yt-dlp-ejs` and "
        "$WEBSEARCH_YT_COOKIES_FROM=firefox",
    ),
    (
        "SABR-only",
        "YouTube enabled SABR streaming for this account/IP; try "
        "`--extractor-args 'youtube:player_client=web_safari,mweb'`",
    ),
)


def _yt_dlp_error_hint(stderr: str, *, cookies_set: bool = False) -> str:
    """Return one consolidated hint for known yt-dlp failures, or "".

    `cookies_set=True` suppresses the Sign-in hint (the user already did
    the cookie step, the failure is something else)."""
    if not stderr:
        return ""
    hints: list[str] = []
    for pattern, hint in _YT_DLP_HINT_PATTERNS:
        if pattern == "Sign in to confirm" and cookies_set:
            continue
        if pattern in stderr:
            hints.append(hint)
    return " — " + "; ".join(hints) if hints else ""

"""Self-diagnostic: where am I installed, what's on PATH, can I reach my proxy.

Motivated by the very real foot-gun where a stale console-script shim from
an older Python install wins over the active pipx install on PATH and the
user sees `ModuleNotFoundError: No module named 'websearch'` with no hint
about which binary they're actually running.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def _have_yt_api() -> bool:
    try:
        import youtube_transcript_api  # noqa: F401
        return True
    except ImportError:
        return False


def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _path_entries() -> list[str]:
    return [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]


def all_websearch_on_path() -> list[str]:
    """Every executable named `websearch` reachable via PATH, in PATH order."""
    found: list[str] = []
    seen: set[str] = set()
    for d in _path_entries():
        cand = Path(d) / "websearch"
        if cand.is_file() and os.access(cand, os.X_OK):
            real = str(cand.resolve())
            if real not in seen:
                seen.add(real)
                found.append(str(cand))
    return found


def active_binary() -> str:
    """Best guess at the binary currently being run."""
    if sys.argv and sys.argv[0]:
        return str(Path(sys.argv[0]).resolve())
    return ""


def module_path() -> str:
    import websearch
    return str(Path(websearch.__file__).parent.resolve())


def proxy_status(proxy: Optional[str] = None) -> dict:
    """Resolve the proxy URL and check whether the host:port is reachable."""
    p = proxy or os.environ.get("WEBSEARCH_PROXY") or ""
    out: dict = {"configured": bool(p), "url": p}
    if not p:
        out["note"] = "No proxy set ($WEBSEARCH_PROXY unset, no --proxy)."
        return out
    try:
        parsed = urlparse(p)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 1080
        out["host"] = host
        out["port"] = port
        with socket.create_connection((host, port), timeout=2):
            out["reachable"] = True
    except (OSError, ValueError) as e:
        out["reachable"] = False
        out["error"] = str(e)
    return out


def searxng_status(searxng_url: Optional[str] = None) -> dict:
    """Check whether SearXNG is configured and serving JSON.

    SearXNG is `search_smart`'s first-choice backend, but it's also the
    most likely thing to be misconfigured — env vars don't propagate to
    non-interactive shells, the instance can be down, or its `json` format
    can be disabled in settings.yml. `doctor` would silently leave the
    user thinking results are degraded when really the primary engine
    isn't being hit at all.
    """
    import requests

    url = searxng_url or os.environ.get("WEBSEARCH_SEARXNG") or ""
    autostart = os.environ.get("WEBSEARCH_SEARXNG_AUTOSTART") or ""
    out: dict = {
        "configured": bool(url),
        "url": url,
        "autostart": autostart,
    }
    if not url:
        out["note"] = (
            "No SearXNG set ($WEBSEARCH_SEARXNG unset, no --searxng) — "
            "search falls back to DuckDuckGo/Bing scraping."
        )
        return out
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        out["host"] = host
        out["port"] = port
        with socket.create_connection((host, port), timeout=2):
            pass
        out["reachable"] = True
    except (OSError, ValueError) as e:
        out["reachable"] = False
        out["error"] = str(e)
        return out
    try:
        r = requests.get(
            f"{url.rstrip('/')}/search",
            params={"q": "websearch doctor probe", "format": "json"},
            timeout=5,
        )
        out["http_status"] = r.status_code
        ct = r.headers.get("Content-Type", "")
        if "json" not in ct.lower():
            out["json_format"] = False
            out["error"] = (
                f"instance returned Content-Type {ct!r} — enable the 'json' "
                "format under search.formats in settings.yml"
            )
        else:
            data = r.json()
            out["json_format"] = True
            out["result_count"] = len(data.get("results") or [])
            unresp = data.get("unresponsive_engines") or []
            if unresp:
                out["unresponsive_engines"] = [
                    f"{name} ({reason})" for name, reason in unresp
                ]
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def report(proxy: Optional[str] = None) -> dict:
    """Collect everything into one structured dict."""
    from . import __version__
    from .cache import default as default_cache
    from . import pdfx

    cands = all_websearch_on_path()
    active = active_binary()
    shim_warning: Optional[str] = None
    if len(cands) > 1:
        shim_warning = (
            f"multiple `websearch` binaries on PATH — first wins: {cands[0]}. "
            "If that's a stale shim from a previous Python install, prefer calling "
            "the pipx install directly (e.g., ~/.local/bin/websearch)."
        )

    rep: dict = {
        "version": __version__,
        "active_binary": active,
        "module_path": module_path(),
        "path_websearch_binaries": cands,
        "shim_warning": shim_warning,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "proxy": proxy_status(proxy),
        "searxng": searxng_status(),
        "pdftotext": {
            "available": pdfx.have_pdftotext(),
            "path": shutil.which("pdftotext"),
        },
        "yt_dlp": {
            "available": shutil.which("yt-dlp") is not None,
            "path": shutil.which("yt-dlp"),
        },
        "youtube_transcript_api": {
            "available": _have_yt_api(),
        },
        "faster_whisper": {
            "available": _have_faster_whisper(),
            "model": os.environ.get("WEBSEARCH_WHISPER_MODEL", "small"),
        },
        "yt_cookies_from": os.environ.get("WEBSEARCH_YT_COOKIES_FROM", ""),
    }
    try:
        rep["cache"] = default_cache().stats()
    except Exception as e:
        rep["cache"] = {"error": str(e)}
    return rep


def format_human(rep: dict) -> str:
    """Render the report as a readable text block."""
    lines: list[str] = []
    lines.append(f"websearch {rep.get('version','?')}")
    lines.append(f"  active binary : {rep.get('active_binary','?')}")
    lines.append(f"  module path   : {rep.get('module_path','?')}")
    py = rep.get("python", {})
    lines.append(f"  python        : {py.get('version','?')} ({py.get('executable','?')})")

    cands = rep.get("path_websearch_binaries", [])
    lines.append("")
    lines.append("PATH binaries (first wins):")
    if not cands:
        lines.append("  (none — only the active binary's dir on PATH?)")
    for c in cands:
        marker = "  *" if c == rep.get("active_binary") else "   "
        lines.append(f"{marker} {c}")
    if rep.get("shim_warning"):
        lines.append("")
        lines.append(f"!! {rep['shim_warning']}")

    pr = rep.get("proxy", {})
    lines.append("")
    lines.append("Proxy:")
    if not pr.get("configured"):
        lines.append(f"  not configured — {pr.get('note','')}")
    else:
        reach = "OK" if pr.get("reachable") else f"UNREACHABLE ({pr.get('error','?')})"
        lines.append(f"  {pr.get('url')}  [{reach}]")

    sx = rep.get("searxng", {})
    lines.append("")
    lines.append("SearXNG (preferred search backend):")
    if not sx.get("configured"):
        lines.append(f"  not configured — {sx.get('note','')}")
    elif not sx.get("reachable"):
        lines.append(f"  {sx.get('url')}  [UNREACHABLE ({sx.get('error','?')})]")
        if sx.get("autostart"):
            lines.append(f"  autostart: {sx['autostart']}")
    elif sx.get("json_format") is False:
        lines.append(f"  {sx.get('url')}  [JSON FORMAT DISABLED — {sx.get('error','?')}]")
    else:
        rc = sx.get("result_count", 0)
        lines.append(f"  {sx.get('url')}  [OK, smoke probe returned {rc} results]")
        if sx.get("autostart"):
            lines.append(f"  autostart: {sx['autostart']}")
        unresp = sx.get("unresponsive_engines") or []
        if unresp:
            lines.append(f"  unresponsive: {', '.join(unresp)}")

    tools = []
    info = rep.get("pdftotext", {})
    mark = "ok" if info.get("available") else "missing"
    tools.append(f"  pdftotext  : {mark}  ({info.get('path') or '-'})")

    # yt_dlp itself works fine, but YouTube has bot-checked the anonymous
    # path since early 2026 — without cookies, transcript fetches fail with
    # "Sign in to confirm you're not a bot". Mark accordingly so `doctor`
    # doesn't lull the user into thinking YT transcripts will Just Work.
    yt_info = rep.get("yt_dlp", {})
    yt_api = rep.get("youtube_transcript_api", {})
    yt_cookies = rep.get("yt_cookies_from") or ""
    if not yt_info.get("available"):
        yt_mark = "missing"
    elif yt_cookies:
        yt_mark = f"ok  (YT cookies: {yt_cookies})"
    elif yt_api.get("available"):
        yt_mark = "ok  (YT transcripts route via yt_api; yt_dlp fallback bot-checked without $WEBSEARCH_YT_COOKIES_FROM)"
    else:
        yt_mark = "ok* (YT transcripts BROKEN without $WEBSEARCH_YT_COOKIES_FROM=safari|chrome|firefox|edge or yt_api)"
    tools.append(f"  yt_dlp     : {yt_mark}  ({yt_info.get('path') or '-'})")

    yt_api_mark = "ok" if yt_api.get("available") else "missing (pipx inject websearch youtube-transcript-api)"
    tools.append(f"  yt_api     : {yt_api_mark}")
    fw = rep.get("faster_whisper", {})
    if fw.get("available"):
        tools.append(f"  whisper    : ok  (faster-whisper, model={fw.get('model','small')})")
    else:
        tools.append("  whisper    : missing (pipx inject websearch faster-whisper) — needed for --whisper on captionless YT")
    lines.append("")
    lines.append("External tools:")
    lines.extend(tools)

    c = rep.get("cache", {})
    lines.append("")
    if "error" in c:
        lines.append(f"Cache: error: {c['error']}")
    else:
        lines.append(
            f"Cache: {c.get('count',0)} entries, {(c.get('total_body_bytes',0) or 0)/1024/1024:.1f} MB "
            f"at {c.get('path','?')}"
        )
    return "\n".join(lines)

"""Self-diagnostic: where am I installed, what's on PATH, can I reach my proxy.

Motivated by the very real foot-gun where a stale console-script shim from
an older Python install wins over the active pipx install on PATH and the
user sees `ModuleNotFoundError: No module named 'websearch'` with no hint
about which binary they're actually running.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import socket
import subprocess
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


def _yt_dlp_info() -> dict:
    """yt-dlp availability + version + age in one place."""
    path = shutil.which("yt-dlp")
    version, age_days = _yt_dlp_version_and_age(path)
    return {
        "available": path is not None,
        "path": path,
        "version": version,
        "age_days": age_days,
    }


def _have_yt_dlp_ejs() -> bool:
    """Import-time check — accurate only if yt-dlp lives in THIS Python.
    Most users have yt-dlp installed in a separate env (system python /
    homebrew / pipx-isolated), so this returns False even when yt-dlp
    can find yt-dlp-ejs in its own env. Kept for tests; doctor uses
    `_yt_dlp_probe` which queries yt-dlp's actual environment."""
    try:
        import yt_dlp_ejs  # noqa: F401
        return True
    except ImportError:
        return False


def _yt_dlp_probe(path: Optional[str], runtime: Optional[str] = None) -> dict:
    """Ask yt-dlp itself what optional libraries it sees and which JS
    runtimes it can use. Runs `yt-dlp --verbose --simulate <fake-url>`
    which is cheap (no network) and emits the `[debug] Optional libraries`
    + `[debug] JS runtimes` lines we need. ~200ms.

    The JS runtime question matters because yt-dlp defaults to only
    enabling `deno` — even with node/bun installed and yt_dlp_ejs present,
    YouTube's n-challenge can't be solved unless the user passes
    `--js-runtimes node` (or sets $WEBSEARCH_YT_JS_RUNTIME so websearch
    passes it). Without that, audio downloads fail with
    "Requested format is not available".
    """
    if not path:
        return {}
    runtime = runtime if runtime is not None else os.environ.get(
        "WEBSEARCH_YT_JS_RUNTIME", "node")
    cmd = [path, "--verbose", "--simulate", "https://websearch.probe.invalid/"]
    if runtime:
        cmd[1:1] = ["--js-runtimes", runtime]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        return {}
    text = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    out: dict = {"runtime_requested": runtime}
    for line in text.splitlines():
        if line.startswith("[debug] Optional libraries:"):
            libs = line.split(":", 1)[1].strip()
            out["yt_dlp_ejs"] = "yt_dlp_ejs" in libs
            out["optional_libs"] = libs
        elif line.startswith("[debug] JS runtimes:"):
            out["js_runtimes"] = line.split(":", 1)[1].strip()
    return out


# Known-stable YouTube video used for the n-challenge end-to-end probe.
# A short, public, captioned video that's been live for years — if
# yt-dlp can extract real audio formats for this one, the EJS toolchain
# is actually working, not just superficially installed.
_PROBE_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", 19s
_PROBE_FORMAT_RE = re.compile(r"^\d+\s+\w+\s+\d+x\d+", re.MULTILINE)


def n_challenge_probe(yt_dlp_path: Optional[str], runtime: Optional[str] = None,
                      timeout: float = 15.0) -> dict:
    """Actually solve YouTube's n-challenge end-to-end. The libs-installed
    checks (yt_dlp_ejs + JS runtime) can both be 'ok' while extraction
    still fails because YouTube changed something this week. This probe
    runs `yt-dlp --list-formats` on a known-stable test video and counts
    the non-storyboard formats returned. >5 = working; 0 (only `sb` storyboard
    formats) = broken; failed run = broken.

    Returns dict with keys:
        - status: 'ok' | 'broken' | 'unknown'
        - format_count: int (non-storyboard formats; storyboard-only = broken)
        - error: optional str when the run failed outright
    """
    if not yt_dlp_path:
        return {"status": "unknown", "format_count": 0,
                "error": "yt-dlp not installed"}
    runtime = runtime if runtime is not None else os.environ.get(
        "WEBSEARCH_YT_JS_RUNTIME", "node")
    cmd = [yt_dlp_path, "--quiet", "--no-warnings", "--list-formats", "--skip-download"]
    if runtime:
        cmd[1:1] = ["--js-runtimes", runtime]
    cookies = os.environ.get("WEBSEARCH_YT_COOKIES_FROM")
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    cmd += ["--", _PROBE_VIDEO]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"status": "broken", "format_count": 0, "error": str(e)}
    text = r.stdout.decode("utf-8", errors="replace")
    # Format-line pattern: ID EXT RESOLUTION ... — storyboard rows are
    # "sb0 mhtml 48x27" which doesn't match the `\d+ \w+ \d+x\d+` shape.
    formats = _PROBE_FORMAT_RE.findall(text)
    return {
        "status": "ok" if formats else "broken",
        "format_count": len(formats),
        "error": None if formats else "no playable formats returned (n-challenge or PO Token)",
    }


_YT_DLP_VERSION_RE = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})")


def _yt_dlp_version_and_age(path: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Return (version_str, age_in_days). YouTube/yt-dlp move fast in 2026 —
    anything older than ~60 days is likely missing fixes for new YouTube
    anti-scraping changes. Returns (None, None) if yt-dlp is missing or the
    version output can't be parsed."""
    if not path:
        return None, None
    try:
        r = subprocess.run([path, "--version"], capture_output=True, timeout=5)
    except Exception:
        return None, None
    if r.returncode != 0:
        return None, None
    ver = r.stdout.decode("utf-8", errors="replace").strip()
    m = _YT_DLP_VERSION_RE.search(ver)
    if not m:
        return ver, None
    try:
        d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return ver, None
    age = (_dt.date.today() - d).days
    return ver, age


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
        "yt_dlp": _yt_dlp_info(),
        # Probe yt-dlp's own env (NOT this Python) — yt-dlp typically lives in
        # a separate venv, so a Python-side import check would give the wrong
        # answer. See _yt_dlp_probe docstring.
        "yt_dlp_probe": _yt_dlp_probe(shutil.which("yt-dlp")),
        # End-to-end probe: ask yt-dlp to list formats for a known-stable
        # test video. Costs ~2-5s of network, but it's the only way to tell
        # "libs installed" from "actually working today" given YouTube's
        # weekly anti-yt-dlp changes.
        "yt_dlp_n_challenge": n_challenge_probe(shutil.which("yt-dlp")),
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
    # path since early 2026 — without cookies (and as of mid-2026, an EJS
    # solver + PO Token), transcript fetches and audio downloads fail.
    yt_info = rep.get("yt_dlp", {})
    yt_api = rep.get("youtube_transcript_api", {})
    yt_probe = rep.get("yt_dlp_probe", {})
    yt_cookies = rep.get("yt_cookies_from") or ""
    if not yt_info.get("available"):
        yt_mark = "missing"
    else:
        ver = yt_info.get("version") or "?"
        age = yt_info.get("age_days")
        age_part = ""
        if age is not None:
            if age > 60:
                age_part = f" — STALE ({age}d old; YouTube changes break old yt-dlp, run `pip install -U 'yt-dlp[default]'`)"
            else:
                age_part = f" ({age}d old)"
        if yt_cookies:
            yt_mark = f"ok  v{ver}{age_part}  (YT cookies: {yt_cookies})"
        elif yt_api.get("available"):
            yt_mark = f"ok  v{ver}{age_part}  (YT transcripts route via yt_api; yt_dlp fallback bot-checked without $WEBSEARCH_YT_COOKIES_FROM)"
        else:
            yt_mark = f"ok* v{ver}{age_part}  (YT transcripts BROKEN without $WEBSEARCH_YT_COOKIES_FROM=firefox|chrome|edge or yt_api)"
    tools.append(f"  yt_dlp     : {yt_mark}  ({yt_info.get('path') or '-'})")

    yt_api_mark = "ok" if yt_api.get("available") else "missing (pipx inject websearch youtube-transcript-api)"
    tools.append(f"  yt_api     : {yt_api_mark}")

    # EJS + JS runtime status comes from probing yt-dlp's own env (not the
    # websearch venv). Both pieces are needed for YouTube audio downloads
    # (transcript --whisper) since mid-2026.
    if yt_info.get("available"):
        has_ejs = yt_probe.get("yt_dlp_ejs", False)
        runtimes = yt_probe.get("js_runtimes", "none")
        runtime_active = runtimes and runtimes != "none"
        if has_ejs and runtime_active:
            tools.append(f"  yt_ejs     : ok  (yt_dlp_ejs + JS runtime: {runtimes}) — YT audio downloads should work")
        elif has_ejs and not runtime_active:
            tools.append(f"  yt_ejs     : partial — yt_dlp_ejs is installed but no JS runtime detected (needed: node/deno/bun/quickjs). Install Node: `brew install node`")
        elif not has_ejs and runtime_active:
            tools.append(f"  yt_ejs     : partial — JS runtime {runtimes} is available but yt_dlp_ejs is missing. Install: `pip install yt-dlp-ejs` (into yt-dlp's Python env, not this one)")
        else:
            tools.append("  yt_ejs     : missing  (need BOTH `pip install yt-dlp-ejs` and a JS runtime like `brew install node`). YT audio downloads will fail with 'Requested format is not available'.")

    # End-to-end n-challenge probe. The libs check above can be "ok"
    # while YouTube actually rejects the solver because they shipped a
    # new challenge variant this week. Only this probe catches that.
    n_probe = rep.get("yt_dlp_n_challenge", {})
    if yt_info.get("available") and n_probe:
        status = n_probe.get("status")
        count = n_probe.get("format_count", 0)
        if status == "ok":
            tools.append(f"  yt_n_chal  : ok  (end-to-end probe extracted {count} formats from test video — YouTube downloads working today)")
        elif status == "broken":
            err = n_probe.get("error") or "no formats returned"
            tools.append(f"  yt_n_chal  : BROKEN ({err}) — even with libs installed, YouTube isn't serving playable formats. Try `pip install -U 'yt-dlp[default]' yt-dlp-ejs` and check yt-dlp/wiki/EJS")
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

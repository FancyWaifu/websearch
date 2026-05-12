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
        "pdftotext": {
            "available": pdfx.have_pdftotext(),
            "path": shutil.which("pdftotext"),
        },
        "yt_dlp": {
            "available": shutil.which("yt-dlp") is not None,
            "path": shutil.which("yt-dlp"),
        },
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

    tools = []
    for k in ("pdftotext", "yt_dlp"):
        info = rep.get(k, {})
        mark = "ok" if info.get("available") else "missing"
        tools.append(f"  {k:10s} : {mark}  ({info.get('path') or '-'})")
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

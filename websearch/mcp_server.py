"""MCP (Model Context Protocol) stdio server for websearch.

Exposes `search`, `fetch`, and `research` as MCP tools so an MCP client
(Claude Code, etc.) can call them natively — no Bash shell-out, no PATH-shim
gotcha, structured results instead of scraped stdout.

Run:       websearch mcp
Register:  claude mcp add websearch -- ~/.local/bin/websearch mcp

The `mcp` package is an optional dependency. Install it into the websearch
environment with:  pipx inject websearch mcp
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

from .cache import default as default_cache
from .core import (
    fetch_many,
    fetch_smart,
    html_to_text,
    search_many,
    search_smart,
)
from . import rerank as _rerank


_MISSING_MCP_HINT = (
    "the 'mcp' package is required for `websearch mcp`.\n"
    "Install it into the websearch environment with:\n"
    "    pipx inject websearch mcp\n"
)

_BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def _ip_is_public(addr: str) -> bool:
    """True iff `addr` parses to an IP outside the private/loopback/link-local/
    multicast/reserved/unspecified ranges. Catches IPv4 and IPv6 alike."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _url_safety_error(url: str) -> Optional[str]:
    """Return a human-readable reason the URL is unsafe to fetch from the MCP
    server, or None if it's a public http(s) URL.

    Blocks: non-http(s) schemes (file://, gopher://, ...), bare or
    DNS-resolved private/loopback/link-local/multicast IPs (cloud metadata
    169.254.169.254, LAN hosts 192.168.x.x, localhost, ...), and explicit
    localhost-style hostnames. This is the SSRF gate for an MCP-exposed
    fetch — an MCP client (or prompt injection through search results) must
    not be able to reach internal services through this process."""
    try:
        p = urlparse(url)
    except Exception as e:  # noqa: BLE001
        return f"unparseable URL: {e}"
    if p.scheme not in ("http", "https"):
        return f"only http(s) URLs are allowed (got scheme: {p.scheme or 'none'})"
    host = (p.hostname or "").lower()
    if not host:
        return "URL has no host"
    if host in _BLOCKED_HOSTS:
        return f"host {host!r} is blocked"
    # Resolve to all addresses (handles IPv6 + multi-A records). Any
    # non-public answer fails the check — DNS rebinding is out of scope
    # for a single-shot validate-then-fetch, but a single private address
    # is enough to refuse.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"DNS resolution failed for {host!r}: {e}"
    for info in infos:
        if not _ip_is_public(info[4][0]):
            return f"host {host!r} resolves to a non-public address ({info[4][0]})"
    return None


def build_server():
    """Construct the FastMCP server with the websearch tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(_MISSING_MCP_HINT) from e

    mcp = FastMCP("websearch")

    @mcp.tool()
    def search(query: str, max_results: int = 10, trust: str = "medium") -> str:
        """Search the web and return ranked results (title, URL, snippet).

        trust: 'any' (no filter), 'medium' (drop SEO/affiliate spam),
        'high' (trusted sources only). Tries SearXNG (if $WEBSEARCH_SEARXNG
        is set), then DuckDuckGo, then Bing.
        """
        try:
            engine, results = search_smart(
                query, max_results=max_results, trust=trust
            )
        except Exception as e:  # noqa: BLE001 - surface to the client cleanly
            return f"search failed: {e}"
        if not results:
            return f"No results for: {query}"
        lines = [f"# {len(results)} results for {query!r}  (engine: {engine})"]
        for r in results:
            lines.append(f"\n[{r.rank}] {r.title}\n    {r.url}")
            if r.snippet:
                lines.append(f"    {r.snippet}")
        return "\n".join(lines)

    @mcp.tool()
    def fetch(url: str, max_chars: int = 4000) -> str:
        """Fetch a URL and return its readable text (Wayback fallback on
        failure, transcript extraction for YouTube). Truncated to max_chars
        at a clean boundary."""
        unsafe = _url_safety_error(url)
        if unsafe:
            return f"refused to fetch {url}: {unsafe}"
        try:
            res = fetch_smart(url, cache=default_cache())
        except Exception as e:  # noqa: BLE001
            return f"fetch failed: {e}"
        if res.error:
            return f"ERROR fetching {url}: {res.error}"
        body = html_to_text(res.text, max_chars=max_chars)
        tag = " [PDF]" if res.is_pdf else ""
        return f"# {url}\n(status {res.status}, via {res.via}{tag})\n\n{body}"

    @mcp.tool()
    def research(
        question: str,
        related_queries: Optional[list[str]] = None,
        depth: int = 5,
        trust: str = "medium",
        max_chars_per_source: int = 2500,
    ) -> str:
        """Multi-query research: search several angles of a question, dedupe
        and rerank the pool, fetch the top `depth` sources, and return one
        consolidated markdown document with the source content inline."""
        queries = [question] + [q for q in (related_queries or []) if q.strip()]
        try:
            report = search_many(queries, dedupe=True, trust=trust)
        except Exception as e:  # noqa: BLE001
            return f"research failed: {e}"

        merged = report.get("unique", []) or []
        if len(report.get("queries", {})) > 1:
            merged = _rerank.boost_by_query_count(report["queries"], merged)
        merged = _rerank.rerank(merged, question)
        if not merged:
            return f"No results for: {question}"

        # Defense-in-depth: drop poisoned/internal URLs before fetching, so a
        # search result pointing at e.g. 169.254.169.254 can't reach internal
        # services through this process.
        merged = [r for r in merged if _url_safety_error(r.get("url", "")) is None]
        top = merged[: max(1, depth)]
        out = [
            f"# Research: {question}",
            f"_{len(merged)} unique results across {len(queries)} queries; "
            f"showing top {len(top)}_\n",
        ]
        fetched = fetch_many([r["url"] for r in top], cache=default_cache())
        by_url = {fr.url: fr for fr in fetched}
        for r in top:
            out.append(f"## {r.get('title') or r['url']}\n{r['url']}")
            fr = by_url.get(r["url"])
            if fr is None or fr.error:
                err = fr.error if fr else "not fetched"
                out.append(f"_(could not fetch: {err})_\n")
                continue
            out.append("\n" + html_to_text(fr.text, max_chars=max_chars_per_source) + "\n")
        return "\n".join(out)

    return mcp


def run() -> None:
    """Build and run the MCP server over stdio. Blocks until the client
    disconnects. Raises ImportError (with an install hint) if `mcp` is
    not installed."""
    build_server().run()

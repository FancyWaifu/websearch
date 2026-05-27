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

from typing import Optional

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

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
from . import openalex as _openalex


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
        """Search the web and return a ranked list of (title, URL, snippet).

        Use this for: discovering URLs to read next, finding multiple
        perspectives on a question, or surveying what's been written on a
        topic. Returns the engine name used + up to `max_results` results.
        Does NOT fetch page content — pair with `fetch` when you need the
        actual body.

        Args:
            query: the search query string. Phrases in double quotes are
                exact-match.
            max_results: cap on results returned (default 10, max ~20 in
                practice based on engine behavior).
            trust: filter strictness. 'any' = no filter, 'medium' (default)
                drops SEO/affiliate spam via the reputation blocklist,
                'high' restricts to .gov/.edu/journals/major news.

        Returns:
            Markdown text. Header line names the engine ("engine: searxng"
            or "engine: duckduckgo" etc.) so the caller can see which
            backend served the results.
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
        """Fetch one URL and return its readable text content.

        Use this when you have a URL (from `search`, the user, or your
        own reasoning) and need the actual page content. Handles three
        special cases automatically:
            - PDF URLs -> extracts text via pdftotext (no manual flag)
            - YouTube URLs -> fetches the captions/transcript via the
              youtube-transcript-api (no video download)
            - Failed direct fetch -> falls back to the Wayback Machine

        Args:
            url: an http(s) URL. Internal/private addresses (169.254/16,
                10/8, 192.168/16, localhost, file://, etc.) are refused
                for SSRF safety.
            max_chars: truncate the body at this many chars. Truncation
                is at a paragraph/sentence boundary, never mid-word.

        Returns:
            Markdown with a header line ("status N, via direct|wayback|
            cache") followed by the cleaned text. On failure returns
            "ERROR fetching <url>: <reason>" — check for that prefix.
        """
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
        """Multi-query web research: search → dedupe → rerank → fetch → bundle.

        Use this when you'd otherwise call `search` + several `fetch`es
        for the same question. Saves round-trips and produces one
        consolidated markdown document with the source bodies inlined,
        ready to cite or summarize.

        Args:
            question: the primary research question (drives ranking).
            related_queries: extra angles to search in parallel; results
                are deduped against the primary query's pool. Pass 1-3
                related framings for best coverage.
            depth: how many top sources to actually fetch (default 5).
                More = better recall, slower, more tokens in output.
            trust: same as `search` — 'any' / 'medium' / 'high'.
            max_chars_per_source: per-fetched-source truncation (default
                2500). Reduce when you'll combine with several other tool
                calls in the same context window.

        Returns:
            Markdown document with a results list, then `## <title>` +
            URL + extracted body for each fetched source. Failed fetches
            appear as `_(could not fetch: <reason>)_` so the caller can
            see what was attempted.
        """
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

    @mcp.tool()
    def cite(urls: list[str]) -> str:
        """Generate a markdown citation block for one or more URLs.

        Use this when you've referenced URLs in a response and want a
        clean `## Sources` block at the bottom. Fetches each URL just
        enough to extract its `<title>`; falls through to the URL itself
        if the title can't be read.

        Args:
            urls: list of http(s) URLs to cite. Order is preserved.

        Returns:
            Markdown:
                Sources:
                - [Page Title](https://...)
                - [Other Page](https://...)
            One entry per input URL, even if the fetch failed.
        """
        if not urls:
            return "Sources:\n(none)"
        clean: list[str] = []
        for u in urls:
            if _url_safety_error(u):
                clean.append(f"- {u}  _(refused: internal URL)_")
                continue
            try:
                r = fetch_smart(u, cache=default_cache())
            except Exception as e:  # noqa: BLE001
                clean.append(f"- [{u}]({u})  _(fetch failed: {e})_")
                continue
            title = ""
            if r.text:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(r.text, "lxml")
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                except Exception:
                    pass
            clean.append(f"- [{title or u}]({u})")
        return "Sources:\n" + "\n".join(clean)

    @mcp.tool()
    def execute(code: str, timeout: int = 60) -> str:
        """Execute a Python snippet that calls websearch tools directly.

        Anthropic's "code execution with MCP" pattern: instead of the
        model round-tripping intermediate tool results through its
        context (search -> N URLs -> N fetches -> filter -> summarize),
        let the model write one script that chains the calls. Only the
        final value or printed output returns to the model context,
        which can be 10-100x cheaper for multi-step workflows.

        Available functions in the script's namespace:
            search(query, max_results=10, trust='medium') -> list[dict]
            fetch(url, max_chars=4000) -> dict {'url','status','body','error'}
            research(question, related=None, depth=5) -> str (markdown)
            papers(query, year_min=None, ...) -> str
            json: the json module, pre-imported for convenience.

        Return value protocol: assign to `result` (any type — will be
        str()'d) OR print to stdout. If both, `result` wins.

        Trust model: code runs in the websearch process's Python
        interpreter with full user permissions. The MCP transport is
        the trust boundary. Don't expose this server over an untrusted
        network or to agents you wouldn't give shell access to.

        Args:
            code: Python source. Multi-line OK. Tabs/spaces both fine.
            timeout: max seconds to run (default 60).

        Returns:
            String. On error: "ERROR: <type>: <message>\\n<traceback>\\n
            ---stdout---\\n<captured>". On success: the str(result) if
            `result` was set, else captured stdout, else
            "(no output)".
        """
        import contextlib
        import io
        import json as _json
        import signal
        import traceback

        # Build the namespace the agent's script will see. Each tool
        # function returns a dict/list/str — JSON-friendly types so
        # the agent can json.dumps the final result if it wants.
        def _search(query, max_results=10, trust="medium"):
            engine, results = search_smart(query, max_results=max_results, trust=trust)
            return [
                {"rank": r.rank, "title": r.title, "url": r.url, "snippet": r.snippet}
                for r in results
            ]

        def _fetch(url, max_chars=4000):
            unsafe = _url_safety_error(url)
            if unsafe:
                return {"url": url, "status": 0, "body": "",
                        "error": f"refused: {unsafe}"}
            res = fetch_smart(url, cache=default_cache())
            body = html_to_text(res.text, max_chars=max_chars) if res.text else ""
            return {
                "url": url,
                "final_url": res.final_url,
                "status": res.status,
                "via": res.via,
                "body": body,
                "error": res.error,
            }

        def _research(question, related=None, depth=5):
            return research(question, related_queries=related, depth=depth)

        def _papers(query, max_results=5, year_min=None, year_max=None,
                    min_citations=None, oa_only=False):
            return papers(query, max_results=max_results, year_min=year_min,
                          year_max=year_max, min_citations=min_citations,
                          oa_only=oa_only)

        namespace: dict = {
            "search": _search,
            "fetch": _fetch,
            "research": _research,
            "papers": _papers,
            "json": _json,
            "__builtins__": __builtins__,
        }

        # SIGALRM-based timeout. Only works on POSIX; on Windows fall
        # through to no timeout (still better than nothing). The handler
        # raises TimeoutError which the exec catches like any other.
        prev_handler = None
        if hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"execute() exceeded {timeout}s budget")
            prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, namespace)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            return (f"ERROR: {type(e).__name__}: {e}\n{tb}"
                    f"\n---stdout---\n{buf.getvalue()}")
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)

        if "result" in namespace:
            return str(namespace["result"])
        out = buf.getvalue()
        return out if out else "(no output and no `result` variable set)"

    @mcp.tool()
    def papers(
        query: str,
        max_results: int = 5,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: Optional[int] = None,
        oa_only: bool = False,
    ) -> str:
        """Search OpenAlex (250M+ academic works) for scholarly papers.

        Use this INSTEAD of `search` when you specifically want
        peer-reviewed work: title, authors, year, journal, citation count,
        DOI, OA URL, and abstract. Far higher signal than scraping Google
        Scholar — OpenAlex has structured metadata you'd otherwise have to
        regex out of HTML.

        Args:
            query: the research topic (free text, not boolean).
            max_results: 5-25 typical; OpenAlex returns ranked by
                relevance.
            year_min, year_max: bound the publication year window.
            min_citations: drop papers with fewer than N citations
                (good for filtering to established work).
            oa_only: when True, restrict to open-access papers that have
                a downloadable PDF.

        Returns:
            Markdown listing per paper: title, authors, year, journal,
            citation count, DOI link, OA PDF link if available, and the
            abstract. Header reports the total match count so you know
            the search wasn't suspiciously narrow.
        """
        try:
            articles, _total = _openalex.fetch_articles(
                query,
                max_results=max_results,
                year_min=year_min,
                year_max=year_max,
                min_citations=min_citations,
                oa_only=oa_only,
                cache=default_cache(),
            )
        except Exception as e:  # noqa: BLE001
            return f"papers search failed: {e}"
        if not articles:
            return f"No papers found for: {query}"
        return _openalex.to_markdown(query, articles)

    return mcp


def run() -> None:
    """Build and run the MCP server over stdio. Blocks until the client
    disconnects. Raises ImportError (with an install hint) if `mcp` is
    not installed."""
    build_server().run()

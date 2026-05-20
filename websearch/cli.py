"""Command-line interface for websearch."""
from __future__ import annotations

import argparse
import json
import sys
import time

try:
    from . import __version__
    from .cache import default as default_cache
    from .core import (
        download,
        extract_title,
        fetch_direct,
        fetch_many,
        fetch_smart,
        fetch_wayback,
        grep_lines,
        html_to_text,
        search_bing,
        search_duckduckgo,
        search_many,
        search_smart,
        selftest,
    )
    from . import reputation
    from . import rerank as _rerank
    from . import doctor as _doctor
    from . import dates as _dates
    from . import pdfx as _pdfx
    from . import usernames as _usernames
    from . import gh as _gh
except ImportError as _imp_err:
    sys.stderr.write(
        "websearch: failed to import its own package.\n"
        f"  Active binary: {sys.argv[0] if sys.argv else '?'}\n"
        f"  Python:        {sys.executable}\n"
        f"  Error:         {_imp_err}\n"
        "\n"
        "This usually means a stale console-script shim is on PATH from an\n"
        "older Python install. Reinstall with:\n"
        "    cd ~/websearch && pipx install -e . --force\n"
        "Then invoke `~/.local/bin/websearch` directly to bypass any stale shim.\n"
    )
    sys.exit(2)


# ----------------------------- helpers --------------------------------------


def _add_proxy_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--proxy",
        default=None,
        help="proxy URL (e.g., socks5h://127.0.0.1:1080). "
        "Falls back to $WEBSEARCH_PROXY env var.",
    )


def _add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass disk cache for this call (do not read or write)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cache on read but still write the fresh response",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="max acceptable cache age in seconds (None = any age)",
    )


def _resolve_cache(args: argparse.Namespace):
    return None if getattr(args, "no_cache", False) else default_cache()


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """Shared search-quality flags: date / trust / preference."""
    p.add_argument(
        "--since",
        default=None,
        metavar="WHEN",
        help="restrict to results after this date. Accepts YYYY-MM-DD, "
        "YYYY, or shorthand d/w/m/y (past day/week/month/year). "
        "Honored by DuckDuckGo; Bing ignores it.",
    )
    p.add_argument(
        "--trust",
        choices=["any", "medium", "high"],
        default="any",
        help="reputation filter: 'medium' drops SEO/affiliate farms, "
        "'high' keeps only trusted sources (.gov, .edu, journals, major "
        "news). Default: any.",
    )
    p.add_argument(
        "--prefer",
        choices=list(reputation.TRUSTED.keys()),
        default=None,
        help="boost results from this source category and reorder by "
        "reputation score (academic, gov, news, reference).",
    )


def _add_body_args(p: argparse.ArgumentParser) -> None:
    """Shared body-extraction flags used by fetch + search --fetch-top."""
    p.add_argument(
        "--grep",
        default=None,
        metavar="PATTERN",
        help="after extraction, return only lines matching this regex "
        "(case-insensitive) with surrounding context. Applied before --max-chars.",
    )
    p.add_argument(
        "--grep-context",
        type=int,
        default=2,
        help="lines of context around each --grep match (default: 2)",
    )
    p.add_argument(
        "--section",
        default=None,
        metavar="NAMES",
        help="comma-separated section names to keep from fetched bodies "
        "(e.g. 'abstract,conclusion'). Honors aliases for common headings.",
    )


def _add_rerank_args(p: argparse.ArgumentParser) -> None:
    """Shared search-quality flags: rerank + keyword require."""
    p.add_argument(
        "--rerank",
        action="store_true",
        help="after dedupe, rerank results by TF-IDF similarity to the "
        "primary query. Cheap, no model dependency.",
    )
    p.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="TERMS",
        help="drop results whose title+snippet contains none of these "
        "comma-separated terms (case-insensitive). Repeatable.",
    )


def _enforce_since(fetched: list, since: str) -> list:
    """Drop fetched results whose extracted published date is before `since`.

    Results without an extractable date are KEPT (we don't have grounds to
    drop them, and dropping silently would be worse than leaving them in).
    """
    cutoff = _dates.parse_since(since)
    if not cutoff:
        return fetched
    out = []
    for r in fetched:
        body = r.text if hasattr(r, "text") else r.get("body") or r.get("text") or ""
        # Only meaningful on HTML — PDFs and transcripts don't carry these meta tags
        if not body or body.lstrip().startswith("%PDF"):
            out.append(r)
            continue
        pub = _dates.extract_published(body)
        if pub is None or pub >= cutoff:
            # Annotate for the renderer
            if hasattr(r, "__dict__"):
                setattr(r, "published_date", pub.isoformat() if pub else None)
            elif isinstance(r, dict):
                r["published_date"] = pub.isoformat() if pub else None
            out.append(r)
    return out


def _postprocess_body(
    body: str,
    content_type: str,
    *,
    raw: bool,
    max_chars,
    skip_chars: int,
    mode: str,
    grep: str | None = None,
    grep_context: int = 2,
    section: str | None = None,
) -> str:
    """Apply extraction → section-filter → grep → truncation to a fetched body."""
    looks_html = "html" in (content_type or "").lower() or body.lstrip().startswith("<")
    if not raw and looks_html:
        body = html_to_text(body, max_chars=None, skip_chars=skip_chars, mode=mode)
    if section:
        sections = [s.strip() for s in section.split(",") if s.strip()]
        if sections:
            body = _pdfx.extract_sections(body, sections)
    if grep:
        filtered = grep_lines(body, grep, context=grep_context)
        body = filtered if filtered else f"[no lines matched /{grep}/]"
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"
    return body


def _print_results_text(engine: str, results: list) -> None:
    print(f"# {len(results)} results from {engine}\n")
    for r in results:
        print(f"[{r.rank}] {r.title}")
        print(f"    {r.url}")
        if r.snippet:
            print(f"    {r.snippet}")
        print()


def _render_compact_report(report: dict, *, header: str | None = None) -> str:
    """Compact markdown renderer for single-query, multi-query, or research reports.

    Shared by --compact on `search` and by the `research` subcommand.
    """
    parts: list[str] = []
    if header:
        parts.append(f"# {header}")

    # Single-query (cmd_search with --fetch-top, compact=True)
    if "results" in report and "engine" in report:
        q = report.get("query", "")
        parts.append(f"# Search: {q}  ({report['engine']})")
        for r in report["results"]:
            host = reputation.domain_of(r["url"])
            parts.append(f"\n[{r['rank']}] {r['title']}  — {host}")
            parts.append(f"    {r['url']}")
            if r.get("snippet"):
                parts.append(f"    {r['snippet']}")

    # Multi-query / research
    if "unique" in report:
        nq = len(report.get("queries", {}))
        parts.append(f"# Search: {nq} queries → {len(report['unique'])} unique results")
        for r in report["unique"]:
            host = reputation.domain_of(r["url"])
            fq = f"  (from: {r.get('from_query','')})" if r.get("from_query") else ""
            parts.append(f"\n[{r['rank']}] {r['title']}  — {host}{fq}")
            parts.append(f"    {r['url']}")
            if r.get("snippet"):
                parts.append(f"    {r['snippet']}")

    # Fetched content block
    if report.get("fetched"):
        parts.append("\n" + "=" * 70)
        parts.append("# Fetched content")
        parts.append("=" * 70)
        for fr in report["fetched"]:
            host = reputation.domain_of(fr.get("url", ""))
            parts.append(f"\n## {fr.get('url')}  ({host})")
            meta = (
                f"via={fr.get('via')} status={fr.get('status')}"
                f" cache={'yes' if fr.get('from_cache') else 'no'}"
            )
            if fr.get("is_pdf"):
                meta += " [PDF]"
            if fr.get("tried_wayback"):
                meta += f" wayback_status={fr.get('wayback_status')}"
                if fr.get("wayback_error"):
                    meta += f" wayback_error={fr.get('wayback_error')}"
            parts.append(f"_{meta}_")
            if fr.get("error"):
                parts.append(f"ERROR: {fr['error']}")
                continue
            body = fr.get("body", "")
            if body:
                parts.append("")
                parts.append(body)
            parts.append("\n" + "-" * 60)
    return "\n".join(parts)


def _print_fetch_text(
    result,
    raw: bool,
    max_chars,
    skip_chars: int = 0,
    mode: str = "article",
    grep: str | None = None,
    grep_context: int = 2,
):
    print(f"# {result.url}", file=sys.stderr)
    cache_note = ""
    if result.from_cache:
        cache_note = f" (cached, age={int(result.cache_age_seconds or 0)}s)"
    pdf_note = " [PDF]" if result.is_pdf else ""
    print(
        f"# via={result.via}{cache_note}{pdf_note} status={result.status} type={result.content_type}",
        file=sys.stderr,
    )
    if result.tried_wayback:
        wb_part = (
            f"wayback_status={result.wayback_status}"
            + (f" wayback_error={result.wayback_error}" if result.wayback_error else "")
        )
        print(f"# {wb_part}", file=sys.stderr)
    if result.error:
        print(f"# error: {result.error}", file=sys.stderr)
        return 1
    looks_html = "html" in (result.content_type or "").lower() or result.text.lstrip().startswith("<")
    if looks_html and not raw:
        title = extract_title(result.text)
        if title:
            print(f"# {title}\n")
    body = _postprocess_body(
        result.text,
        result.content_type,
        raw=raw,
        max_chars=max_chars,
        skip_chars=skip_chars,
        mode=mode,
        grep=grep,
        grep_context=grep_context,
    )
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")
    return 0


# ----------------------------- commands -------------------------------------


def _body_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        raw=args.raw,
        max_chars=args.max_chars,
        skip_chars=args.skip_chars,
        mode=args.mode,
        grep=getattr(args, "grep", None),
        grep_context=getattr(args, "grep_context", 2),
        section=getattr(args, "section", None),
    )


def _attach_fetched(container: dict, fetched: list, args: argparse.Namespace) -> None:
    container["fetched"] = []
    for fr in fetched:
        d = fr.to_dict()
        body = d.pop("text", "")
        d["body"] = _postprocess_body(body, fr.content_type, **_body_kwargs(args))
        container["fetched"].append(d)


def cmd_search(args: argparse.Namespace) -> int:
    proxy = args.proxy

    # Collect queries: positional + any -q flags
    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    if args.q:
        queries.extend(args.q)
    if not queries:
        print("error: provide a query (positional or -q)", file=sys.stderr)
        return 2

    # Parse comma-separated exclude domains
    exclude: list[str] = []
    for e in args.exclude or []:
        exclude.extend([x.strip() for x in e.split(",") if x.strip()])

    filter_kwargs = dict(
        since=args.since,
        trust=args.trust,
        prefer=args.prefer,
    )

    require_terms: list[str] = []
    for r in (getattr(args, "require", None) or []):
        require_terms.extend([t.strip() for t in r.split(",") if t.strip()])
    do_rerank = getattr(args, "rerank", False)

    # Multi-query path
    if len(queries) > 1:
        report = search_many(
            queries,
            max_results=args.max,
            proxy=proxy,
            parallel=args.parallel,
            dedupe=True,
            exclude=exclude or None,
            **filter_kwargs,
        )
        _apply_rerank_pipeline(report, queries[0], require_terms, do_rerank)
        if args.fetch_top and report.get("unique"):
            top_urls = [r["url"] for r in report["unique"][: args.fetch_top]]
            fetched = fetch_many(
                top_urls,
                timeout=args.timeout,
                proxy=proxy,
                parallel=args.parallel,
                cache=_resolve_cache(args),
                max_age=args.max_age,
                refresh=args.refresh,
            )
            _attach_fetched(report, fetched, args)
        if args.compact:
            print(_render_compact_report(report))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # Single-query path
    query = queries[0]
    excl = exclude or None
    try:
        if args.engine == "auto":
            engine, results = search_smart(
                query, max_results=args.max, proxy=proxy, exclude=excl, **filter_kwargs
            )
        elif args.engine == "ddg":
            engine, results = "duckduckgo", search_duckduckgo(
                query, max_results=args.max, proxy=proxy, exclude=excl, **filter_kwargs
            )
        elif args.engine == "bing":
            engine, results = "bing", search_bing(
                query, max_results=args.max, proxy=proxy, exclude=excl, **filter_kwargs
            )
        else:
            print(f"unknown engine: {args.engine}", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"search failed: {e}", file=sys.stderr)
        return 1

    # Optional rerank + require on the single-query result set.
    if require_terms or do_rerank:
        dicts = [r.to_dict() for r in results]
        if require_terms:
            dicts = _rerank.filter_required(dicts, require_terms)
        if do_rerank:
            dicts = _rerank.rerank(dicts, query)
        # Reflect back into SearchResult list to keep downstream code uniform
        rebuilt = []
        for d in dicts:
            from .core import SearchResult as _SR
            rebuilt.append(_SR(
                rank=d.get("rank", 0),
                title=d.get("title", ""),
                url=d.get("url", ""),
                snippet=d.get("snippet", ""),
            ))
        results = rebuilt

    if args.fetch_top and results:
        top_urls = [r.url for r in results[: args.fetch_top]]
        fetched = fetch_many(
            top_urls,
            timeout=args.timeout,
            proxy=proxy,
            parallel=args.parallel,
            cache=_resolve_cache(args),
            max_age=args.max_age,
            refresh=args.refresh,
        )
        out = {
            "engine": engine,
            "query": query,
            "results": [r.to_dict() for r in results],
        }
        _attach_fetched(out, fetched, args)
        if args.compact:
            print(_render_compact_report(out))
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if args.compact or (not args.json):
        _print_results_text(engine, results)
    else:
        print(
            json.dumps(
                {"engine": engine, "results": [r.to_dict() for r in results]}, indent=2
            )
        )
    return 0


def _apply_rerank_pipeline(
    report: dict, primary_query: str, require_terms: list[str], do_rerank: bool
) -> dict:
    """Cross-query boost → require-keyword filter → optional TF-IDF rerank.

    Mutates `report['unique']` and re-ranks 1..N.
    """
    merged = report.get("unique", [])
    if not merged:
        return report
    if len(report.get("queries", {})) > 1:
        merged = _rerank.boost_by_query_count(report["queries"], merged)
    if require_terms:
        merged = _rerank.filter_required(merged, require_terms)
    if do_rerank and primary_query:
        merged = _rerank.rerank(merged, primary_query)
    report["unique"] = merged
    return report


def _research_frontmatter(args: argparse.Namespace, queries: list[str], report: dict) -> str:
    """YAML frontmatter for piping research output into notes systems."""
    import datetime as _dt
    proxy_used = bool(args.proxy or __import__("os").environ.get("WEBSEARCH_PROXY"))
    unique = report.get("unique", []) or []
    fetched = report.get("fetched", []) or []
    fm = [
        "---",
        f"title: {json.dumps(queries[0])}",
        f"timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"trust: {args.trust}",
        f"depth: {args.depth}",
        f"proxy_used: {str(proxy_used).lower()}",
        f"sources_unique: {len(unique)}",
        f"sources_fetched: {len(fetched)}",
        "queries:",
    ]
    for q in queries:
        fm.append(f"  - {json.dumps(q)}")
    fm.append("---")
    return "\n".join(fm)


def cmd_research(args: argparse.Namespace) -> int:
    """Preset: multi-query search → fetch top N → compact markdown or JSON.

    Matches the real research workflow: ask a question, have the tool
    pull trusted sources, and hand back one clean document instead of
    a raw JSON dump.
    """
    proxy = args.proxy

    queries: list[str] = []
    if args.question:
        queries.append(args.question)
    if args.q:
        queries.extend(args.q)
    if not queries:
        print("error: provide a question (positional or -q)", file=sys.stderr)
        return 2

    exclude: list[str] = []
    for e in args.exclude or []:
        exclude.extend([x.strip() for x in e.split(",") if x.strip()])

    require_terms: list[str] = []
    for r in (args.require or []):
        require_terms.extend([t.strip() for t in r.split(",") if t.strip()])

    report = search_many(
        queries,
        max_results=args.max,
        proxy=proxy,
        parallel=args.parallel,
        dedupe=True,
        exclude=exclude or None,
        since=args.since,
        trust=args.trust,
        prefer=args.prefer,
    )

    _apply_rerank_pipeline(report, queries[0], require_terms, args.rerank)

    if report.get("unique"):
        top_urls = [r["url"] for r in report["unique"][: args.depth]]
        fetched = fetch_many(
            top_urls,
            timeout=args.timeout,
            proxy=proxy,
            parallel=args.parallel,
            cache=_resolve_cache(args),
            max_age=args.max_age,
            refresh=args.refresh,
        )
        if args.enforce_since and args.since:
            fetched = _enforce_since(fetched, args.since)
        _attach_fetched(report, fetched, args)

    if args.format == "json":
        out = {
            "queries": queries,
            "trust": args.trust,
            "depth": args.depth,
            "results": report.get("unique", []),
            "fetched": report.get("fetched", []),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    header = f"Research: {queries[0]}"
    if len(queries) > 1:
        header += f"  (+{len(queries)-1} related)"
    parts: list[str] = []
    if not args.no_frontmatter:
        parts.append(_research_frontmatter(args, queries, report))
        parts.append("")
    parts.append(_render_compact_report(report, header=header))
    print("\n".join(parts))
    return 0


def _apply_jq(text: str, expr: str) -> str:
    """Pipe text through system `jq -r EXPR`. Errors print to stderr; on
    failure we return the original text so the user can see what was wrong."""
    import shutil as _shutil
    import subprocess as _sp
    if not _shutil.which("jq"):
        print("warning: --jq specified but `jq` is not on PATH; ignoring", file=sys.stderr)
        return text
    try:
        r = _sp.run(["jq", "-r", expr], input=text, capture_output=True, text=True, timeout=20)
    except _sp.TimeoutExpired:
        print("warning: jq timed out after 20s; returning raw body", file=sys.stderr)
        return text
    if r.returncode != 0:
        print(f"warning: jq returned rc={r.returncode}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return text
    return r.stdout


def cmd_fetch(args: argparse.Namespace) -> int:
    proxy = args.proxy
    cache = _resolve_cache(args)
    bkwargs = _body_kwargs(args)
    verify = not getattr(args, "insecure", False)

    # Single URL
    if len(args.urls) == 1:
        url = args.urls[0]
        kwargs = dict(
            timeout=args.timeout,
            proxy=proxy,
            cache=cache,
            max_age=args.max_age,
            refresh=args.refresh,
            verify=verify,
        )
        if args.via == "direct":
            result = fetch_direct(url, **kwargs)
        elif args.via == "wayback":
            result = fetch_wayback(url, **kwargs)
        else:
            result = fetch_smart(url, **kwargs)

        if args.json:
            body = _postprocess_body(result.text, result.content_type, **bkwargs)
            if getattr(args, "jq", None):
                body = _apply_jq(body, args.jq)
            out = result.to_dict()
            out["body"] = body
            out.pop("text", None)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0 if result.error is None else 1

        # When --jq is set, treat the body as JSON and emit the filter output directly
        if getattr(args, "jq", None) and result.text:
            print(_apply_jq(result.text, args.jq), end="" if "\n" in _apply_jq(result.text, args.jq) else "\n")
            return 0 if result.error is None else 1

        return _print_fetch_text(
            result,
            args.raw,
            args.max_chars,
            args.skip_chars,
            args.mode,
            grep=bkwargs["grep"],
            grep_context=bkwargs["grep_context"],
        )

    # Multiple URLs
    results = fetch_many(
        args.urls,
        timeout=args.timeout,
        proxy=proxy,
        parallel=args.parallel,
        via=args.via,
        cache=cache,
        max_age=args.max_age,
        refresh=args.refresh,
        verify=verify,
    )

    if args.json:
        out = []
        for r in results:
            d = r.to_dict()
            body = d.pop("text", "")
            d["body"] = _postprocess_body(body, r.content_type, **bkwargs)
            out.append(d)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if all(r.error is None for r in results) else 1

    if args.compact:
        report = {"fetched": []}
        for r in results:
            d = r.to_dict()
            body = d.pop("text", "")
            d["body"] = _postprocess_body(body, r.content_type, **bkwargs)
            report["fetched"].append(d)
        print(_render_compact_report(report))
        return 0 if all(r.error is None for r in results) else 1

    rc = 0
    for i, r in enumerate(results):
        if i:
            print("\n" + "=" * 70 + "\n")
        rc |= _print_fetch_text(
            r,
            args.raw,
            args.max_chars,
            args.skip_chars,
            args.mode,
            grep=bkwargs["grep"],
            grep_context=bkwargs["grep_context"],
        )
    return rc


def cmd_text(args: argparse.Namespace) -> int:
    """Shortcut: fetch and emit clean text."""
    args.via = "smart"
    args.raw = False
    args.json = False
    args.parallel = 4
    args.urls = [args.url]
    args.compact = False
    return cmd_fetch(args)


def cmd_cite(args: argparse.Namespace) -> int:
    """Fetch URLs, extract titles, emit a markdown citation block."""
    proxy = args.proxy
    cache = _resolve_cache(args)
    results = fetch_many(
        args.urls,
        timeout=args.timeout,
        proxy=proxy,
        parallel=min(args.parallel, len(args.urls)),
        cache=cache,
    )
    lines = []
    for r in results:
        if r.error or not r.text:
            title = r.url
        else:
            title = extract_title(r.text) or r.url
        # Strip site suffix like " | Wikipedia" if --short
        if args.short and " | " in title:
            title = title.split(" | ", 1)[0]
        lines.append(f"- [{title}]({r.url})")
    print("Sources:")
    for line in lines:
        print(line)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    proxy = args.proxy
    out = args.output or args.url.rsplit("/", 1)[-1] or "download.bin"

    last_print = [0.0]

    def progress(written: int, total):
        if args.quiet:
            return
        now = time.time()
        # Throttle progress to twice per second
        if now - last_print[0] < 0.5 and (total is None or written < total):
            return
        last_print[0] = now
        if total:
            pct = 100 * written / total
            print(
                f"\r  {written/1024/1024:7.1f} / {total/1024/1024:7.1f} MB  {pct:5.1f}%",
                end="",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"\r  {written/1024/1024:7.1f} MB",
                end="",
                file=sys.stderr,
                flush=True,
            )

    t0 = time.time()
    status, written = download(
        args.url, out, timeout=args.timeout, proxy=proxy, progress=progress
    )
    if not args.quiet:
        print(file=sys.stderr)  # newline after progress
    elapsed = time.time() - t0

    if status >= 400 or written == 0:
        print(f"download failed: HTTP {status}, {written} bytes", file=sys.stderr)
        return 1

    print(
        f"saved {written:,} bytes to {out} in {elapsed:.1f}s ({written/1024/1024/max(elapsed,0.01):.1f} MB/s)",
        file=sys.stderr,
    )
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    cache = default_cache()
    if args.action == "stats":
        s = cache.stats()
        print(json.dumps(s, indent=2, default=str))
        return 0
    if args.action == "clear":
        n = cache.clear(older_than=args.older_than)
        print(f"deleted {n} entries", file=sys.stderr)
        return 0
    print(f"unknown cache action: {args.action}", file=sys.stderr)
    return 2


def cmd_selftest(args: argparse.Namespace) -> int:
    report = selftest(proxy=args.proxy)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Print install / PATH / proxy / tool-availability diagnosis."""
    rep = _doctor.report(proxy=args.proxy)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(_doctor.format_human(rep))
    # Exit non-zero if proxy is configured-but-unreachable, since that's
    # a silent-fallback foot-gun the user explicitly chose to set up.
    pr = rep.get("proxy", {})
    if pr.get("configured") and not pr.get("reachable", True):
        return 1
    return 0


def cmd_github(args: argparse.Namespace) -> int:
    """GitHub user reconnaissance — compact summaries from the public API."""
    cache = _resolve_cache(args)
    kw = dict(timeout=args.timeout, proxy=args.proxy, cache=cache, refresh=args.refresh)
    action = args.action
    user = args.username

    if action == "user":
        result = _gh.user_summary(user, **kw)
    elif action == "repos":
        result = _gh.repos(user, **kw)
    elif action == "emails":
        result = _gh.emails(user, parallel=args.parallel, **kw)
    elif action == "events":
        result = _gh.events(user, **kw)
    elif action == "gists":
        result = _gh.gists(user, **kw)
    elif action == "starred":
        result = _gh.starred(user, limit=args.limit, **kw)
    elif action == "orgs":
        result = _gh.orgs(user, **kw)
    elif action == "full":
        result = _gh.full_report(user, **kw)
    else:
        print(f"unknown action: {action}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_username(args: argparse.Namespace) -> int:
    """Probe a curated list of platforms for an account with `username`."""
    from typing import Optional
    sites: Optional[list[str]] = None
    if args.sites:
        sites = []
        for s in args.sites:
            sites.extend([t.strip() for t in s.split(",") if t.strip()])
    try:
        results = _usernames.enumerate_username(
            args.username,
            sites=sites,
            timeout=args.timeout,
            proxy=args.proxy,
            cache=_resolve_cache(args),
            parallel=args.parallel,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0

    # Human-readable: hits first, then misses
    hits = [r for r in results if r.found]
    misses = [r for r in results if not r.found]
    print(f"# username: {args.username}  ({len(hits)} hits / {len(results)} probed)")
    if hits:
        print("\nFOUND:")
        for r in hits:
            print(f"  {r.site:18s} {r.url}")
    if misses and args.verbose:
        print("\nNot found:")
        for r in misses:
            print(f"  {r.site:18s} {r.note:40s} {r.url}")
    elif misses:
        print(f"\n({len(misses)} sites had no account — use --verbose to see them)")
    return 0


def cmd_reputation(args: argparse.Namespace) -> int:
    if args.action == "explain":
        result = reputation.explain(args.url)
        print(json.dumps(result, indent=2))
        return 0
    if args.action == "list":
        data = reputation.list_category(args.category)
        print(json.dumps(data, indent=2))
        return 0
    print(f"unknown reputation action: {args.action}", file=sys.stderr)
    return 2


# ----------------------------- parser ---------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="websearch",
        description="Better web search, fetch, and download — usable by humans and AI assistants.",
    )
    p.add_argument("--version", action="version", version=f"websearch {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # search
    s = sub.add_parser("search", help="Search the web")
    s.add_argument("query", nargs="?", default=None, help="search query (or use -q)")
    s.add_argument(
        "-q",
        action="append",
        default=[],
        help="additional query (use multiple times for batched parallel searches with deduped results)",
    )
    s.add_argument("-n", "--max", type=int, default=10, help="max results per query (default: 10)")
    s.add_argument(
        "-e",
        "--engine",
        choices=["auto", "ddg", "bing"],
        default="auto",
        help="search engine (default: auto)",
    )
    s.add_argument("--json", action="store_true", help="emit JSON")
    s.add_argument(
        "--fetch-top",
        type=int,
        default=0,
        metavar="N",
        help="after searching, also fetch the top N URLs in parallel and include their bodies",
    )
    s.add_argument("--timeout", type=int, default=20, help="fetch timeout (when --fetch-top used)")
    s.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate fetched body to N chars (when --fetch-top used)",
    )
    s.add_argument(
        "--skip-chars",
        type=int,
        default=0,
        help="skip first N chars of fetched body (when --fetch-top used)",
    )
    s.add_argument(
        "--mode",
        choices=["article", "tables", "raw"],
        default="article",
        help="extraction mode for fetched bodies (default: article)",
    )
    s.add_argument(
        "--raw",
        action="store_true",
        help="emit raw HTML for fetched bodies",
    )
    s.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="parallel workers for batched search and --fetch-top (default: 4)",
    )
    s.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="exclude results from this domain (suffix match, can be repeated, comma-separated also works)",
    )
    s.add_argument(
        "--compact",
        "-c",
        action="store_true",
        help="emit clean markdown instead of JSON for multi-query or --fetch-top output",
    )
    _add_filter_args(s)
    _add_body_args(s)
    _add_rerank_args(s)
    _add_proxy_arg(s)
    _add_cache_args(s)
    s.set_defaults(func=cmd_search)

    # fetch
    f = sub.add_parser("fetch", help="Fetch one or more URLs (smart fallback to Wayback)")
    f.add_argument("urls", nargs="+", help="one or more URLs")
    f.add_argument(
        "--via",
        choices=["smart", "direct", "wayback"],
        default="smart",
        help="fetch method (default: smart)",
    )
    f.add_argument("--raw", action="store_true", help="emit raw HTML instead of extracted text")
    f.add_argument("--json", action="store_true", help="emit JSON envelope")
    f.add_argument("--timeout", type=int, default=20, help="request timeout in seconds")
    f.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate body to N chars (useful for AI consumption)",
    )
    f.add_argument(
        "--skip-chars",
        type=int,
        default=0,
        help="skip first N chars of extracted body (to bypass page chrome)",
    )
    f.add_argument(
        "--mode",
        choices=["article", "tables", "raw"],
        default="article",
        help="extraction mode (default: article). 'tables' = data tables only, 'raw' = full body text",
    )
    f.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="parallel workers when fetching multiple URLs (default: 4)",
    )
    f.add_argument(
        "--insecure",
        "-k",
        action="store_true",
        help="disable TLS certificate verification (use sparingly)",
    )
    f.add_argument(
        "--compact",
        "-c",
        action="store_true",
        help="for batch fetches, emit one clean markdown document instead of separator-divided text",
    )
    f.add_argument(
        "--jq",
        default=None,
        metavar="EXPR",
        help="pipe the fetched body through system `jq -r EXPR` "
        "(requires `jq` on PATH). Useful for trimming JSON API responses.",
    )
    _add_body_args(f)
    _add_proxy_arg(f)
    _add_cache_args(f)
    f.set_defaults(func=cmd_fetch)

    # text
    t = sub.add_parser("text", help="Shortcut: fetch + extract clean text")
    t.add_argument("url")
    t.add_argument("--timeout", type=int, default=20)
    t.add_argument("--max-chars", type=int, default=None)
    t.add_argument("--skip-chars", type=int, default=0)
    t.add_argument(
        "--mode",
        choices=["article", "tables", "raw"],
        default="article",
    )
    t.add_argument(
        "--insecure",
        "-k",
        action="store_true",
        help="disable TLS certificate verification (use sparingly)",
    )
    _add_body_args(t)
    _add_proxy_arg(t)
    _add_cache_args(t)
    t.set_defaults(func=cmd_text)

    # research — preset: multi-query → fetch top → compact markdown
    rs = sub.add_parser(
        "research",
        help="Research preset: multi-query search + fetch top + compact markdown",
    )
    rs.add_argument("question", nargs="?", default=None, help="main research question")
    rs.add_argument(
        "-q",
        action="append",
        default=[],
        help="related query (repeatable, runs in parallel and dedupes with the main question)",
    )
    rs.add_argument(
        "--depth",
        type=int,
        default=5,
        help="how many top URLs to fetch across the deduped result pool (default: 5)",
    )
    rs.add_argument("-n", "--max", type=int, default=10, help="max results per query (default: 10)")
    rs.add_argument("--timeout", type=int, default=20, help="fetch timeout seconds")
    rs.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="truncate each fetched body to N chars (default: 2500)",
    )
    rs.add_argument("--skip-chars", type=int, default=0)
    rs.add_argument(
        "--mode",
        choices=["article", "tables", "raw"],
        default="article",
    )
    rs.add_argument("--raw", action="store_true", help="emit raw HTML for fetched bodies")
    rs.add_argument("--parallel", type=int, default=4)
    rs.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="exclude results from this domain (repeatable/comma-separated)",
    )
    rs.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="output format: 'md' compact markdown (default) or 'json' structured envelope",
    )
    rs.add_argument(
        "--no-frontmatter",
        action="store_true",
        help="suppress YAML frontmatter on the markdown output",
    )
    rs.add_argument(
        "--enforce-since",
        action="store_true",
        help="after fetching, drop results whose extracted published date "
        "is before --since (requires --since). Pages without an extractable "
        "date are kept.",
    )
    _add_filter_args(rs)
    _add_body_args(rs)
    _add_rerank_args(rs)
    _add_proxy_arg(rs)
    _add_cache_args(rs)
    # Research defaults: favor clean sources out of the box
    rs.set_defaults(func=cmd_research, trust="medium")

    # cite
    ci = sub.add_parser("cite", help="Generate a markdown citation block from URLs")
    ci.add_argument("urls", nargs="+", help="one or more URLs to cite")
    ci.add_argument("--timeout", type=int, default=20)
    ci.add_argument("--parallel", type=int, default=4)
    ci.add_argument("--short", action="store_true", help="strip site suffixes from titles")
    _add_proxy_arg(ci)
    _add_cache_args(ci)
    ci.set_defaults(func=cmd_cite)

    # download
    d = sub.add_parser("download", help="Stream a URL to disk (binary-safe)")
    d.add_argument("url")
    d.add_argument("-o", "--output", default=None, help="output path (default: filename from URL)")
    d.add_argument("--timeout", type=int, default=120)
    d.add_argument("--quiet", action="store_true", help="suppress progress output")
    _add_proxy_arg(d)
    d.set_defaults(func=cmd_download)

    # cache
    c = sub.add_parser("cache", help="Manage the disk cache")
    c.add_argument("action", choices=["stats", "clear"])
    c.add_argument(
        "--older-than",
        type=int,
        default=None,
        help="for clear: only delete entries older than N seconds",
    )
    c.set_defaults(func=cmd_cache)

    # selftest
    st = sub.add_parser("selftest", help="Run a smoke test of search and fetch parsers")
    _add_proxy_arg(st)
    st.set_defaults(func=cmd_selftest)

    # doctor
    dr = sub.add_parser("doctor", help="Diagnose install / PATH / proxy / tool availability")
    dr.add_argument("--json", action="store_true", help="emit JSON instead of human-readable text")
    _add_proxy_arg(dr)
    dr.set_defaults(func=cmd_doctor)

    # github
    gh = sub.add_parser("github", help="GitHub user reconnaissance (user/repos/emails/events/full)")
    gh.add_argument(
        "action",
        choices=["user", "repos", "emails", "events", "gists", "starred", "orgs", "full"],
    )
    gh.add_argument("username")
    gh.add_argument("--timeout", type=int, default=15)
    gh.add_argument("--parallel", type=int, default=6, help="parallel workers for `emails` action")
    gh.add_argument("--limit", type=int, default=30, help="max items for `starred` action")
    _add_proxy_arg(gh)
    _add_cache_args(gh)
    gh.set_defaults(func=cmd_github)

    # username
    un = sub.add_parser("username", help="Probe common platforms for accounts matching a username")
    un.add_argument("username", help="username to look up")
    un.add_argument(
        "--sites",
        action="append",
        default=[],
        metavar="LIST",
        help="comma-separated subset of probe names (repeatable). Default: all.",
    )
    un.add_argument("--timeout", type=int, default=15)
    un.add_argument("--parallel", type=int, default=8)
    un.add_argument("--json", action="store_true", help="emit JSON instead of human-readable output")
    un.add_argument("-v", "--verbose", action="store_true", help="also list sites where the user wasn't found")
    _add_proxy_arg(un)
    _add_cache_args(un)
    un.set_defaults(func=cmd_username)

    # reputation
    rp = sub.add_parser("reputation", help="Inspect the reputation filter (explain a URL, list allowlists)")
    rp_sub = rp.add_subparsers(dest="action", required=True)
    rp_ex = rp_sub.add_parser("explain", help="explain why a URL is kept/dropped/boosted")
    rp_ex.add_argument("url")
    rp_ls = rp_sub.add_parser("list", help="list trusted allowlists")
    rp_ls.add_argument("--category", default=None, choices=list(reputation.TRUSTED.keys()))
    rp.set_defaults(func=cmd_reputation)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

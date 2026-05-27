"""Command-line interface for websearch."""
from __future__ import annotations

import argparse
import json
import os
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
        proxy_reachable,
        search_bing,
        search_duckduckgo,
        search_many,
        search_smart,
        selftest,
        smart_truncate,
    )
    from . import reputation
    from . import rerank as _rerank
    from . import doctor as _doctor
    from . import dates as _dates
    from . import pdfx as _pdfx
    from . import usernames as _usernames
    from . import gh as _gh
    from . import openalex as _openalex
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


def _add_searxng_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--searxng",
        default=None,
        metavar="URL",
        help="SearXNG instance URL to use as the primary search engine "
        "(tried before DuckDuckGo/Bing). Falls back to $WEBSEARCH_SEARXNG. "
        "The instance must have the 'json' format enabled.",
    )
    p.add_argument(
        "--searxng-autostart",
        default=None,
        metavar="COMPOSE_OR_NAME",
        help="if a local SearXNG is down, start it before querying. Value is "
        "a docker-compose file path or a container name. Falls back to "
        "$WEBSEARCH_SEARXNG_AUTOSTART.",
    )
    p.add_argument(
        "--brave-key",
        default=None,
        metavar="KEY",
        help="Brave Search API key. If set (or $WEBSEARCH_BRAVE_KEY), Brave "
        "is tried after SearXNG and before DDG/Bing scraping. Free tier "
        "available at https://api.search.brave.com/app/keys.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "searxng", "brave", "tavily", "exa", "duckduckgo", "bing"],
        default="auto",
        help="force a specific search backend instead of the auto-fallback "
        "chain. 'tavily' and 'exa' are AI-native paid APIs ($WEBSEARCH_"
        "TAVILY_KEY / $WEBSEARCH_EXA_KEY); auto (default) uses SearXNG -> "
        "Brave -> DDG -> Bing.",
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
        "--rerank-vector",
        action="store_true",
        help="rerank by sentence-transformer cosine similarity (stronger "
        "for paraphrased queries / synonyms). Needs the [embed] extra: "
        "`pipx inject websearch sentence-transformers`. Model from "
        "$WEBSEARCH_EMBED_MODEL (default 'all-MiniLM-L6-v2', ~90MB). "
        "First call loads model (~5s); subsequent calls fast. If the "
        "extra is missing, silently degrades to --rerank.",
    )
    p.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="TERMS",
        help="drop results whose title+snippet contains none of these "
        "comma-separated terms (case-insensitive). Repeatable.",
    )


def _warn_if_empty(report: dict, args: argparse.Namespace, queries: list[str]) -> None:
    """Emit a stderr WARNING when search returned zero usable results.

    Distinguishes the three common causes so the user knows where to look:
    a filter dropped everything, all engines were unresponsive, or the
    query genuinely matched nothing. Without this, `Search: N queries -> 0
    unique results` is the only signal — easy to misread as a real
    no-match when it's actually a filter or rate-limit issue.
    """
    if report.get("unique"):
        return
    per_query = report.get("queries") or {}
    n_raw = sum(len(v.get("results") or []) for v in per_query.values())
    engine_errors = [
        f"{q!r}: {v['error']}"
        for q, v in per_query.items()
        if v.get("error")
    ]
    msg = [f"WARNING: research returned 0 sources for {queries[0]!r}."]
    if getattr(args, "trust", "any") == "high":
        msg.append("  - --trust high accepts only .gov/.edu/journals/major "
                   "news; try --trust medium for a wider pool.")
    if engine_errors:
        msg.append("  - engine errors: " + "; ".join(engine_errors))
    if n_raw == 0:
        sx = getattr(args, "searxng", None) or os.environ.get("WEBSEARCH_SEARXNG")
        if not sx:
            msg.append("  - SearXNG not configured ($WEBSEARCH_SEARXNG); "
                       "DDG/Bing scraping is rate-limited and often empties.")
        msg.append("  - run `websearch doctor` to check engine reachability.")
    else:
        msg.append(f"  - {n_raw} raw result(s) were dropped by filters "
                   "(--trust / --exclude / --require / reputation block).")
    print("\n".join(msg), file=sys.stderr)


def _make_stream_progress(args: argparse.Namespace):
    """Build a progress callback that prints one stderr line per completed
    fetch, so `research` feels interactive instead of going dark for 10s.
    Suppressed when --format json (callers want a single structured
    output) or --no-stream is set."""
    if getattr(args, "format", None) == "json":
        return None
    start = time.time()

    def progress(idx: int, total: int, fr) -> None:
        elapsed = int((time.time() - start) * 1000)
        status = "ok" if (fr.error is None and (fr.text or "").strip()) else f"FAIL: {fr.error or 'empty'}"
        # Title is unknown at fetch time (it comes from the search snippet,
        # not the body), so just show URL + status.
        print(f"  [{idx + 1}/{total}] {fr.url}  [{status}, {elapsed}ms]",
              file=sys.stderr, flush=True)

    return progress


def _resolve_backend(args: argparse.Namespace) -> Optional[str]:
    """Translate the --backend argparse value into what search_smart wants.
    'auto' means 'use the fallback chain' which is the function's default
    (None), so we map that to None instead of the literal string 'auto'."""
    b = getattr(args, "backend", None)
    return None if b in (None, "auto") else b


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
            iso = pub.isoformat() if pub else None
            if isinstance(r, dict):
                r["published_date"] = iso
            else:
                r.published_date = iso
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
    return smart_truncate(body, max_chars)


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
            if fr.get("published_date"):
                meta += f" published={fr['published_date']}"
            if fr.get("is_pdf"):
                meta += " [PDF]"
            if not fr.get("error") and reputation.looks_affiliate(fr.get("body", "")):
                meta += " [affiliate-disclosure]"
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
    kw = _body_kwargs(args)
    # --max-total-chars: spread a whole-report budget across the fetched
    # sources so a research run can't blow past an output ceiling.
    mtc = getattr(args, "max_total_chars", None)
    if mtc and fetched:
        per = max(600, mtc // len(fetched))
        # max_chars=0 is an explicit "no per-source cap" and must not be
        # treated as unset; only None falls through to `per`.
        existing = kw.get("max_chars")
        if existing is None:
            kw["max_chars"] = per
        elif existing > 0:
            kw["max_chars"] = min(existing, per)
        # else: existing == 0, leave it as 0 (user opted out of per-source cap)
    for fr in fetched:
        d = fr.to_dict()
        body = d.pop("text", "")
        # Surface the page's published date inline (best-effort, from the
        # raw HTML before extraction strips the <meta>/JSON-LD tags).
        if not d.get("published_date"):
            looks_html = "html" in (fr.content_type or "").lower() or body.lstrip().startswith("<")
            pub = _dates.extract_published(body) if looks_html else None
            d["published_date"] = pub.isoformat() if pub else None
        d["body"] = _postprocess_body(body, fr.content_type, **kw)
        container["fetched"].append(d)


def _fetch_top_with_backfill(
    pool: list, depth: int, fetch_kwargs: dict
) -> tuple[list, int]:
    """Fetch `depth` usable bodies from the ranked `pool` of result dicts.

    When a top result fails to fetch (dead link, no Wayback snapshot, block
    page), the next-ranked candidate is pulled in to replace it — so the
    caller gets `depth` real sources whenever the pool runs deep enough,
    instead of burning a slot on an error. If the pool is exhausted before
    `depth` successes, the unrecoverable failures are kept in the returned
    list so the failure stays visible rather than silently hidden.

    Returns (fetched_results, n_backfilled).
    """
    urls = [r["url"] for r in pool]
    original_top = set(urls[:depth])
    good: list = []
    failed: list = []
    idx = 0
    while idx < len(urls) and len(good) < depth:
        need = depth - len(good)
        batch = urls[idx : idx + need]
        idx += len(batch)
        for fr in fetch_many(batch, **fetch_kwargs):
            if fr.error is None and (fr.text or "").strip():
                good.append(fr)
            else:
                failed.append(fr)
    fetched = good[:depth]
    if len(fetched) < depth:
        fetched += failed[: depth - len(fetched)]
    n_backfilled = sum(1 for fr in fetched if fr.url not in original_top)
    return fetched, n_backfilled


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
            searxng=args.searxng,
            searxng_autostart=args.searxng_autostart,
            brave_key=getattr(args, "brave_key", None),
            backend=_resolve_backend(args),
            **filter_kwargs,
        )
        _apply_rerank_pipeline(report, queries[0], require_terms, do_rerank,
                               rerank_vector=getattr(args, "rerank_vector", False))
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
                query, max_results=args.max, proxy=proxy, exclude=excl,
                searxng=args.searxng, searxng_autostart=args.searxng_autostart,
                brave_key=getattr(args, "brave_key", None),
                backend=_resolve_backend(args),
                **filter_kwargs,
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
    report: dict, primary_query: str, require_terms: list[str], do_rerank: bool,
    rerank_vector: bool = False,
) -> dict:
    """Cross-query boost → require-keyword filter → optional rerank.

    Mutates `report['unique']` and re-ranks 1..N. When `rerank_vector` is
    true, uses sentence-transformer cosine similarity (stronger, slower,
    needs the [embed] extra); otherwise falls back to TF-IDF when
    `do_rerank` is set.
    """
    merged = report.get("unique", [])
    if not merged:
        return report
    # Demote first: SERP-admitted partial matches (`Missing: <terms>`) shouldn't
    # outrank real matches no matter what else the pipeline does.
    merged = _rerank.demote_missing_terms(merged)
    if len(report.get("queries", {})) > 1:
        merged = _rerank.boost_by_query_count(report["queries"], merged)
    if require_terms:
        merged = _rerank.filter_required(merged, require_terms)
    if primary_query:
        if rerank_vector:
            if not _rerank.have_sentence_transformers():
                print("WARNING: --rerank-vector needs sentence-transformers; "
                      "falling back to --rerank (TF-IDF). Install with: "
                      "`pipx inject websearch sentence-transformers`",
                      file=sys.stderr)
                merged = _rerank.rerank(merged, primary_query)
            else:
                merged = _rerank.rerank_vector(merged, primary_query)
        elif do_rerank:
            merged = _rerank.rerank(merged, primary_query)
    report["unique"] = merged
    return report


def _research_frontmatter(args: argparse.Namespace, queries: list[str], report: dict) -> str:
    """YAML frontmatter for piping research output into notes systems."""
    import datetime as _dt
    proxy_conf = bool(args.proxy or os.environ.get("WEBSEARCH_PROXY"))
    reach = report.get("_proxy_reachable")
    proxy_used = proxy_conf and reach is True
    unique = report.get("unique", []) or []
    fetched = report.get("fetched", []) or []
    # ISO 8601 with the local UTC offset — bare `2026-05-27T13:04:14` was
    # ambiguous when the report was read days later in another timezone.
    ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    fm = [
        "---",
        f"title: {json.dumps(queries[0])}",
        f"timestamp: {ts}",
        f"trust: {args.trust}",
        f"depth: {args.depth}",
        f"proxy_used: {str(proxy_used).lower()}",
    ]
    if proxy_conf and reach is False:
        fm.append("proxy_note: configured but unreachable — used direct fetches")
    fm += [
        f"sources_unique: {len(unique)}",
        f"sources_fetched: {len(fetched)}",
        f"sources_backfilled: {report.get('_backfilled', 0)}",
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

    # #5: warn loudly when a proxy is configured but dead, instead of
    # silently falling back to direct (no stealth).
    reach = proxy_reachable(proxy)
    if reach is False:
        print(
            "WARNING: a proxy is configured (--proxy / $WEBSEARCH_PROXY) but "
            "unreachable — falling back to direct fetches with no stealth.",
            file=sys.stderr,
        )

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
        searxng=args.searxng,
        searxng_autostart=args.searxng_autostart,
        brave_key=getattr(args, "brave_key", None),
        backend=_resolve_backend(args),
    )

    _apply_rerank_pipeline(report, queries[0], require_terms, args.rerank,
                           rerank_vector=getattr(args, "rerank_vector", False))
    if args.max_per_domain and report.get("unique"):
        report["unique"] = reputation.cap_per_domain(
            report["unique"], args.max_per_domain
        )
    report["_proxy_reachable"] = reach
    report["_backfilled"] = 0
    _warn_if_empty(report, args, queries)

    if report.get("unique"):
        # #4: fetch the top `depth`, backfilling failed fetches from the
        # rest of the ranked pool so dead links don't waste a source slot.
        progress_cb = _make_stream_progress(args) if getattr(args, "stream", True) else None
        fetched, n_backfilled = _fetch_top_with_backfill(
            report["unique"],
            args.depth,
            dict(
                timeout=args.timeout,
                proxy=proxy,
                parallel=args.parallel,
                cache=_resolve_cache(args),
                max_age=args.max_age,
                refresh=args.refresh,
                use_whisper=getattr(args, "whisper", False),
                progress=progress_cb,
            ),
        )
        report["_backfilled"] = n_backfilled
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

    use_whisper = getattr(args, "whisper", False)

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
            result = fetch_direct(url, use_whisper=use_whisper, **kwargs)
        elif args.via == "wayback":
            result = fetch_wayback(url, **kwargs)
        else:
            result = fetch_smart(url, use_whisper=use_whisper, **kwargs)

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
            filtered = _apply_jq(result.text, args.jq)
            print(filtered, end="" if "\n" in filtered else "\n")
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
        use_whisper=use_whisper,
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


def cmd_papers(args: argparse.Namespace) -> int:
    """Search OpenAlex for academic papers."""
    from pathlib import Path

    mailto = _openalex.resolve_mailto(args.mailto)
    cache = _resolve_cache(args)

    if args.count_only:
        n = _openalex.preview_search(
            args.query,
            mailto=mailto,
            year_min=args.year_min, year_max=args.year_max,
            min_citations=args.min_citations,
            oa_only=args.oa_only, field=args.field,
            cache=cache, max_age=args.max_age, refresh=args.refresh,
            proxy=args.proxy,
        )
        print(json.dumps({"query": args.query, "count": n}))
        return 0

    def _progress(page: int, have: int, want: int) -> None:
        sys.stderr.write(f"\r  page {page}, {have}/{want} articles...")
        sys.stderr.flush()

    articles, total = _openalex.fetch_articles(
        args.query,
        max_results=args.max,
        mailto=mailto,
        year_min=args.year_min, year_max=args.year_max,
        min_citations=args.min_citations,
        oa_only=args.oa_only, field=args.field,
        sort_by=args.sort,
        cache=cache, max_age=args.max_age, refresh=args.refresh,
        proxy=args.proxy,
        progress=None if args.format == "json" else _progress,
    )
    if args.format != "json":
        sys.stderr.write("\r" + " " * 60 + "\r")

    if not articles:
        print(f"No articles found for: {args.query!r}", file=sys.stderr)
        return 1

    # Optional: download OA PDFs
    pdf_paths: dict[str, str] = {}
    if args.download_pdfs:
        out_dir = Path(args.pdf_dir).expanduser()
        for art in articles:
            p = _openalex.download_pdf(art, out_dir, proxy=args.proxy)
            if p:
                pdf_paths[art.openalex_id] = str(p)
        sys.stderr.write(f"# downloaded {len(pdf_paths)} PDFs to {out_dir}\n")

    # Optional: fetch related articles per result
    related_map: dict[str, list[dict]] = {}
    if args.related:
        for art in articles:
            related_map[art.openalex_id] = _openalex.fetch_related_articles(
                art.openalex_id, limit=args.related, mailto=mailto,
                cache=cache, max_age=args.max_age, refresh=args.refresh,
                proxy=args.proxy,
            )

    # Render
    fmt = args.format
    if fmt == "json":
        out = {
            "query": args.query,
            "total_match_count": total,
            "returned": len(articles),
            "articles": [a.to_dict() for a in articles],
        }
        if args.citation_graph:
            out["citation_network"] = _openalex.citation_network(articles)
        if related_map:
            out["related"] = related_map
        if pdf_paths:
            out["pdfs"] = pdf_paths
        print(json.dumps(out, indent=2, default=str))
    elif fmt == "md":
        print(_openalex.to_markdown(args.query, articles))
        if args.citation_graph:
            print("\n## Citation network (internal edges)\n")
            print(json.dumps(_openalex.citation_network(articles), indent=2))
    elif fmt == "bibtex":
        print(_openalex.to_bibtex(articles))
    elif fmt == "csv":
        sys.stdout.write(_openalex.to_csv(articles))
    elif fmt == "ris":
        print(_openalex.to_ris(articles))
    else:
        print(f"unknown format: {fmt}", file=sys.stderr)
        return 2

    sys.stderr.write(f"# matched {total} works, returned {len(articles)}\n")
    return 0


def cmd_reputation(args: argparse.Namespace) -> int:
    if args.action == "explain":
        result = reputation.explain(args.url)
        print(json.dumps(result, indent=2))
        return 0
    if args.action == "list":
        data = reputation.list_category(args.category)
        data["user_blocklist"] = sorted(reputation.user_blocklist())
        data["user_allowlist"] = sorted(reputation.user_allowlist())
        print(json.dumps(data, indent=2))
        return 0
    if args.action in ("block", "allow", "unblock", "unallow"):
        kind = "block" if args.action in ("block", "unblock") else "allow"
        remove = args.action.startswith("un")
        path = reputation.edit_user_list(kind, args.domain, remove=remove)
        verb = "removed from" if remove else "added to"
        print(f"{args.domain} {verb} user {kind}list → {path}")
        return 0
    print(f"unknown reputation action: {args.action}", file=sys.stderr)
    return 2


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run websearch as an MCP stdio server (search/fetch/research tools)."""
    from . import mcp_server
    try:
        mcp_server.run()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


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
    _add_searxng_arg(s)
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
        "--whisper",
        action="store_true",
        help="for YouTube URLs with no captions, download audio and transcribe "
        "locally via faster-whisper (needs `pipx inject websearch faster-whisper`). "
        "Model size: $WEBSEARCH_WHISPER_MODEL (default 'small').",
    )
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
    t.add_argument("--whisper", action="store_true", help="YT captionless videos via faster-whisper")
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
    rs.add_argument(
        "--max-per-domain",
        type=int,
        default=None,
        metavar="N",
        help="keep at most N results per domain before fetching, so one "
        "SEO/affiliate site can't monopolize the fetched sources",
    )
    rs.add_argument("--timeout", type=int, default=20, help="fetch timeout seconds")
    rs.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="truncate each fetched body to N chars (default: 2500)",
    )
    rs.add_argument(
        "--max-total-chars",
        type=int,
        default=None,
        metavar="N",
        help="cap total fetched-content size for the whole run; the budget "
        "is split evenly across fetched sources (per-source floor 600). "
        "Useful for keeping research output within an agent token budget.",
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
    _add_searxng_arg(rs)
    _add_cache_args(rs)
    rs.add_argument(
        "--whisper",
        action="store_true",
        help="for any YouTube source with no captions, transcribe audio "
        "locally via faster-whisper (opt-in, slow). See `fetch --whisper`.",
    )
    rs.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        default=True,
        help="suppress the per-source stderr progress lines that print as "
        "each fetch completes. On by default; auto-suppressed for "
        "--format json. Use when piping stderr matters.",
    )
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

    # papers — academic paper search via OpenAlex
    pp = sub.add_parser(
        "papers",
        help="Search OpenAlex for academic papers (with filters and export formats)",
    )
    pp.add_argument("query", help="search query (e.g., 'CRISPR gene editing')")
    pp.add_argument("-n", "--max", type=int, default=25,
                    help="max articles to return (default: 25)")
    pp.add_argument("--year-min", type=int, default=None, metavar="YYYY")
    pp.add_argument("--year-max", type=int, default=None, metavar="YYYY")
    pp.add_argument("--min-citations", type=int, default=None,
                    help="drop works with fewer than this many citations")
    pp.add_argument("--oa-only", action="store_true",
                    help="restrict to open-access works")
    pp.add_argument("--field", default=None, metavar="FIELD",
                    help="restrict by topic field (e.g., 'Medicine', 'Computer Science')")
    pp.add_argument("--sort", choices=["citations", "newest", "oldest"], default=None,
                    help="sort order (default: OpenAlex relevance)")
    pp.add_argument("--format", choices=["md", "json", "bibtex", "csv", "ris"],
                    default="md",
                    help="output format (default: md)")
    pp.add_argument("--count-only", action="store_true",
                    help="print only the total match count, do not fetch articles")
    pp.add_argument("--related", type=int, default=0, metavar="N",
                    help="also fetch N related works per article (json/md only)")
    pp.add_argument("--citation-graph", action="store_true",
                    help="include internal citation network (json/md only)")
    pp.add_argument("--download-pdfs", action="store_true",
                    help="download OA PDFs to --pdf-dir")
    pp.add_argument("--pdf-dir", default="./pdfs",
                    help="directory to save PDFs (default: ./pdfs)")
    pp.add_argument(
        "--mailto",
        default=None,
        metavar="EMAIL",
        help="contact email for the OpenAlex polite pool (higher rate "
        "limits). Falls back to $OPENALEX_API_KEY.",
    )
    _add_proxy_arg(pp)
    _add_cache_args(pp)
    pp.set_defaults(func=cmd_papers)

    # reputation
    rp = sub.add_parser("reputation", help="Inspect the reputation filter (explain a URL, list allowlists)")
    rp_sub = rp.add_subparsers(dest="action", required=True)
    rp_ex = rp_sub.add_parser("explain", help="explain why a URL is kept/dropped/boosted")
    rp_ex.add_argument("url")
    rp_ls = rp_sub.add_parser("list", help="list trusted allowlists + user lists")
    rp_ls.add_argument("--category", default=None, choices=list(reputation.TRUSTED.keys()))
    for _act, _help in (
        ("block", "add a domain to the persistent user blocklist"),
        ("allow", "add a domain to the persistent user allowlist"),
        ("unblock", "remove a domain from the user blocklist"),
        ("unallow", "remove a domain from the user allowlist"),
    ):
        _p = rp_sub.add_parser(_act, help=_help)
        _p.add_argument("domain", help="domain or URL (host is extracted)")
    rp.set_defaults(func=cmd_reputation)

    # mcp — run as an MCP stdio server
    mp = sub.add_parser(
        "mcp",
        help="Run as an MCP stdio server (exposes search/fetch/research as "
        "MCP tools). Register with: claude mcp add websearch -- websearch mcp",
    )
    mp.set_defaults(func=cmd_mcp)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Core fetch and search logic."""
from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional
from urllib.parse import quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .cache import Cache, default as default_cache
from . import pdfx
from . import reputation

try:
    import trafilatura  # type: ignore

    HAVE_TRAFILATURA = True
except Exception:
    HAVE_TRAFILATURA = False


# Rotating realistic user agents — many sites 403 on python-requests/* defaults.
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _headers(referer: Optional[str] = None) -> dict:
    h = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        # Drop 'br' — requires the brotli package and many Python installs lack it.
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h


def _proxies(proxy: Optional[str] = None) -> Optional[dict]:
    """Build a requests proxies dict.

    Resolution order: explicit arg > $WEBSEARCH_PROXY env > None.
    Use socks5h://host:port for DNS-through-proxy (recommended for stealth).
    """
    p = proxy or os.environ.get("WEBSEARCH_PROXY")
    if not p:
        return None
    return {"http": p, "https": p}


# A module-level Session reuses connections and cookies across calls
# within one process invocation. Cleared on process exit.
_session: Optional[requests.Session] = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


# ----------------------------- Result types --------------------------------


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str
    via: str  # "direct" | "wayback" | "cache"
    error: Optional[str] = None
    from_cache: bool = False
    cache_age_seconds: Optional[float] = None
    is_pdf: bool = False
    # Smart-fallback transparency: if direct failed and wayback was tried,
    # these fields record what happened on the wayback attempt.
    tried_wayback: bool = False
    wayback_status: Optional[int] = None
    wayback_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class SearchResult:
    rank: int
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EngineError:
    engine: str
    error: str


# ----------------------------- Fetching ------------------------------------


def _do_request(
    method: str,
    url: str,
    timeout: int,
    proxy: Optional[str],
    data: Optional[dict] = None,
    referer: Optional[str] = None,
    verify: bool = True,
) -> requests.Response:
    s = session()
    return s.request(
        method,
        url,
        headers=_headers(referer=referer),
        timeout=timeout,
        allow_redirects=True,
        proxies=_proxies(proxy),
        data=data,
        verify=verify,
    )


# Patterns that indicate the page is a captcha/block challenge rather
# than real search results. Used to give better errors than "selectors stale".
BLOCK_PATTERNS = [
    "captcha",
    "verify you are human",
    "unusual traffic",
    "automated queries",
    "access denied",
    "rate limit",
    "/sorry/",  # Google
    "challenge-platform",  # Cloudflare
]

# Markers that indicate the search engine returned a real "no results" page
# (as opposed to broken parsing or a block). When matched, an empty result
# list is returned cleanly instead of raising.
NO_RESULTS_PATTERNS = [
    "no results found",
    "no results.",
    "no results for",
    "did not match any",
    "we couldn't find anything",
    "your search returned no",
    "<div class=\"no-results\"",
    "results-no_results",
]


def _looks_blocked(html: str) -> Optional[str]:
    low = html.lower()
    for pat in BLOCK_PATTERNS:
        if pat in low:
            return pat
    return None


def _looks_empty(html: str) -> Optional[str]:
    low = html.lower()
    for pat in NO_RESULTS_PATTERNS:
        if pat in low:
            return pat
    return None


def fetch_direct(
    url: str,
    timeout: int = 20,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = None,
    refresh: bool = False,
    verify: bool = True,
) -> FetchResult:
    """Fetch a URL directly. Honors cache if provided and `refresh=False`.

    If the response is a PDF (by content-type, magic bytes, or .pdf URL),
    extracts text via pdftotext (if installed) and returns that as `text`.
    """
    cache = cache if cache is not None else default_cache()

    # Cache lookup
    if cache and not refresh:
        hit = cache.get("GET", url, max_age=max_age)
        if hit is not None:
            return FetchResult(
                url=hit.url,
                final_url=hit.final_url,
                status=hit.status,
                content_type=hit.content_type,
                text=hit.body,
                via="cache",
                error=None,
                from_cache=True,
                cache_age_seconds=hit.age_seconds,
                is_pdf="application/pdf" in (hit.content_type or "").lower(),
            )

    try:
        r = _do_request("GET", url, timeout, proxy, verify=verify)
    except requests.RequestException as e:
        return FetchResult(url, url, 0, "", "", "direct", str(e))

    content_type = r.headers.get("Content-Type", "")
    is_pdf = pdfx.looks_like_pdf(content_type, r.content[:8] if r.content else None, url)

    if r.status_code >= 400:
        text = ""
    elif is_pdf:
        # Extract text from PDF via pdftotext
        text = pdfx.extract(r.content)
    else:
        text = r.text

    result = FetchResult(
        url=url,
        final_url=r.url,
        status=r.status_code,
        content_type=content_type,
        text=text,
        via="direct",
        error=None if r.status_code < 400 else f"HTTP {r.status_code}",
        is_pdf=is_pdf,
    )
    # Only cache successful responses (text-only, including extracted PDF text)
    if cache and result.status and result.status < 400 and result.text:
        cache.put(
            "GET",
            url,
            result.final_url,
            result.status,
            result.content_type,
            result.text,
        )
    return result


def fetch_wayback(
    url: str,
    timeout: int = 20,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = None,
    refresh: bool = False,
) -> FetchResult:
    """Fetch the latest archived snapshot from the Wayback Machine."""
    api = f"https://archive.org/wayback/available?url={quote_plus(url)}"
    try:
        r = _do_request("GET", api, timeout, proxy)
        data = r.json()
        snap = data.get("archived_snapshots", {}).get("closest")
        if not snap or not snap.get("available"):
            return FetchResult(url, url, 0, "", "", "wayback", "no snapshot available")
        snap_url = snap["url"]
        # Reuse fetch_direct so the snapshot itself benefits from cache
        snap_res = fetch_direct(
            snap_url, timeout=timeout, proxy=proxy, cache=cache, max_age=max_age, refresh=refresh
        )
        snap_res.url = url
        snap_res.via = "wayback"
        return snap_res
    except (requests.RequestException, ValueError) as e:
        return FetchResult(url, url, 0, "", "", "wayback", str(e))


def fetch_smart(
    url: str,
    timeout: int = 20,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = None,
    refresh: bool = False,
) -> FetchResult:
    """Try direct first; on 4xx/5xx or network error, fall back to Wayback.

    If Wayback is tried, the attempt result is recorded on the returned
    FetchResult (tried_wayback, wayback_status, wayback_error) regardless
    of which result is returned. This makes failures debuggable.
    """
    direct = fetch_direct(
        url, timeout=timeout, proxy=proxy, cache=cache, max_age=max_age, refresh=refresh
    )
    if direct.status and direct.status < 400 and direct.text:
        return direct

    # Direct failed — try Wayback and record what happened
    wb = fetch_wayback(
        url, timeout=timeout, proxy=proxy, cache=cache, max_age=max_age, refresh=refresh
    )

    if wb.status and wb.status < 400 and wb.text:
        wb.tried_wayback = True
        wb.wayback_status = wb.status
        return wb

    # Both failed — return the more informative one but annotate
    chosen = direct if direct.status else wb
    chosen.tried_wayback = True
    chosen.wayback_status = wb.status
    chosen.wayback_error = wb.error
    return chosen


def fetch_many(
    urls: Iterable[str],
    timeout: int = 20,
    proxy: Optional[str] = None,
    parallel: int = 4,
    via: str = "smart",
    cache: Optional[Cache] = None,
    max_age: Optional[float] = None,
    refresh: bool = False,
) -> list[FetchResult]:
    """Fetch many URLs concurrently. Preserves input order in the result."""
    fns = {"smart": fetch_smart, "direct": fetch_direct, "wayback": fetch_wayback}
    fn = fns.get(via, fetch_smart)
    urls = list(urls)
    results: list[Optional[FetchResult]] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futures = {
            ex.submit(
                fn,
                url,
                timeout=timeout,
                proxy=proxy,
                cache=cache,
                max_age=max_age,
                refresh=refresh,
            ): i
            for i, url in enumerate(urls)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = FetchResult(urls[i], urls[i], 0, "", "", via, str(e))
    return [r for r in results if r is not None]  # type: ignore[return-value]


# ----------------------------- Download ------------------------------------


def download(
    url: str,
    out_path: str,
    timeout: int = 120,
    proxy: Optional[str] = None,
    chunk_size: int = 64 * 1024,
    progress: Optional[callable] = None,
) -> tuple[int, int]:
    """Stream a URL to disk. Returns (status_code, bytes_written).

    `progress` is an optional callback `(written, total_or_none)`.
    """
    s = session()
    with s.get(
        url,
        headers=_headers(),
        timeout=timeout,
        stream=True,
        allow_redirects=True,
        proxies=_proxies(proxy),
    ) as r:
        if r.status_code >= 400:
            return r.status_code, 0
        total_str = r.headers.get("Content-Length")
        total = int(total_str) if total_str and total_str.isdigit() else None
        written = 0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written, total)
        return r.status_code, written


# ----------------------------- Text extraction -----------------------------


def _extract_tables(html: str) -> str:
    """Extract all <table> elements as plain text, one per block."""
    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []
    for i, table in enumerate(soup.find_all("table"), start=1):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            parts.append(f"--- table {i} ---\n" + "\n".join(rows))
    return "\n\n".join(parts)


def _extract_raw(html: str) -> str:
    """Strip scripts/styles, return all body text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    return (soup.body or soup).get_text("\n", strip=True)


def html_to_text(
    html: str,
    max_chars: Optional[int] = None,
    skip_chars: int = 0,
    mode: str = "article",
) -> str:
    """Extract readable text from HTML.

    mode:
      - "article" (default): trafilatura main-content extraction with BS fallback
      - "tables": extract all <table> elements only
      - "raw":    full body text, scripts/styles stripped

    skip_chars: drop the first N characters of the extracted text before
        applying max_chars. Useful when page chrome appears at the top of
        an extraction and you want to skip past it.
    """
    text = ""
    if mode == "tables":
        text = _extract_tables(html)
    elif mode == "raw":
        text = _extract_raw(html)
    else:  # "article"
        if HAVE_TRAFILATURA:
            try:
                text = (
                    trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=True,
                        favor_recall=True,
                        no_fallback=False,
                    )
                    or ""
                )
            except Exception:
                text = ""
        if not text:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            text = main.get_text("\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text)
    if skip_chars and skip_chars < len(text):
        text = text[skip_chars:]
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"
    return text


def grep_lines(
    text: str,
    pattern: str,
    context: int = 2,
    ignore_case: bool = True,
) -> str:
    """Return lines matching pattern with N lines of context around each.

    Output separates non-adjacent match groups with `--` (like grep -C).
    Returns the empty string if nothing matches.
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error:
        rx = re.compile(re.escape(pattern), flags)
    lines = text.splitlines()
    keep: set[int] = set()
    for i, line in enumerate(lines):
        if rx.search(line):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                keep.add(j)
    if not keep:
        return ""
    out: list[str] = []
    prev = -2
    for i in sorted(keep):
        if out and i > prev + 1:
            out.append("--")
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def _since_to_ddg_df(since: Optional[str]) -> Optional[str]:
    """Map a --since value to DuckDuckGo's `df` parameter.

    Accepts: 'd' / 'w' / 'm' / 'y' (past day/week/month/year),
             'Nd' / 'Nw' / 'Nm' / 'Ny' (coerced to closest bucket),
             'YYYY-MM-DD' (open-ended range starting that date).
    Returns None if unrecognized (best-effort, never raises).
    """
    if not since:
        return None
    s = since.strip().lower()
    if s in ("d", "w", "m", "y"):
        return s
    m = re.match(r"^(\d+)([dwmy])$", s)
    if m:
        return m.group(2)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return f"{s}.."
    if re.match(r"^\d{4}$", s):
        return f"{s}-01-01.."
    return None


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


# ----------------------------- Searching -----------------------------------


# Selector candidates per engine. We try each until one yields results.
# This makes parsers resilient to minor HTML redesigns.
DDG_RESULT_SELECTORS = [
    {"item": "div.result", "title": "a.result__a", "snippet": "a.result__snippet, div.result__snippet"},
    {"item": "div.results_links", "title": "a.result__a", "snippet": "div.result__snippet"},
    {"item": "article", "title": "a", "snippet": "p"},
]

BING_RESULT_SELECTORS = [
    {"item": "li.b_algo", "title": "h2 a", "snippet": "div.b_caption p, p.b_lineclamp2, p.b_lineclamp3, p.b_lineclamp4"},
    {"item": "li.b_algo", "title": "h2 a", "snippet": "div.b_caption"},
    {"item": "div.b_algo", "title": "h2 a", "snippet": "p"},
]


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        # strip leading "www."
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _filter_excluded(
    results: list["SearchResult"], exclude: Optional[list[str]]
) -> list["SearchResult"]:
    """Drop results whose domain matches any entry in `exclude` (suffix match)."""
    if not exclude:
        return results
    excl = [e.lower().lstrip(".") for e in exclude if e]
    out: list[SearchResult] = []
    for r in results:
        host = _domain_of(r.url)
        if any(host == e or host.endswith("." + e) for e in excl):
            continue
        out.append(r)
    # Re-rank
    for i, r in enumerate(out, start=1):
        r.rank = i
    return out


def _parse_results(html: str, selectors_list: list[dict], unwrap_ddg: bool = False) -> list[SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    for sels in selectors_list:
        results: list[SearchResult] = []
        for i, item in enumerate(soup.select(sels["item"]), start=1):
            a = item.select_one(sels["title"])
            snip = item.select_one(sels["snippet"]) if sels.get("snippet") else None
            if not a:
                continue
            href = a.get("href", "")
            if unwrap_ddg:
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = unquote(m.group(1))
            if not href or href.startswith("#"):
                continue
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    title=a.get_text(" ", strip=True),
                    url=href,
                    snippet=snip.get_text(" ", strip=True) if snip else "",
                )
            )
        if results:
            return results
    return []


def search_duckduckgo(
    query: str,
    max_results: int = 10,
    timeout: int = 20,
    proxy: Optional[str] = None,
    exclude: Optional[list[str]] = None,
    since: Optional[str] = None,
    trust: str = "any",
    prefer: Optional[str] = None,
) -> list[SearchResult]:
    """Scrape DuckDuckGo's HTML endpoint. No API key needed."""
    url = "https://html.duckduckgo.com/html/"
    data: dict[str, str] = {"q": query, "kl": "us-en"}
    df = _since_to_ddg_df(since)
    if df:
        data["df"] = df
    try:
        r = _do_request(
            "POST",
            url,
            timeout,
            proxy,
            data=data,
            referer="https://duckduckgo.com/",
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"DuckDuckGo request failed: {e}") from e

    results = _parse_results(r.text, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    if not results and r.status_code == 200:
        block = _looks_blocked(r.text)
        if block:
            raise RuntimeError(f"DuckDuckGo blocked us (matched '{block}' in response)")
        if _looks_empty(r.text):
            return []  # genuinely no results — not an error
        raise RuntimeError(
            "DuckDuckGo returned 200 but no results parsed — selectors may be stale "
            "(HTML structure changed?)"
        )
    results = _filter_excluded(results, exclude)
    results = reputation.filter_and_rank(results, trust=trust, prefer=prefer)
    return results[:max_results]


def search_bing(
    query: str,
    max_results: int = 10,
    timeout: int = 20,
    proxy: Optional[str] = None,
    exclude: Optional[list[str]] = None,
    since: Optional[str] = None,
    trust: str = "any",
    prefer: Optional[str] = None,
) -> list[SearchResult]:
    """Scrape Bing search results as a fallback engine.

    `since` is best-effort on Bing — it doesn't honor a stable public date
    operator, so we skip it here rather than corrupt the query. DDG handles
    date filtering cleanly; for strict date filtering prefer engine=ddg.
    """
    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
    try:
        r = _do_request("GET", url, timeout, proxy, referer="https://www.bing.com/")
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Bing request failed: {e}") from e

    results = _parse_results(r.text, BING_RESULT_SELECTORS)
    if not results and r.status_code == 200:
        block = _looks_blocked(r.text)
        if block:
            raise RuntimeError(f"Bing blocked us (matched '{block}' in response)")
        if _looks_empty(r.text):
            return []  # genuinely no results — not an error
        raise RuntimeError(
            "Bing returned 200 but no results parsed — selectors may be stale "
            "(HTML structure changed?)"
        )
    results = _filter_excluded(results, exclude)
    results = reputation.filter_and_rank(results, trust=trust, prefer=prefer)
    return results[:max_results]


def search_smart(
    query: str,
    max_results: int = 10,
    proxy: Optional[str] = None,
    exclude: Optional[list[str]] = None,
    since: Optional[str] = None,
    trust: str = "any",
    prefer: Optional[str] = None,
) -> tuple[str, list[SearchResult]]:
    """Try DuckDuckGo first, then Bing.

    A genuine empty-results return from one engine is treated as a successful
    search, not an error — we still try the next engine in case it has results,
    but we won't fail loudly if both legitimately return zero.
    """
    errors: list[EngineError] = []
    saw_empty = False
    for engine_name, engine in (("duckduckgo", search_duckduckgo), ("bing", search_bing)):
        try:
            results = engine(
                query,
                max_results=max_results,
                proxy=proxy,
                exclude=exclude,
                since=since,
                trust=trust,
                prefer=prefer,
            )
            if results:
                return engine_name, results
            saw_empty = True  # legitimate empty
        except Exception as e:
            errors.append(EngineError(engine_name, str(e)))
            time.sleep(0.5)
    if saw_empty and not errors:
        return "duckduckgo", []  # genuinely no hits anywhere
    detail = "; ".join(f"{e.engine}: {e.error}" for e in errors)
    raise RuntimeError(f"all search engines failed: {detail}")


def search_many(
    queries: Iterable[str],
    max_results: int = 10,
    proxy: Optional[str] = None,
    parallel: int = 4,
    dedupe: bool = True,
    exclude: Optional[list[str]] = None,
    since: Optional[str] = None,
    trust: str = "any",
    prefer: Optional[str] = None,
) -> dict:
    """Run multiple search queries in parallel and return combined results.

    Returns a dict with per-query results and (if dedupe=True) a unified
    deduplicated list of all unique URLs across all queries, which is
    re-scored/reordered by reputation so the best sources float to the top.
    """
    queries = list(queries)
    per_query: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futures = {
            ex.submit(
                search_smart,
                q,
                max_results=max_results,
                proxy=proxy,
                exclude=exclude,
                since=since,
                trust=trust,
                prefer=prefer,
            ): q
            for q in queries
        }
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                engine, results = fut.result()
                per_query[q] = {"engine": engine, "results": [r.to_dict() for r in results]}
            except Exception as e:
                per_query[q] = {"engine": None, "results": [], "error": str(e)}

    out: dict = {"queries": per_query}
    if dedupe:
        seen: set[str] = set()
        merged: list[dict] = []
        for q in queries:
            for r in per_query.get(q, {}).get("results", []):
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
                merged.append({**r, "from_query": q})
        # Re-apply reputation across the merged list so prefer/trust work
        # against the combined pool, not just within each query.
        merged = reputation.filter_and_rank(merged, trust=trust, prefer=prefer)
        out["unique"] = merged
    return out


# ----------------------------- Selftest ------------------------------------


def selftest(proxy: Optional[str] = None) -> dict:
    """Run a quick smoke test of search and fetch parsers. Returns a dict report."""
    report: dict = {"timestamp": time.time(), "checks": []}

    def check(name: str, fn):
        entry: dict = {"name": name}
        t0 = time.time()
        try:
            entry["result"] = fn()
            entry["ok"] = True
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
        entry["elapsed_s"] = round(time.time() - t0, 2)
        report["checks"].append(entry)

    # Use a generic query — some specific queries trigger captcha challenges.
    q = "wikipedia"
    check(
        f"ddg search '{q}'",
        lambda: f"{len(search_duckduckgo(q, max_results=3, proxy=proxy))} results",
    )
    check(
        f"bing search '{q}'",
        lambda: f"{len(search_bing(q, max_results=3, proxy=proxy))} results",
    )
    check(
        "fetch wikipedia.org (smart)",
        lambda: f"status={fetch_smart('https://en.wikipedia.org/wiki/Maimonides', proxy=proxy, cache=None).status}",
    )
    check("trafilatura available", lambda: HAVE_TRAFILATURA)
    report["ok"] = all(c["ok"] for c in report["checks"])
    return report

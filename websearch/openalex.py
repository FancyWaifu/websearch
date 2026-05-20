"""OpenAlex academic-paper search.

Ported from the research_tool Flask app. OpenAlex is a free, no-auth REST
API covering ~250M scholarly works with abstracts (as inverted indexes),
citation counts, OA URLs, journal info, and funder/grant data.

Auth is optional — passing a contact email via `--mailto` or
`OPENALEX_API_KEY` puts the request in the "polite pool" with higher rate
limits. See https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication.
"""
from __future__ import annotations

import csv
import io
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

from .cache import Cache, default as default_cache
from .core import _proxies, _headers, session as _session

OPENALEX_BASE = "https://api.openalex.org"
PER_PAGE = 50
# Per-call default TTL for OpenAlex JSON responses. Paper metadata is stable
# — citations move slowly, abstracts don't change — so a long cache window is
# the right default. Override with --max-age on the CLI.
DEFAULT_MAX_AGE = 24 * 60 * 60


@dataclass
class Article:
    openalex_id: str
    title: str
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]
    journal: Optional[str]
    cited_by_count: int
    is_oa: bool
    oa_url: Optional[str]
    abstract: str
    type: str
    topics: list[str] = field(default_factory=list)
    funders: list[dict] = field(default_factory=list)
    referenced_works: list[str] = field(default_factory=list)
    related_works: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# HTTP — uses the websearch Cache + session + proxy stack
# ---------------------------------------------------------------------------

def _get_json(
    url: str,
    params: dict,
    *,
    cache: Optional[Cache],
    max_age: Optional[float],
    refresh: bool,
    proxy: Optional[str],
    timeout: int = 30,
) -> dict:
    """GET with caching. Mirrors core.fetch_direct's cache contract.

    The cache key includes `params` (via Cache._key), so different filter
    combos cache independently. Cursor pagination (which produces unique
    URLs per page) is intentionally not cached by callers — see
    `_fetch_articles_paginated`.
    """
    if cache and not refresh:
        hit = cache.get("GET", url, data=params, max_age=max_age)
        if hit is not None:
            import json as _json
            return _json.loads(hit.body)

    r = _session().get(
        url,
        params=params,
        headers=_headers(),
        proxies=_proxies(proxy),
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.text

    if cache:
        cache.put(
            "GET",
            url,
            r.url,
            r.status_code,
            r.headers.get("Content-Type", "application/json"),
            body,
            data=params,
        )

    import json as _json
    return _json.loads(body)


# ---------------------------------------------------------------------------
# Abstract reconstruction
# ---------------------------------------------------------------------------

def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """OpenAlex stores abstracts as {word: [positions...]}. Rebuild plaintext."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    text = " ".join(word for _, word in word_positions)
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------

def build_filters(
    *,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_citations: Optional[int] = None,
    oa_only: bool = False,
    field: Optional[str] = None,
) -> list[str]:
    """Build OpenAlex filter clauses. Exposed for tests."""
    filters: list[str] = []
    if year_min and year_max:
        filters.append(f"publication_year:{year_min}-{year_max}")
    elif year_min:
        filters.append(f"publication_year:{year_min}-")
    elif year_max:
        filters.append(f"publication_year:-{year_max}")
    if min_citations:
        filters.append(f"cited_by_count:>{min_citations - 1}")
    if oa_only:
        filters.append("open_access.is_oa:true")
    if field:
        filters.append(f"topics.field.display_name.search:{field}")
    return filters


def _sort_param(sort_by: Optional[str]) -> Optional[str]:
    return {
        "citations": "cited_by_count:desc",
        "newest": "publication_year:desc",
        "oldest": "publication_year:asc",
    }.get(sort_by or "")


def _parse_work(work: dict) -> Article:
    authors = [
        a.get("author", {}).get("display_name", "Unknown")
        for a in work.get("authorships", [])
    ]
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    oa_info = work.get("open_access") or {}
    return Article(
        openalex_id=work.get("id", ""),
        title=work.get("display_name") or work.get("title") or "Untitled",
        authors=authors,
        year=work.get("publication_year"),
        doi=work.get("doi"),
        journal=source.get("display_name"),
        cited_by_count=work.get("cited_by_count", 0),
        is_oa=oa_info.get("is_oa", False),
        oa_url=oa_info.get("oa_url"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index") or {}),
        type=work.get("type", ""),
        topics=[t.get("display_name", "") for t in (work.get("topics") or [])[:5]],
        funders=[
            {
                "name": g.get("funder_display_name", "Unknown"),
                "award_id": g.get("award_id", ""),
            }
            for g in (work.get("grants") or [])
        ],
        referenced_works=work.get("referenced_works", []) or [],
        related_works=work.get("related_works", []) or [],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preview_search(
    query: str,
    *,
    mailto: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_citations: Optional[int] = None,
    oa_only: bool = False,
    field: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = DEFAULT_MAX_AGE,
    refresh: bool = False,
    proxy: Optional[str] = None,
) -> int:
    """Return the total count of matching works without fetching pages."""
    params: dict = {"search": query, "per_page": 1, "page": 1}
    if mailto:
        params["mailto"] = mailto
    filters = build_filters(
        year_min=year_min, year_max=year_max,
        min_citations=min_citations, oa_only=oa_only, field=field,
    )
    if filters:
        params["filter"] = ",".join(filters)
    data = _get_json(
        f"{OPENALEX_BASE}/works", params,
        cache=cache if cache is not None else default_cache(),
        max_age=max_age, refresh=refresh, proxy=proxy,
    )
    return data.get("meta", {}).get("count", 0)


def fetch_articles(
    query: str,
    *,
    max_results: int = 25,
    mailto: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_citations: Optional[int] = None,
    oa_only: bool = False,
    field: Optional[str] = None,
    sort_by: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = DEFAULT_MAX_AGE,
    refresh: bool = False,
    proxy: Optional[str] = None,
    progress: Optional[callable] = None,
) -> tuple[list[Article], int]:
    """Fetch up to `max_results` articles from OpenAlex via cursor pagination.

    Returns (articles, total_match_count).
    """
    cache_obj = cache if cache is not None else default_cache()
    articles: list[Article] = []
    total_count = 0
    cursor: Optional[str] = "*"
    page = 0

    filters = build_filters(
        year_min=year_min, year_max=year_max,
        min_citations=min_citations, oa_only=oa_only, field=field,
    )
    sort = _sort_param(sort_by)

    while len(articles) < max_results and cursor:
        per_page = min(PER_PAGE, max_results - len(articles))
        params: dict = {"search": query, "per_page": per_page, "cursor": cursor}
        if mailto:
            params["mailto"] = mailto
        if filters:
            params["filter"] = ",".join(filters)
        if sort:
            params["sort"] = sort

        page += 1
        if progress:
            progress(page, len(articles), max_results)

        data = _get_json(
            f"{OPENALEX_BASE}/works", params,
            cache=cache_obj, max_age=max_age, refresh=refresh, proxy=proxy,
        )
        if page == 1:
            total_count = data.get("meta", {}).get("count", 0)
        results = data.get("results", [])
        if not results:
            break
        cursor = data.get("meta", {}).get("next_cursor")
        for work in results:
            articles.append(_parse_work(work))

    return articles, total_count


def fetch_related_articles(
    openalex_id: str,
    *,
    limit: int = 5,
    mailto: Optional[str] = None,
    cache: Optional[Cache] = None,
    max_age: Optional[float] = DEFAULT_MAX_AGE,
    refresh: bool = False,
    proxy: Optional[str] = None,
) -> list[dict]:
    """Fetch related works for a given OpenAlex work ID."""
    short_id = openalex_id.rsplit("/", 1)[-1] if "/" in openalex_id else openalex_id
    params: dict = {"filter": f"related_to:{short_id}", "per_page": limit}
    if mailto:
        params["mailto"] = mailto
    try:
        data = _get_json(
            f"{OPENALEX_BASE}/works", params,
            cache=cache if cache is not None else default_cache(),
            max_age=max_age, refresh=refresh, proxy=proxy,
        )
    except requests.RequestException:
        return []
    return [
        {
            "openalex_id": w.get("id", ""),
            "title": w.get("display_name", "Untitled"),
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
            "cited_by_count": w.get("cited_by_count", 0),
            "is_oa": (w.get("open_access") or {}).get("is_oa", False),
        }
        for w in data.get("results", [])
    ]


def citation_network(articles: list[Article]) -> dict:
    """Build a nodes/edges graph from articles' referenced_works.

    Only edges between articles in the input set are emitted, so the graph
    shows the internal citation structure of a result set rather than the
    full reference list.
    """
    id_set = {a.openalex_id for a in articles}
    nodes = [
        {"id": a.openalex_id, "title": a.title, "year": a.year, "citations": a.cited_by_count}
        for a in articles
    ]
    edges = [
        {"source": a.openalex_id, "target": ref}
        for a in articles
        for ref in a.referenced_works
        if ref in id_set
    ]
    return {"nodes": nodes, "edges": edges}


def download_pdf(
    article: Article,
    out_dir: Path,
    *,
    proxy: Optional[str] = None,
    timeout: int = 60,
) -> Optional[Path]:
    """Download an OA PDF if available. Returns the local path or None.

    Bypasses the SQLite cache (which is for text), writes directly to disk,
    and verifies Content-Type contains 'pdf' before saving — OA URLs
    occasionally redirect to an HTML landing page.
    """
    if not article.oa_url:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\s-]", "", article.title)[:60].strip().replace(" ", "_")
    path = out_dir / f"{safe or article.openalex_id.rsplit('/', 1)[-1]}.pdf"
    if path.exists():
        return path
    try:
        r = _session().get(
            article.oa_url,
            timeout=timeout,
            stream=True,
            headers={**_headers(), "Accept": "application/pdf"},
            proxies=_proxies(proxy),
        )
        if r.status_code != 200 or "pdf" not in r.headers.get("Content-Type", "").lower():
            return None
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return path
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

def to_bibtex(articles: list[Article]) -> str:
    """Emit BibTeX @article entries with synthetic LastnameYear_N cite keys."""
    entries = []
    for i, art in enumerate(articles):
        first_author = art.authors[0].split()[-1] if art.authors else "Unknown"
        year = art.year or "n.d."
        key = re.sub(r"[^a-zA-Z0-9_]", "", f"{first_author}{year}_{i+1}")
        authors_str = " and ".join(art.authors[:10])
        doi_field = ""
        if art.doi:
            doi_field = f"  doi = {{{art.doi.replace('https://doi.org/', '')}}},\n"
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{art.title}}},\n"
            f"  author = {{{authors_str}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{{art.journal or 'N/A'}}},\n"
            f"{doi_field}"
            f"}}"
        )
    return "\n\n".join(entries)


def to_csv(articles: list[Article]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Title", "Authors", "Year", "Journal", "DOI", "Citations",
                "Open Access", "Funders", "Abstract"])
    for art in articles:
        w.writerow([
            art.title,
            "; ".join(art.authors[:10]),
            str(art.year or ""),
            art.journal or "",
            art.doi or "",
            str(art.cited_by_count),
            "Yes" if art.is_oa else "No",
            "; ".join(f["name"] for f in art.funders) or "N/A",
            art.abstract,
        ])
    return buf.getvalue()


def to_ris(articles: list[Article]) -> str:
    out = []
    for art in articles:
        lines = ["TY  - JOUR", f"TI  - {art.title}"]
        lines += [f"AU  - {a}" for a in art.authors[:10]]
        if art.year:
            lines.append(f"PY  - {art.year}")
        if art.journal:
            lines.append(f"JO  - {art.journal}")
        if art.doi:
            lines.append(f"DO  - {art.doi.replace('https://doi.org/', '')}")
        if art.abstract:
            lines.append(f"AB  - {art.abstract}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def to_markdown(query: str, articles: list[Article]) -> str:
    """Compact markdown report — same shape as research_tool's format_markdown."""
    lines = [f'# Research Results: "{query}"', f"Retrieved {len(articles)} articles", ""]
    for i, art in enumerate(articles, 1):
        authors_str = ", ".join(art.authors[:5])
        if len(art.authors) > 5:
            authors_str += f" (+{len(art.authors) - 5} more)"
        lines += [
            f"## {i}. {art.title}",
            f"- **Authors:** {authors_str}",
            f"- **Year:** {art.year or 'N/A'}",
            f"- **Journal:** {art.journal or 'N/A'}",
            f"- **DOI:** {art.doi or 'N/A'}",
            f"- **Citations:** {art.cited_by_count}",
            f"- **Open Access:** {'Yes' if art.is_oa else 'No'}",
        ]
        if art.funders:
            lines.append(f"- **Funded By:** {', '.join(f['name'] for f in art.funders)}")
        if art.topics:
            lines.append(f"- **Topics:** {', '.join(art.topics)}")
        lines.append("")
        if art.abstract:
            lines += ["### Abstract", art.abstract, ""]
        lines += ["---", ""]
    return "\n".join(lines)


def resolve_mailto(explicit: Optional[str]) -> Optional[str]:
    """Resolve OpenAlex polite-pool email: explicit > OPENALEX_API_KEY env."""
    return explicit or os.environ.get("OPENALEX_API_KEY") or None

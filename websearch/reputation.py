"""Domain reputation and source-type filtering.

Heuristic allow/block lists used by the `--trust` and `--prefer` flags to
drop obvious SEO/affiliate farms and preferentially rank trusted sources
(.gov, .edu, peer-reviewed journals, established news outlets).

The lists are intentionally small and opinionated — grow them from real
noise observed during use, not by trying to be exhaustive.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse


# Domains observed producing AI-slop / affiliate-farm content during real
# research sessions. Only add entries with a concrete reason.
BLOCKLIST: set[str] = {
    "vape-warehouse.com",
    "metrovapemall.com",
    "answers.com",
    "ehow.com",
    "chegg.com",
    "coursehero.com",
}


# Trusted sources grouped by category. Used by --prefer and --trust high.
TRUSTED: dict[str, set[str]] = {
    "academic": {
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "nejm.org",
        "bmj.com",
        "thelancet.com",
        "jamanetwork.com",
        "cell.com",
        "plos.org",
        "springer.com",
        "link.springer.com",
        "academic.oup.com",
        "onlinelibrary.wiley.com",
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "semanticscholar.org",
        "scholar.google.com",
    },
    "gov": {
        "cdc.gov",
        "nih.gov",
        "fda.gov",
        "nhs.uk",
        "who.int",
        "gov.uk",
        "europa.eu",
        "un.org",
        "oecd.org",
        "sec.gov",
        "ftc.gov",
        "epa.gov",
        "usda.gov",
        "treasury.gov",
        "ssa.gov",
        "whitehouse.gov",
        "congress.gov",
    },
    "news": {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "theguardian.com",
        "ft.com",
        "economist.com",
        "bloomberg.com",
        "npr.org",
        "pbs.org",
        "propublica.org",
        "theatlantic.com",
        "newyorker.com",
    },
    "reference": {
        "en.wikipedia.org",
        "wikipedia.org",
        "britannica.com",
        "archive.org",
        "stanford.plato.edu",
        "plato.stanford.edu",
    },
}

# TLDs that are implicitly trusted
TRUSTED_TLD_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".edu",
    ".mil",
    ".int",
    ".ac.uk",
    ".edu.au",
    ".gov.uk",
)


# Cheap URL-shape signals of SEO spam
SEO_SPAM_URL_PATTERNS = [
    re.compile(r"[?&](ref|utm_|aff|affiliate)=", re.IGNORECASE),
    re.compile(r"/affiliate/", re.IGNORECASE),
    re.compile(r"buyers?-?roadmap", re.IGNORECASE),
    re.compile(r"/ranked-?guide", re.IGNORECASE),
]


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _in_category(host: str, category: str) -> bool:
    domains = TRUSTED.get(category, set())
    return host in domains or any(host.endswith("." + d) for d in domains)


def category_of(host: str) -> Optional[str]:
    for cat in TRUSTED:
        if _in_category(host, cat):
            return cat
    return None


def score(url: str) -> int:
    """Heuristic reputation score. Higher = more trusted.

    Anchors:
      +3  explicit trusted allowlist
      +2  trusted TLD (.gov, .edu, etc.)
      -2  SEO spam URL pattern
      -10 explicit blocklist
    """
    host = domain_of(url)
    if not host:
        return 0
    if host in BLOCKLIST or any(host.endswith("." + b) for b in BLOCKLIST):
        return -10
    s = 0
    if category_of(host):
        s += 3
    if any(host.endswith(tld) for tld in TRUSTED_TLD_SUFFIXES):
        s += 2
    for pat in SEO_SPAM_URL_PATTERNS:
        if pat.search(url):
            s -= 2
            break
    return s


def filter_and_rank(
    results: list,
    trust: str = "any",
    prefer: Optional[str] = None,
) -> list:
    """Apply trust filtering and preferential ranking.

    trust:
      - "any":    no filter (default)
      - "medium": drop blocklisted/spam (score <= -5)
      - "high":   keep only explicit trusted sources (score >= 2)

    prefer: boost category matches (+5) then stable-sort by score desc.

    Works on both SearchResult dataclasses and plain dicts. Results are
    re-ranked 1..N after filtering.
    """
    if not results:
        return results

    def get_url(r):
        return r.url if hasattr(r, "url") else r.get("url", "")

    scored: list[tuple[int, int, object]] = []
    for i, r in enumerate(results):
        url = get_url(r)
        s = score(url)
        if prefer and category_of(domain_of(url)) == prefer:
            s += 5
        scored.append((s, i, r))

    if trust == "medium":
        scored = [x for x in scored if x[0] > -5]
    elif trust == "high":
        scored = [x for x in scored if x[0] >= 2]

    if prefer or trust != "any":
        scored.sort(key=lambda x: (-x[0], x[1]))

    out = [x[2] for x in scored]
    for i, r in enumerate(out, start=1):
        if hasattr(r, "rank"):
            r.rank = i
        elif isinstance(r, dict):
            r["rank"] = i
    return out

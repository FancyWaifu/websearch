"""Domain reputation and source-type filtering.

Heuristic allow/block lists used by the `--trust` and `--prefer` flags to
drop obvious SEO/affiliate farms and preferentially rank trusted sources
(.gov, .edu, peer-reviewed journals, established news outlets).

The lists are intentionally small and opinionated — grow them from real
noise observed during use, not by trying to be exhaustive.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Persistent user block/allow lists. Plain-text, one domain per line, '#'
# comments allowed. Curated via `websearch reputation block|allow <domain>`
# so tuning source quality never requires editing this file.
# ---------------------------------------------------------------------------
USER_CONFIG_DIR = Path(
    os.environ.get("WEBSEARCH_CONFIG_DIR", Path.home() / ".config" / "websearch")
)
_USER_BLOCK_FILE = "blocklist.txt"
_USER_ALLOW_FILE = "allowlist.txt"
_user_cache: dict[str, set[str]] = {}


def _load_domain_file(name: str) -> set[str]:
    p = USER_CONFIG_DIR / name
    out: set[str] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip().lower()
            if line.startswith("www."):
                line = line[4:]
            if line:
                out.add(line)
    return out


def user_blocklist() -> set[str]:
    if "block" not in _user_cache:
        _user_cache["block"] = _load_domain_file(_USER_BLOCK_FILE)
    return _user_cache["block"]


def user_allowlist() -> set[str]:
    if "allow" not in _user_cache:
        _user_cache["allow"] = _load_domain_file(_USER_ALLOW_FILE)
    return _user_cache["allow"]


def _matches(host: str, domains: set[str]) -> bool:
    return host in domains or any(host.endswith("." + d) for d in domains)


def edit_user_list(kind: str, domain: str, remove: bool = False) -> str:
    """Add or remove `domain` from the user block/allow list on disk.

    `kind` is "block" or "allow". Returns the absolute path of the file
    written. Invalidates the in-process cache so the change takes effect
    immediately within a long-running process (e.g. the MCP server).
    """
    if kind not in ("block", "allow"):
        raise ValueError("kind must be 'block' or 'allow'")
    fname = _USER_BLOCK_FILE if kind == "block" else _USER_ALLOW_FILE
    domain = domain_of(domain) or domain.strip().lower().lstrip("www.")
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    p = USER_CONFIG_DIR / fname
    current = _load_domain_file(fname)
    if remove:
        current.discard(domain)
    else:
        current.add(domain)
    header = f"# websearch user {kind}list — one domain per line\n"
    p.write_text(header + "\n".join(sorted(current)) + "\n", encoding="utf-8")
    _user_cache.pop(kind, None)
    return str(p)


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
    re.compile(r"[?&](ref|utm_|aff|affiliate|tag|campaign)=", re.IGNORECASE),
    re.compile(r"/affiliate/", re.IGNORECASE),
    re.compile(r"buyers?-?roadmap", re.IGNORECASE),
    re.compile(r"/ranked-?guide", re.IGNORECASE),
    re.compile(r"/best-\w+-(of-)?20\d\d", re.IGNORECASE),
    re.compile(r"/(coupon|promo|discount)-?codes?", re.IGNORECASE),
]


# Affiliate / SEO-doorway signals found in page text or result snippets.
# These catch the affiliate-blog failure mode that URL shape alone misses
# (e.g. a clean-looking URL whose body is monetized listicle filler).
AFFILIATE_TEXT_PATTERNS = [
    re.compile(r"\baffiliate links?\b", re.IGNORECASE),
    re.compile(r"\bwe (may )?earn (a )?commission\b", re.IGNORECASE),
    re.compile(r"\bat no (extra|additional) cost to you\b", re.IGNORECASE),
    re.compile(r"\bcommission(s)? (at|from|on)\b", re.IGNORECASE),
    re.compile(r"\b(discount|promo|coupon) code\b", re.IGNORECASE),
    re.compile(r"\bbest \w[\w\s]{0,30}? of 20\d\d\b", re.IGNORECASE),
    re.compile(r"\bbuy (it )?now\b", re.IGNORECASE),
    re.compile(r"\bshop now\b", re.IGNORECASE),
]


def looks_affiliate(text: str) -> bool:
    """True when `text` (a snippet, or a fetched body) trips an affiliate
    signal. Used both pre-fetch (snippet demotion) and post-fetch (output
    annotation), so the caller can warn that a source is monetized."""
    if not text:
        return False
    return any(p.search(text) for p in AFFILIATE_TEXT_PATTERNS)


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


def score(url: str, text: str = "") -> int:
    """Heuristic reputation score. Higher = more trusted.

    Anchors:
      +3  explicit trusted allowlist
      +2  trusted TLD (.gov, .edu, etc.)
      -2  SEO spam URL pattern
      -3  affiliate signal in `text` (title+snippet, when supplied)
      -10 explicit blocklist

    `text` is optional so existing url-only callers are unaffected; pass a
    result's title+snippet to demote affiliate-blog content pre-fetch.
    """
    host = domain_of(url)
    if not host:
        return 0
    if _matches(host, BLOCKLIST) or _matches(host, user_blocklist()):
        return -10
    s = 0
    if category_of(host) or _matches(host, user_allowlist()):
        s += 3
    if any(host.endswith(tld) for tld in TRUSTED_TLD_SUFFIXES):
        s += 2
    for pat in SEO_SPAM_URL_PATTERNS:
        if pat.search(url):
            s -= 2
            break
    if text and looks_affiliate(text):
        s -= 3
    return s


def explain(url: str) -> dict:
    """Human-readable trace of why a URL would be kept, dropped, or boosted.

    Used by `websearch reputation explain URL` to debug "where did my source go?"
    """
    host = domain_of(url)
    reasons: list[str] = []
    if not host:
        return {"url": url, "host": "", "score": 0, "reasons": ["unparseable URL"]}
    if _matches(host, BLOCKLIST):
        reasons.append(f"BLOCKLISTED ({host}): score -10 → dropped at trust=medium and above")
    if _matches(host, user_blocklist()):
        reasons.append(f"USER blocklist match ({host}): score -10 → dropped at trust=medium+")
    cat = category_of(host)
    if cat:
        reasons.append(f"trusted/{cat} allowlist match: +3")
    if _matches(host, user_allowlist()):
        reasons.append(f"USER allowlist match ({host}): +3")
    if any(host.endswith(tld) for tld in TRUSTED_TLD_SUFFIXES):
        reasons.append("trusted TLD: +2")
    for pat in SEO_SPAM_URL_PATTERNS:
        if pat.search(url):
            reasons.append(f"SEO-spam URL pattern '{pat.pattern}': -2")
            break
    s = score(url)
    if not reasons:
        reasons.append("no signals — neutral score 0 (kept at trust=any, dropped at trust=high)")
    return {
        "url": url,
        "host": host,
        "category": cat,
        "score": s,
        "kept_at_trust_medium": s > -5,
        "kept_at_trust_high": s >= 2,
        "reasons": reasons,
    }


def list_category(category: Optional[str] = None) -> dict:
    """Return the allowlist contents. category=None returns all."""
    if category:
        return {category: sorted(TRUSTED.get(category, set()))}
    return {cat: sorted(domains) for cat, domains in TRUSTED.items()}


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

    def get_text(r):
        if hasattr(r, "title"):
            return f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}"
        return f"{r.get('title', '')} {r.get('snippet', '')}"

    scored: list[tuple[int, int, object]] = []
    for i, r in enumerate(results):
        url = get_url(r)
        s = score(url, get_text(r))
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


def cap_per_domain(results: list, n: int) -> list:
    """Keep at most `n` results per domain, preserving rank order.

    Stops one SEO/affiliate farm from monopolizing a research run's fetch
    slots. Ranks are rewritten 1..N on the survivors.
    """
    if not results or n <= 0:
        return results

    def get_url(r):
        return r.url if hasattr(r, "url") else r.get("url", "")

    seen: dict[str, int] = {}
    kept: list = []
    for r in results:
        host = domain_of(get_url(r))
        if seen.get(host, 0) >= n:
            continue
        seen[host] = seen.get(host, 0) + 1
        kept.append(r)
    for i, r in enumerate(kept, start=1):
        if hasattr(r, "rank"):
            r.rank = i
        elif isinstance(r, dict):
            r["rank"] = i
    return kept

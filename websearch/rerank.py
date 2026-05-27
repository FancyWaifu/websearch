"""Lightweight TF-IDF rerank + multi-query consensus boost + keyword filter.

Pure stdlib (math + collections). Designed for tens-to-hundreds of search
snippets, not for serious IR — but it fixes the common failure mode where
the search engine matches a high-frequency phrase in the query and ignores
the rare, defining terms.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Aggressive small stopword list — keeps query-significant terms intact
# without sklearn. Add words here if they show up dominating snippet scores.
_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "in", "on", "at", "of", "to", "for", "with", "by", "from", "as",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "their", "them",
    "i", "you", "he", "she", "we", "they", "me", "us",
    "not", "no", "if", "then", "than", "so", "such", "also",
    "what", "why", "how", "when", "where", "which", "who", "whom",
    "some", "any", "all", "each", "every", "other", "more", "most",
    "about", "into", "over", "under", "out",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _doc_text(r: dict) -> str:
    """Concatenate the fields available on a SearchResult dict."""
    return " ".join([r.get("title", ""), r.get("snippet", "")])


def score_against_query(results: list[dict], query: str) -> list[tuple[float, dict]]:
    """Return [(score, result), ...] in original order. TF-IDF cosine, ish."""
    q_tokens = _tokens(query)
    if not q_tokens or not results:
        return [(0.0, r) for r in results]

    docs = [_tokens(_doc_text(r)) for r in results]
    n_docs = len(docs)

    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))

    def idf(term: str) -> float:
        # +1 smoothing so missing-doc-frequency doesn't divide by zero
        return math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0

    q_vec: dict[str, float] = {}
    for term, count in Counter(q_tokens).items():
        q_vec[term] = count * idf(term)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    out: list[tuple[float, dict]] = []
    for r, d in zip(results, docs):
        if not d:
            out.append((0.0, r))
            continue
        d_vec: dict[str, float] = {}
        for term, count in Counter(d).items():
            d_vec[term] = count * idf(term)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec.get(t, 0.0) * d_vec.get(t, 0.0) for t in q_vec)
        out.append((dot / (q_norm * d_norm), r))
    return out


def rerank(results: list[dict], query: str) -> list[dict]:
    """Reorder results by TF-IDF similarity to the query, descending.

    Original order is preserved as a tiebreaker so zero-overlap docs don't
    get scrambled.
    """
    if not results:
        return results
    scored = score_against_query(results, query)
    indexed = [(s, i, r) for i, (s, r) in enumerate(scored)]
    indexed.sort(key=lambda x: (-x[0], x[1]))
    out = [r for _, _, r in indexed]
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out


def filter_required(results: list[dict], required: Iterable[str]) -> list[dict]:
    """Drop results whose title+snippet contain none of the required terms.

    Match is case-insensitive substring on the lowered concatenation. A result
    keeps if it matches ANY of the required terms (OR semantics).
    """
    terms = [t.strip().lower() for t in required if t and t.strip()]
    if not terms:
        return results
    kept: list[dict] = []
    for r in results:
        haystack = _doc_text(r).lower()
        if any(t in haystack for t in terms):
            kept.append(r)
    for i, r in enumerate(kept, start=1):
        r["rank"] = i
    return kept


_MISSING_RE = re.compile(r"\bMissing:\s*\S", re.IGNORECASE)


def demote_missing_terms(results: list[dict]) -> list[dict]:
    """Push results whose snippet contains a `Missing: ...` annotation to the
    bottom of the list.

    Google (and SearXNG when it proxies Google) appends `Missing: <terms>`
    to a snippet when the page didn't actually contain some of the queried
    terms — these are the engine *admitting* the match is partial. Without
    this demotion, a hyper-specific query like
    "Mizzou INFOTC 4910 digital forensics syllabus 2026" can return an NIH
    spreadsheet as #1 just because it happened to contain "2026". Original
    order is preserved within each bucket so we don't fight the engine's
    relevance ordering on real matches.
    """
    if not results:
        return results
    kept, demoted = [], []
    for r in results:
        snippet = r.get("snippet", "") or ""
        (demoted if _MISSING_RE.search(snippet) else kept).append(r)
    out = kept + demoted
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out


def boost_by_query_count(per_query: dict, merged: list[dict]) -> list[dict]:
    """Re-sort `merged` so URLs that appeared in multiple sub-queries float up.

    `per_query`: {query_string: {"results": [dict, ...], ...}, ...}
    `merged`: deduped union list — each entry's rank is rewritten in place.
    """
    if not merged:
        return merged
    url_count: Counter[str] = Counter()
    for q, info in per_query.items():
        for r in info.get("results", []):
            url_count[r.get("url", "")] += 1
    if not url_count:
        return merged
    # Stable sort: more-query-coverage first, then current order.
    indexed = [(url_count.get(r.get("url", ""), 0), i, r) for i, r in enumerate(merged)]
    indexed.sort(key=lambda x: (-x[0], x[1]))
    out = [r for _, _, r in indexed]
    for i, r in enumerate(out, start=1):
        r["rank"] = i
        r["query_hits"] = url_count.get(r.get("url", ""), 0)
    return out

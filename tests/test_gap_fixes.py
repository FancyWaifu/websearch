"""Tests for the gap fixes found by the 20-probe stress test.

Covers: ad-redirect drops, soft-404 detection, mermaid/diagram stripping,
SERP "Missing:" demotion, PDF whitespace collapse, ISO8601 TZ on
research frontmatter timestamps.
"""
from __future__ import annotations

import datetime as _dt
from unittest import mock

import pytest

from websearch.core import (
    BING_RESULT_SELECTORS,
    DDG_RESULT_SELECTORS,
    _canonical_url,
    _is_ad_redirect,
    _looks_soft_404,
    _parse_results,
    _strip_noprose_fences,
)
from websearch.pdfx import _collapse_pdf_whitespace
from websearch.rerank import demote_missing_terms


# ---------- Ad-redirect drops (Bug 2) ----------

@pytest.mark.parametrize(
    "url",
    [
        "https://duckduckgo.com/y.js?ad_domain=cybernews.com&u3=foo",
        "https://duckduckgo.com/y.js?ad_provider=bingv7aa&ad_type=txad",
        "https://www.bing.com/aclick?ld=abc&u=base64here",
        "http://bing.com/aclick?x=1",
    ],
)
def test_is_ad_redirect_matches_known_sponsored_wrappers(url):
    assert _is_ad_redirect(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/page",
        "https://duckduckgo.com/?q=test",       # search, not ad
        "https://www.bing.com/search?q=test",    # search, not ad
        "https://www.bing.com/ck/a?u=a1abc",     # organic ck/a wrapper, not aclick
    ],
)
def test_is_ad_redirect_no_false_positives(url):
    assert _is_ad_redirect(url) is False


def test_parse_results_drops_ddg_ad_redirects():
    html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/y.js?ad_domain=cybernews.com">Sponsored</a>
      <a class="result__snippet">ad</a>
    </div>
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Forganic">Organic</a>
      <a class="result__snippet">real</a>
    </div>
    """
    results = _parse_results(html, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    # The sponsored result is dropped; the organic one unwrapped and kept.
    assert len(results) == 1
    assert results[0].url == "https://example.org/organic"


# ---------- Soft-404 detection (Bug 6) ----------

@pytest.mark.parametrize(
    "body",
    [
        "<h1>404 Not Found</h1>",
        "We can't find the page you're looking for.",
        "Sorry, we can't find that page",
        "Page not found",
        "<title>Oops! Looks like this page got lost</title>",
        "This page does not exist.",
    ],
)
def test_looks_soft_404_matches_common_error_ui(body):
    assert _looks_soft_404(body) is not None


def test_looks_soft_404_ignores_normal_articles():
    body = "A long article about why the moon landings happened in 1969."
    assert _looks_soft_404(body) is None


def test_looks_soft_404_matches_new_phrasings():
    """2026-05-27 additions — patterns that the original set missed."""
    for body in (
        "Sorry, that URL is invalid",
        "This content has been removed by the author.",
        "Nothing here. Try the homepage.",
        "HTTP Status 404",
    ):
        assert _looks_soft_404(body) is not None, f"missed: {body!r}"


def test_looks_soft_404_detects_error_path_in_url():
    """A final_url containing /404, /not-found etc. should flag without
    even reading the body."""
    body = "Some unrelated content that wouldn't otherwise trigger detection"
    assert _looks_soft_404(body, "https://example.com/404") is not None
    assert _looks_soft_404(body, "https://example.com/not-found") is not None
    assert _looks_soft_404(body, "https://example.com/page-not-found/") is not None
    assert _looks_soft_404(body, "https://example.com/error") is not None
    # Negative: real article paths shouldn't fire
    assert _looks_soft_404(body, "https://example.com/blog/article") is None


def test_looks_soft_404_skips_huge_bodies_for_speed():
    # Even with the phrase present, very long bodies are skipped — error
    # pages are short, real articles dwarf the regex cost otherwise.
    body = "page not found " + ("x" * 20000)
    assert _looks_soft_404(body) is None


# ---------- Mermaid / diagram fence stripping (Bug 7) ----------

def test_strip_noprose_fences_removes_mermaid_block():
    text = "Before\n\n```mermaid\nflowchart LR\n  A --> B\n  B --> C\n```\n\nAfter"
    out = _strip_noprose_fences(text)
    assert "flowchart" not in out
    assert "A --> B" not in out
    assert "[diagram omitted]" in out
    assert "Before" in out and "After" in out


def test_strip_noprose_fences_removes_bare_mermaid_without_fences():
    # The case from probe p02 (eBPF) — Mermaid leaked as plain text.
    text = (
        "Some prose.\n\n"
        "flowchart LR\n"
        "subgraph Kernel\n"
        "  KSRC --> DWARF\n"
        "  DWARF --> PAHOLE\n"
        "end\n\n"
        "More prose."
    )
    out = _strip_noprose_fences(text)
    assert "flowchart" not in out.lower() or "[diagram omitted]" in out
    assert "Some prose." in out
    assert "More prose." in out


def test_strip_noprose_fences_leaves_python_code_alone():
    text = "Here is code:\n\n```python\ndef f(x):\n    return x * 2\n```\n"
    out = _strip_noprose_fences(text)
    assert "def f(x)" in out
    assert "[diagram omitted]" not in out


# ---------- SERP "Missing: term" demotion (Bug 9) ----------

def test_demote_missing_terms_pushes_partial_matches_to_bottom():
    results = [
        {"url": "a", "title": "T1", "snippet": "talks about X. Missing: foo bar"},
        {"url": "b", "title": "T2", "snippet": "real match for foo bar"},
        {"url": "c", "title": "T3", "snippet": "another real match"},
    ]
    out = demote_missing_terms(results)
    assert [r["url"] for r in out] == ["b", "c", "a"]
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_demote_missing_terms_preserves_order_within_buckets():
    results = [
        {"url": "a", "title": "", "snippet": "real 1"},
        {"url": "b", "title": "", "snippet": "Missing: x"},
        {"url": "c", "title": "", "snippet": "real 2"},
        {"url": "d", "title": "", "snippet": "Missing: y z"},
    ]
    out = demote_missing_terms(results)
    assert [r["url"] for r in out] == ["a", "c", "b", "d"]


def test_demote_missing_terms_is_case_insensitive_and_safe_on_empty():
    assert demote_missing_terms([]) == []
    # Lower-cased variant should still match.
    out = demote_missing_terms([
        {"url": "a", "title": "", "snippet": "ok"},
        {"url": "b", "title": "", "snippet": "missing: foo"},
    ])
    assert [r["url"] for r in out] == ["a", "b"]


# ---------- PDF whitespace collapse (Bug 5) ----------

def test_collapse_pdf_whitespace_strips_column_padding():
    # A two-column abstract from pdftotext -layout.
    text = "          Abstract\n          This is the first line\n          and the second.\n"
    out = _collapse_pdf_whitespace(text)
    assert out.startswith("Abstract")
    assert "This is the first line" in out
    # No line should start with 4+ spaces.
    assert not any(line.startswith("    ") for line in out.splitlines())


def test_collapse_pdf_whitespace_keeps_small_indents():
    text = "  indented two spaces\nplain line\n"
    out = _collapse_pdf_whitespace(text)
    # 2 leading spaces are kept (might be intentional indentation).
    assert out.startswith("  indented")


def test_collapse_pdf_whitespace_collapses_inline_runs():
    text = "Column A          Column B          Column C\n"
    out = _collapse_pdf_whitespace(text)
    assert out == "Column A Column B Column C\n"


# ---------- ISO 8601 TZ on frontmatter timestamp (Bug 11) ----------

# ---------- URL canonicalization for dedup ----------

@pytest.mark.parametrize(
    "url,want",
    [
        # arxiv abs/pdf/html variants → all collapse to /abs/<id>
        ("https://arxiv.org/abs/2506.05364", "https://arxiv.org/abs/2506.05364"),
        ("https://arxiv.org/pdf/2506.05364", "https://arxiv.org/abs/2506.05364"),
        ("https://arxiv.org/pdf/2506.05364.pdf", "https://arxiv.org/abs/2506.05364"),
        ("https://arxiv.org/pdf/2506.05364v2", "https://arxiv.org/abs/2506.05364"),
        ("https://arxiv.org/pdf/2506.05364v2.pdf", "https://arxiv.org/abs/2506.05364"),
        ("https://arxiv.org/html/2506.05364", "https://arxiv.org/abs/2506.05364"),
        # Subdomain (some mirrors)
        ("https://www.arxiv.org/pdf/2506.05364", "https://www.arxiv.org/abs/2506.05364"),
        # NOTE: pre-2007 cross-list IDs (cs.AI/0001001) aren't canonicalized —
        # the slash inside the ID confuses the regex. Modern IDs (post-2007,
        # numeric-only) are the vast majority of what we'd see in 2026.
    ],
)
def test_canonical_url_collapses_arxiv_variants(url, want):
    assert _canonical_url(url) == want


def test_canonical_url_strips_mobile_subdomain():
    assert _canonical_url("https://m.example.com/page") == "https://example.com/page"
    assert _canonical_url("https://mobile.bbc.co.uk/news/x") == "https://bbc.co.uk/news/x"


def test_canonical_url_strips_fragment():
    assert _canonical_url("https://example.com/article#section-3") == "https://example.com/article"


def test_canonical_url_strips_trailing_slash():
    assert _canonical_url("https://example.com/article/") == "https://example.com/article"
    # But keeps trailing slash on bare domain (count <= 3 slashes total)
    assert _canonical_url("https://example.com/") == "https://example.com/"


def test_canonical_url_passes_through_unrelated_urls():
    assert _canonical_url("https://example.com/article") == "https://example.com/article"


# ---------- Round-robin multi-query merge ----------

def test_interleave_dedupe_round_robins_across_queries():
    """Each query should contribute its rank-1 first, then rank-2, etc.
    Old sequential behavior would output q1's whole list first, starving
    q2/q3 of early slots — and after rerank, of fetch slots."""
    from websearch.core import _interleave_dedupe
    per_query = {
        "q1": {"results": [{"url": "a"}, {"url": "b"}, {"url": "c"}]},
        "q2": {"results": [{"url": "d"}, {"url": "e"}]},
        "q3": {"results": [{"url": "f"}]},
    }
    merged = _interleave_dedupe(["q1", "q2", "q3"], per_query)
    urls = [r["url"] for r in merged]
    # rank-1 from each: a, d, f. then rank-2: b, e. then rank-3: c.
    assert urls == ["a", "d", "f", "b", "e", "c"]
    # from_query tag preserved
    assert merged[0]["from_query"] == "q1"
    assert merged[1]["from_query"] == "q2"
    assert merged[2]["from_query"] == "q3"


def test_interleave_dedupe_canonicalizes_for_dedup():
    """arxiv abs and pdf URLs of the same paper should NOT both appear."""
    from websearch.core import _interleave_dedupe
    per_query = {
        "q1": {"results": [{"url": "https://arxiv.org/abs/2506.05364"}]},
        "q2": {"results": [{"url": "https://arxiv.org/pdf/2506.05364"}]},
    }
    merged = _interleave_dedupe(["q1", "q2"], per_query)
    assert len(merged) == 1
    assert merged[0]["url"] == "https://arxiv.org/abs/2506.05364"


def test_interleave_dedupe_handles_uneven_lengths_and_empty():
    from websearch.core import _interleave_dedupe
    assert _interleave_dedupe([], {}) == []
    assert _interleave_dedupe(["q"], {"q": {"results": []}}) == []
    # One query has many more results than the other — the long one
    # should still get all its remaining results after the short one is exhausted.
    per_query = {
        "long": {"results": [{"url": f"l{i}"} for i in range(5)]},
        "short": {"results": [{"url": "s0"}]},
    }
    merged = _interleave_dedupe(["long", "short"], per_query)
    urls = [r["url"] for r in merged]
    assert urls == ["l0", "s0", "l1", "l2", "l3", "l4"]


def test_research_frontmatter_timestamp_has_timezone():
    """Smoke test: build a minimal args/report and check the timestamp shape.
    Format is `YYYY-MM-DDTHH:MM:SS+HH:MM` (or `-HH:MM` / `Z` if UTC)."""
    import argparse
    from websearch.cli import _research_frontmatter

    args = argparse.Namespace(
        proxy=None, trust="medium", depth=4,
    )
    report = {"unique": [], "fetched": [], "_proxy_reachable": None, "_backfilled": 0}
    fm = _research_frontmatter(args, ["q"], report)
    ts_line = next(line for line in fm.splitlines() if line.startswith("timestamp:"))
    ts = ts_line.split(":", 1)[1].strip()
    parsed = _dt.datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None, f"timestamp {ts!r} has no tzinfo"

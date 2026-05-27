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

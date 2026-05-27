"""Parser tests against fixture HTML.

These exist because the live DDG/Bing HTML drifts. If a fixture stops
parsing it's a clear signal that the selector list needs updating
*before* the next live research session breaks.
"""
from pathlib import Path

import pytest

from websearch.core import (
    BING_RESULT_SELECTORS,
    DDG_RESULT_SELECTORS,
    _looks_blocked,
    _looks_empty,
    _parse_results,
    _unwrap_bing,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_ddg_parses_and_unwraps_redirect_urls():
    html = _load("ddg.html")
    results = _parse_results(html, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    assert len(results) == 3
    # First result: title, real URL after unwrapping uddg=
    assert results[0].title == "QUIC - Wikipedia"
    assert results[0].url == "https://en.wikipedia.org/wiki/QUIC"
    assert "QUIC is a general-purpose" in results[0].snippet
    # Ranks are 1..N
    assert [r.rank for r in results] == [1, 2, 3]
    # All URLs unwrapped to absolute https
    for r in results:
        assert r.url.startswith("https://")


def test_bing_parses_with_lineclamp_snippets():
    html = _load("bing.html")
    results = _parse_results(html, BING_RESULT_SELECTORS, unwrap_ddg=False)
    assert len(results) == 3
    assert results[0].title == "QUIC - Wikipedia"
    assert results[0].url == "https://en.wikipedia.org/wiki/QUIC"
    assert "transport layer network protocol" in results[0].snippet


def test_blocked_detection_matches_captcha_phrasing():
    html = _load("ddg_blocked.html")
    pat = _looks_blocked(html)
    assert pat is not None
    # Should also fail to parse as real results
    results = _parse_results(html, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    assert results == []


def test_empty_detection_distinguishes_no_results_from_block():
    html = _load("ddg_empty.html")
    assert _looks_blocked(html) is None
    assert _looks_empty(html) is not None
    results = _parse_results(html, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    assert results == []


def test_unwrap_bing_decodes_padded_target():
    # b64("https://example.com/path") = aHR0cHM6Ly9leGFtcGxlLmNvbS9wYXRo (no padding needed)
    url = "https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wYXRo&p=ignored"
    assert _unwrap_bing(url) == "https://example.com/path"


def test_unwrap_bing_decodes_unpadded_target():
    # b64("https://en.wikipedia.org/wiki/QUIC") needs 1 byte of padding;
    # bing serves it unpadded, the unwrapper must add padding back.
    url = (
        "https://www.bing.com/ck/a?!&&"
        "u=a1aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUVVJQw"
        "&p=other"
    )
    assert _unwrap_bing(url) == "https://en.wikipedia.org/wiki/QUIC"


def test_unwrap_bing_is_noop_on_non_bing_url():
    direct = "https://example.com/article"
    assert _unwrap_bing(direct) == direct


def test_unwrap_bing_handles_missing_or_bogus_u_param():
    # No u= at all → leave the wrapper as-is (better than dropping)
    no_u = "https://www.bing.com/ck/a?!&&p=foo"
    assert _unwrap_bing(no_u) == no_u
    # u= present but base64 garbage → don't claim a result
    bogus = "https://www.bing.com/ck/a?!&&u=a1!!!notbase64!!!"
    assert _unwrap_bing(bogus) == bogus


def test_unwrap_bing_rejects_non_http_decode():
    # b64("file:///etc/passwd") = ZmlsZTovLy9ldGMvcGFzc3dk — if Bing's
    # wrapper ever pointed at file://, refuse to surface it (defense in
    # depth: scheme guards should also catch this downstream).
    url = "https://www.bing.com/ck/a?!&&u=a1ZmlsZTovLy9ldGMvcGFzc3dk"
    assert _unwrap_bing(url) == url


def test_parse_results_unwraps_bing_ck_inside_html():
    html = """
    <li class="b_algo">
      <h2><a href="https://www.bing.com/ck/a?!&amp;&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wYXRo&amp;p=x">Example</a></h2>
      <div class="b_caption"><p>An example.</p></div>
    </li>
    """
    results = _parse_results(html, BING_RESULT_SELECTORS, unwrap_ddg=False)
    assert len(results) == 1
    assert results[0].url == "https://example.com/path"


def test_ddg_unwrap_handles_missing_uddg():
    """If a result lacks the uddg= wrapper (DDG sometimes serves direct URLs),
    we still take the href as-is rather than dropping the result."""
    html = """
    <div class="result">
      <a class="result__a" href="https://example.org/page">Direct URL</a>
      <a class="result__snippet">Direct snippet</a>
    </div>
    """
    results = _parse_results(html, DDG_RESULT_SELECTORS, unwrap_ddg=True)
    assert len(results) == 1
    assert results[0].url == "https://example.org/page"

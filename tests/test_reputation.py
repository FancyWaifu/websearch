"""Tests for the reputation scoring and filter/rank logic."""
from dataclasses import dataclass

from websearch import reputation


@dataclass
class _R:
    rank: int
    title: str
    url: str
    snippet: str = ""


def test_score_blocklist_dominates():
    assert reputation.score("https://answers.com/q/foo") == -10
    # Subdomain of a blocklisted host
    assert reputation.score("https://www.answers.com/q/foo") == -10


def test_score_trusted_tld_and_allowlist_combine():
    # cdc.gov is in the gov allowlist (+3) AND has a trusted TLD (.gov, +2)
    assert reputation.score("https://www.cdc.gov/page") == 5
    # Random .gov gets +2 from TLD only
    assert reputation.score("https://obscure-county.gov/x") == 2


def test_score_seo_url_pattern_penalty():
    # The SEO_SPAM_URL_PATTERNS list catches obvious affiliate signals.
    assert reputation.score("https://random.example/article?aff=foo") == -2
    assert reputation.score("https://random.example/affiliate/widget") == -2
    assert reputation.score("https://random.example/buyers-roadmap") == -2


def test_category_of_distinguishes_categories():
    assert reputation.category_of("nature.com") == "academic"
    assert reputation.category_of("cdc.gov") == "gov"
    assert reputation.category_of("bbc.com") == "news"
    assert reputation.category_of("en.wikipedia.org") == "reference"
    assert reputation.category_of("unknown.example") is None


def test_filter_and_rank_trust_high_keeps_only_top_sources():
    results = [
        _R(1, "spam", "https://random.example/a"),
        _R(2, "wiki", "https://en.wikipedia.org/wiki/X"),
        _R(3, "blocked", "https://answers.com/q"),
        _R(4, "gov", "https://www.cdc.gov/page"),
    ]
    out = reputation.filter_and_rank(results, trust="high")
    urls = [r.url for r in out]
    assert "https://answers.com/q" not in urls
    assert "https://random.example/a" not in urls
    assert "https://www.cdc.gov/page" in urls
    assert "https://en.wikipedia.org/wiki/X" in urls
    # Ranks renumbered after filtering
    assert [r.rank for r in out] == list(range(1, len(out) + 1))


def test_filter_and_rank_prefer_boosts_category():
    results = [
        _R(1, "wiki", "https://en.wikipedia.org/wiki/X"),
        _R(2, "academic", "https://www.nature.com/article"),
    ]
    out = reputation.filter_and_rank(results, trust="any", prefer="academic")
    # Academic should now outrank reference
    assert out[0].url == "https://www.nature.com/article"
    assert out[1].url == "https://en.wikipedia.org/wiki/X"


def test_filter_and_rank_works_on_dicts_too():
    results = [
        {"rank": 1, "url": "https://answers.com/q", "title": "blocked"},
        {"rank": 2, "url": "https://www.cdc.gov/page", "title": "gov"},
    ]
    out = reputation.filter_and_rank(results, trust="medium")
    urls = [r["url"] for r in out]
    assert "https://answers.com/q" not in urls
    assert out[0]["rank"] == 1

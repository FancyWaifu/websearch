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


def test_score_seo_farms_added_to_blocklist():
    """The 2026-05-27 blocklist additions — domains observed outranking
    real press during research probes."""
    for url in (
        "https://invisioncommunity.co.uk/marathon-2026-bungies-extraction-shooter-is-it-succeeding/",
        "https://www.exitlag.com/blog/marathon/",
        "https://powermoves.blog/smart-home",
        "https://talkofthehouse.com/best-smart-home",
        "https://lumbercapital.com/2x4-prices",
    ):
        assert reputation.score(url) == -10, f"{url} should be blocklisted"


def test_score_gaming_and_tech_press_in_news_allowlist():
    """Gaming/tech press should land at +3 (news category) — without this
    they tie at 0 with random SEO blogs in research result ranking."""
    for url in (
        "https://www.ign.com/articles/anything",
        "https://www.videogameschronicle.com/review/anything",
        "https://www.gamedeveloper.com/business/anything",
        "https://www.polygon.com/anything",
        "https://www.eurogamer.net/anything",
        "https://www.pcgamer.com/anything",
        "https://arstechnica.com/anything",
        "https://www.theverge.com/anything",
    ):
        assert reputation.score(url) == 3, f"{url} should match news allowlist"


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


def test_score_text_param_optional_and_backward_compatible():
    # url-only callers are unchanged: no text -> no affiliate penalty.
    assert reputation.score("https://random.example/article") == 0


def test_score_affiliate_text_penalty():
    aff = "This post may contain affiliate links. We may earn a commission."
    assert reputation.score("https://random.example/article", aff) == -3
    # Clean text -> no penalty.
    assert reputation.score("https://random.example/article", "plain prose") == 0


def test_looks_affiliate_detects_signals():
    assert reputation.looks_affiliate("at no extra cost to you")
    assert reputation.looks_affiliate("Best CBD Oils of 2026")
    assert reputation.looks_affiliate("use our discount code")
    assert not reputation.looks_affiliate("a neutral encyclopedia article")
    assert not reputation.looks_affiliate("")


def test_filter_and_rank_demotes_affiliate_snippet():
    results = [
        _R(1, "Affiliate blog", "https://blog.example/x",
           "This page may contain affiliate links."),
        _R(2, "Neutral page", "https://other.example/y", "plain summary"),
    ]
    ranked = reputation.filter_and_rank(results, trust="medium")
    # Both survive medium trust, but the neutral page ranks above the
    # affiliate one after the -3 penalty.
    assert ranked[0].url == "https://other.example/y"


def test_cap_per_domain_limits_and_reranks():
    results = [
        _R(1, "a1", "https://spam.example/1"),
        _R(2, "a2", "https://spam.example/2"),
        _R(3, "a3", "https://spam.example/3"),
        _R(4, "b1", "https://other.example/1"),
    ]
    capped = reputation.cap_per_domain(results, 2)
    assert [r.url for r in capped] == [
        "https://spam.example/1",
        "https://spam.example/2",
        "https://other.example/1",
    ]
    assert [r.rank for r in capped] == [1, 2, 3]


def test_user_block_allow_lists_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(reputation, "USER_CONFIG_DIR", tmp_path)
    reputation._user_cache.clear()

    # Unknown domain is neutral.
    assert reputation.score("https://random.example/x") == 0

    reputation.edit_user_list("block", "random.example")
    assert reputation.score("https://random.example/x") == -10
    # Subdomain also caught.
    assert reputation.score("https://sub.random.example/x") == -10

    reputation.edit_user_list("allow", "goodsite.example")
    assert reputation.score("https://goodsite.example/x") == 3

    # Removal restores neutrality.
    reputation.edit_user_list("block", "random.example", remove=True)
    assert reputation.score("https://random.example/x") == 0

    reputation._user_cache.clear()


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

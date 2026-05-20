"""Tests for the OpenAlex paper-search module.

These are unit tests against fixtures and pure helpers — no live HTTP.
The wire format from OpenAlex is stable, but if it drifts these tests
will catch it before a live search session breaks.
"""
from __future__ import annotations

import csv
import io

import pytest

from websearch.openalex import (
    Article,
    build_filters,
    citation_network,
    reconstruct_abstract,
    to_bibtex,
    to_csv,
    to_markdown,
    to_ris,
    _parse_work,
    _sort_param,
)


# ---------------------------------------------------------------------------
# Fixture: a synthetic OpenAlex "work" record covering all parsed fields
# ---------------------------------------------------------------------------

SAMPLE_WORK = {
    "id": "https://openalex.org/W123",
    "display_name": "An Example Paper",
    "publication_year": 2023,
    "doi": "https://doi.org/10.1000/example",
    "cited_by_count": 42,
    "type": "article",
    "primary_location": {"source": {"display_name": "Nature"}},
    "open_access": {"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
    "authorships": [
        {"author": {"display_name": "Alice Smith"}},
        {"author": {"display_name": "Bob Jones"}},
    ],
    "abstract_inverted_index": {
        "This": [0], "is": [1], "an": [2], "abstract.": [3],
    },
    "topics": [{"display_name": "Genetics"}, {"display_name": "Biology"}],
    "grants": [
        {"funder_display_name": "NIH", "award_id": "R01-12345"},
    ],
    "referenced_works": ["https://openalex.org/W456"],
    "related_works": ["https://openalex.org/W789"],
}


def _sample_articles() -> list[Article]:
    a1 = _parse_work(SAMPLE_WORK)
    a2 = _parse_work({
        **SAMPLE_WORK,
        "id": "https://openalex.org/W456",
        "display_name": "Cited Paper",
        "referenced_works": [],
    })
    return [a1, a2]


# ---------------------------------------------------------------------------
# reconstruct_abstract
# ---------------------------------------------------------------------------

def test_reconstruct_abstract_orders_words_by_position():
    inv = {"world": [1], "hello": [0], "!": [2]}
    assert reconstruct_abstract(inv) == "hello world !"


def test_reconstruct_abstract_handles_repeated_words():
    inv = {"the": [0, 3], "cat": [1], "saw": [2], "dog": [4]}
    assert reconstruct_abstract(inv) == "the cat saw the dog"


def test_reconstruct_abstract_strips_html_tags():
    inv = {"<i>italic</i>": [0], "text": [1]}
    assert reconstruct_abstract(inv) == "italic text"


def test_reconstruct_abstract_empty_returns_blank():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


# ---------------------------------------------------------------------------
# build_filters
# ---------------------------------------------------------------------------

def test_build_filters_year_range():
    assert build_filters(year_min=2020, year_max=2024) == [
        "publication_year:2020-2024"
    ]


def test_build_filters_year_min_only_is_open_ended():
    assert build_filters(year_min=2020) == ["publication_year:2020-"]


def test_build_filters_year_max_only_is_open_ended():
    assert build_filters(year_max=2024) == ["publication_year:-2024"]


def test_build_filters_min_citations_uses_strict_gt():
    # Storing min_citations=10 must include works with exactly 10 — the
    # research_tool implementation does this by emitting `>9`.
    assert build_filters(min_citations=10) == ["cited_by_count:>9"]


def test_build_filters_combined():
    fs = build_filters(
        year_min=2020, year_max=2024,
        min_citations=5, oa_only=True, field="Medicine",
    )
    assert fs == [
        "publication_year:2020-2024",
        "cited_by_count:>4",
        "open_access.is_oa:true",
        "topics.field.display_name.search:Medicine",
    ]


def test_build_filters_empty_when_no_args():
    assert build_filters() == []


# ---------------------------------------------------------------------------
# _sort_param
# ---------------------------------------------------------------------------

def test_sort_param_maps_known_choices():
    assert _sort_param("citations") == "cited_by_count:desc"
    assert _sort_param("newest") == "publication_year:desc"
    assert _sort_param("oldest") == "publication_year:asc"


def test_sort_param_unknown_returns_none():
    assert _sort_param(None) is None
    assert _sort_param("") is None
    assert _sort_param("bogus") is None


# ---------------------------------------------------------------------------
# _parse_work
# ---------------------------------------------------------------------------

def test_parse_work_extracts_all_fields():
    a = _parse_work(SAMPLE_WORK)
    assert a.openalex_id == "https://openalex.org/W123"
    assert a.title == "An Example Paper"
    assert a.authors == ["Alice Smith", "Bob Jones"]
    assert a.year == 2023
    assert a.doi == "https://doi.org/10.1000/example"
    assert a.journal == "Nature"
    assert a.cited_by_count == 42
    assert a.is_oa is True
    assert a.oa_url == "https://example.org/paper.pdf"
    assert a.abstract == "This is an abstract."
    assert a.topics == ["Genetics", "Biology"]
    assert a.funders == [{"name": "NIH", "award_id": "R01-12345"}]
    assert a.referenced_works == ["https://openalex.org/W456"]


def test_parse_work_handles_missing_optional_fields():
    minimal = {"id": "https://openalex.org/W1", "display_name": "Bare"}
    a = _parse_work(minimal)
    assert a.title == "Bare"
    assert a.authors == []
    assert a.journal is None
    assert a.is_oa is False
    assert a.oa_url is None
    assert a.abstract == ""
    assert a.topics == []
    assert a.funders == []


def test_parse_work_falls_back_to_untitled():
    a = _parse_work({"id": "x", "display_name": None, "title": None})
    assert a.title == "Untitled"


# ---------------------------------------------------------------------------
# citation_network
# ---------------------------------------------------------------------------

def test_citation_network_only_emits_internal_edges():
    arts = _sample_articles()
    # arts[0] references arts[1] (W456) and also W999 (not in set)
    arts[0].referenced_works = [
        "https://openalex.org/W456",
        "https://openalex.org/W999",  # external — should be dropped
    ]
    graph = citation_network(arts)
    assert len(graph["nodes"]) == 2
    assert graph["edges"] == [
        {
            "source": "https://openalex.org/W123",
            "target": "https://openalex.org/W456",
        }
    ]


def test_citation_network_empty_input():
    g = citation_network([])
    assert g == {"nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

def test_to_bibtex_emits_synthetic_cite_key_and_doi():
    out = to_bibtex(_sample_articles()[:1])
    assert "@article{Smith2023_1," in out
    assert "title = {An Example Paper}" in out
    assert "author = {Alice Smith and Bob Jones}" in out
    assert "doi = {10.1000/example}" in out


def test_to_bibtex_handles_no_authors():
    a = _parse_work({"id": "x", "display_name": "Lonely paper"})
    out = to_bibtex([a])
    assert "Unknownnd_1" in out  # "Unknown" + "n.d." stripped of '.'


def test_to_csv_round_trips_through_csv_reader():
    out = to_csv(_sample_articles())
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == [
        "Title", "Authors", "Year", "Journal", "DOI", "Citations",
        "Open Access", "Funders", "Abstract",
    ]
    assert rows[1][0] == "An Example Paper"
    assert rows[1][1] == "Alice Smith; Bob Jones"
    assert rows[1][5] == "42"
    assert rows[1][6] == "Yes"
    assert rows[1][7] == "NIH"


def test_to_ris_emits_required_tags():
    out = to_ris(_sample_articles()[:1])
    assert out.startswith("TY  - JOUR")
    assert "TI  - An Example Paper" in out
    assert "AU  - Alice Smith" in out
    assert "AU  - Bob Jones" in out
    assert "PY  - 2023" in out
    assert "JO  - Nature" in out
    assert "DO  - 10.1000/example" in out
    assert out.rstrip().endswith("ER  -")


def test_to_markdown_renders_headers_and_abstract():
    md = to_markdown("test query", _sample_articles()[:1])
    assert 'Research Results: "test query"' in md
    assert "## 1. An Example Paper" in md
    assert "**Citations:** 42" in md
    assert "**Open Access:** Yes" in md
    assert "### Abstract" in md
    assert "This is an abstract." in md

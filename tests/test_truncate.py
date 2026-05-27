"""Tests for smart_truncate: clean boundaries + table preservation."""
from websearch.core import smart_truncate


def test_no_truncation_when_under_budget():
    text = "short body"
    assert smart_truncate(text, 100) == text
    assert smart_truncate(text, None) == text


def test_truncates_at_sentence_boundary():
    text = "First sentence here. " + "Second sentence padding. " * 20
    out = smart_truncate(text, 60)
    body = out.split("\n\n... [truncated")[0]
    # Should not end mid-word; ends at a sentence period.
    assert body.endswith(".")
    assert "truncated at ~60 chars" in out


def test_truncates_at_paragraph_boundary():
    text = "Para one is reasonably long here.\n\n" + "x" * 200
    out = smart_truncate(text, 60)
    body = out.split("\n\n... [truncated")[0]
    assert body == "Para one is reasonably long here."


def test_table_starting_in_budget_kept_whole():
    # Table begins before the cut point -> kept whole even past max_chars.
    pre = "Intro text before the table here.\n"
    table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
    text = pre + table + "\n\ntrailing text " * 30
    out = smart_truncate(text, len(pre) + 12)  # cut lands inside the table
    assert "| 5 | 6 |" in out  # whole table survived
    assert "truncated" in out


def test_table_starting_past_budget_dropped():
    # A huge intro pushes the table entirely past budget -> table excluded,
    # not shown headless.
    pre = "Intro. " * 60  # ~420 chars
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    text = pre + "\n" + table
    out = smart_truncate(text, 120)
    assert "| A | B |" not in out
    assert "truncated" in out

"""Best-effort extraction of a page's published date.

Sources, in priority order:
  1. <meta property="article:published_time"> (OpenGraph)
  2. <meta name="date|publish-date|pubdate|dc.date">
  3. JSON-LD `datePublished` (in any <script type="application/ld+json">)
  4. <time datetime="...">  (first one wins)

Returns a `datetime.date` (no time, no timezone) — we only ever care about
"is this page from on/after a target date?" so the noise of timezones isn't
worth the bugs.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Optional

from bs4 import BeautifulSoup


_DATE_META_NAMES = {"date", "publish-date", "pubdate", "dc.date", "dc.date.issued"}
_DATE_PROPERTIES = {"article:published_time", "article:published", "og:published_time"}


def _coerce(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    # Trim Z or timezone offset before fromisoformat in older Pythons
    s = re.sub(r"Z$", "+00:00", s)
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass
    # Just YYYY-MM-DD up front?
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    return None


def _from_jsonld(soup: BeautifulSoup) -> Optional[date]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        if "datePublished" not in raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            # Some sites pack multiple JSON objects or have stray HTML — be lenient
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw)
            if m:
                d = _coerce(m.group(1))
                if d:
                    return d
            continue
        candidates: list = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            v = item.get("datePublished") or item.get("dateCreated")
            if v:
                d = _coerce(str(v))
                if d:
                    return d
    return None


def extract_published(html: str) -> Optional[date]:
    """Return the page's published date, or None if not found."""
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    # 1. OpenGraph / article:* meta tags
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or "").lower()
        if prop in _DATE_PROPERTIES:
            d = _coerce(meta.get("content", ""))
            if d:
                return d

    # 2. Generic <meta name="date">
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or "").lower()
        if name in _DATE_META_NAMES:
            d = _coerce(meta.get("content", ""))
            if d:
                return d

    # 3. JSON-LD
    d = _from_jsonld(soup)
    if d:
        return d

    # 4. First <time datetime="...">
    t = soup.find("time", attrs={"datetime": True})
    if t:
        d = _coerce(t.get("datetime", ""))
        if d:
            return d

    return None


def parse_since(since: str) -> Optional[date]:
    """Coerce a --since value to a cutoff date.

    Accepts: YYYY-MM-DD, YYYY, and the bucket shorthands (d/w/m/y) which
    expand against today.
    """
    if not since:
        return None
    s = since.strip().lower()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None
    if re.match(r"^\d{4}$", s):
        return date(int(s), 1, 1)
    from datetime import timedelta
    today = date.today()
    if s == "d":
        return today - timedelta(days=1)
    if s == "w":
        return today - timedelta(days=7)
    if s == "m":
        return today - timedelta(days=30)
    if s == "y":
        return today - timedelta(days=365)
    m = re.match(r"^(\d+)([dwmy])$", s)
    if m:
        n, unit = int(m[1]), m[2]
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * n
        return today - timedelta(days=days)
    return None

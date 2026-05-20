"""GitHub API helpers for fast user/repo/commit reconnaissance.

The base GitHub API endpoints return huge JSON documents (4-6 KB per user,
even more per repo) that are 90% URL boilerplate. This module hides the
boilerplate and exposes the fields that actually matter for OSINT-style
investigation: user summary, repo names + descriptions, commit author
(name, email) tuples deduped across the whole account.

No auth required — uses the public REST API rate limit (60 req/hr per IP).
For heavy use, set GITHUB_TOKEN in the environment and the client will
attach it as a bearer; the rate limit jumps to 5000/hr.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .core import fetch_direct
from .cache import Cache


API = "https://api.github.com"


def _auth_headers() -> dict[str, str]:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _get_json(
    url: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> tuple[int, object]:
    """Fetch a URL and parse JSON. Returns (status, parsed-or-None)."""
    r = fetch_direct(url, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh)
    if r.status >= 400 or not r.text:
        return r.status, None
    try:
        return r.status, json.loads(r.text)
    except (json.JSONDecodeError, ValueError):
        return r.status, None


def user_summary(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> dict:
    """Return the user fields that matter, not the URL forest."""
    status, data = _get_json(
        f"{API}/users/{username}", timeout=timeout, proxy=proxy, cache=cache, refresh=refresh
    )
    if not isinstance(data, dict):
        return {"username": username, "status": status, "error": "user not found or rate-limited"}
    keep = (
        "login id name company blog location email bio twitter_username "
        "public_repos public_gists followers following created_at updated_at"
    ).split()
    return {k: data.get(k) for k in keep}


def repos(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[dict]:
    """Compact per-repo summary: name, description, language, dates, fork-of."""
    out: list[dict] = []
    page = 1
    while True:
        status, data = _get_json(
            f"{API}/users/{username}/repos?per_page=100&page={page}",
            timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
        )
        if not isinstance(data, list) or not data:
            break
        for r in data:
            parent = (r.get("parent") or {}).get("full_name") if r.get("fork") else None
            out.append({
                "name": r.get("name"),
                "full_name": r.get("full_name"),
                "description": r.get("description"),
                "language": r.get("language"),
                "fork": r.get("fork"),
                "parent": parent,
                "stargazers_count": r.get("stargazers_count"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
                "pushed_at": r.get("pushed_at"),
                "html_url": r.get("html_url"),
                "archived": r.get("archived"),
            })
        if len(data) < 100:
            break
        page += 1
    return out


def repo_commit_authors(
    full_name: str,
    per_page: int = 100,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[tuple[str, str, str]]:
    """Walk a repo's recent commits and return (author_name, author_email, sha)."""
    status, data = _get_json(
        f"{API}/repos/{full_name}/commits?per_page={per_page}",
        timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
    )
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str, str]] = []
    for c in data:
        commit = c.get("commit") or {}
        author = commit.get("author") or {}
        name = (author.get("name") or "").strip()
        email = (author.get("email") or "").strip()
        sha = c.get("sha") or ""
        if name or email:
            out.append((name, email, sha))
    return out


def emails(
    username: str,
    parallel: int = 6,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> dict:
    """Walk all public repos and dedupe commit-author (name, email) pairs.

    Returns a structured dict with per-pair commit counts and per-repo hits.
    This is the single most valuable handle for "what real name/email is this
    GitHub account using?" — author emails are visible on every public commit.
    """
    rs = repos(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh)
    pair_counter: Counter[tuple[str, str]] = Counter()
    per_repo: dict[str, list[tuple[str, str]]] = {}

    def _worker(r):
        return r["full_name"], repo_commit_authors(
            r["full_name"], timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
        )

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        for fut in as_completed([ex.submit(_worker, r) for r in rs]):
            try:
                full_name, authors = fut.result()
            except Exception:
                continue
            seen: set[tuple[str, str]] = set()
            for name, email, _sha in authors:
                pair = (name, email)
                pair_counter[pair] += 1
                seen.add(pair)
            per_repo[full_name] = sorted(seen)

    pairs = [
        {"name": n, "email": e, "commit_count": c}
        for (n, e), c in pair_counter.most_common()
    ]
    return {
        "username": username,
        "repos_scanned": len(rs),
        "unique_author_pairs": pairs,
        "per_repo": per_repo,
    }


def events(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[dict]:
    """Recent public activity, compacted."""
    status, data = _get_json(
        f"{API}/users/{username}/events/public?per_page=30",
        timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
    )
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for e in data:
        payload = e.get("payload") or {}
        commits = payload.get("commits") or []
        out.append({
            "type": e.get("type"),
            "repo": (e.get("repo") or {}).get("name"),
            "created_at": e.get("created_at"),
            "ref": payload.get("ref"),
            "commit_messages": [c.get("message", "").splitlines()[0][:120] for c in commits],
        })
    return out


def gists(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[dict]:
    status, data = _get_json(
        f"{API}/users/{username}/gists?per_page=100",
        timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
    )
    if not isinstance(data, list):
        return []
    return [
        {
            "id": g.get("id"),
            "description": g.get("description"),
            "files": list((g.get("files") or {}).keys()),
            "created_at": g.get("created_at"),
            "html_url": g.get("html_url"),
            "public": g.get("public"),
        }
        for g in data
    ]


def starred(
    username: str,
    limit: int = 30,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[dict]:
    status, data = _get_json(
        f"{API}/users/{username}/starred?per_page={limit}",
        timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
    )
    if not isinstance(data, list):
        return []
    return [
        {"full_name": r.get("full_name"), "description": r.get("description"), "language": r.get("language")}
        for r in data
    ]


def orgs(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> list[str]:
    status, data = _get_json(
        f"{API}/users/{username}/orgs",
        timeout=timeout, proxy=proxy, cache=cache, refresh=refresh,
    )
    if not isinstance(data, list):
        return []
    return [o.get("login") for o in data if o.get("login")]


def full_report(
    username: str,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    refresh: bool = False,
) -> dict:
    """One-shot deep summary: everything the public API exposes about a user."""
    summary = user_summary(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh)
    if summary.get("error"):
        return summary
    return {
        "user": summary,
        "repos": repos(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
        "emails": emails(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
        "events": events(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
        "gists": gists(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
        "starred": starred(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
        "orgs": orgs(username, timeout=timeout, proxy=proxy, cache=cache, refresh=refresh),
    }

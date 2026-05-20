"""Username enumeration across common platforms.

Sherlock-style probe: take a username, hit a curated list of platforms,
report which ones have an account. Detection is per-site because some
return HTTP 404 (cheap), some return 200 with a "user not found" page
that needs content matching.

Kept small and opinionated — sites that frequently block or false-positive
are excluded rather than added with creative heuristics. Grow from real use.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

from .core import fetch_direct
from .cache import Cache


@dataclass
class Probe:
    name: str
    url_template: str
    # If set, presence of this substring in body == "no such user" (status was 200).
    # Without it, only HTTP status is consulted.
    absence_marker: Optional[str] = None
    # If set, MUST appear in body to count as found (defends against generic 200s).
    presence_marker: Optional[str] = None


# Curated list. Each entry has been hand-checked for reliable signal.
# Order is alphabetical by name for predictable output.
#
# Several common platforms (PyPI, Substack, Twitch, Mastodon HTML view, Patreon)
# return HTTP 200 with a JS shell that doesn't include the username, making the
# "does this user exist?" question impossible to answer from HTML alone. We
# either use an API/Webfinger endpoint instead, or drop the probe entirely.
PROBES: list[Probe] = [
    Probe("bitbucket",      "https://bitbucket.org/{u}/",                                absence_marker="Page not found"),
    Probe("codeberg",       "https://codeberg.org/{u}",                                  absence_marker="User does not exist"),
    Probe("devto",          "https://dev.to/{u}",                                        presence_marker="property=\"og:url\" content=\"https://dev.to/"),
    Probe("github",         "https://api.github.com/users/{u}",                          presence_marker="\"login\""),
    Probe("gitlab",         "https://gitlab.com/api/v4/users?username={u}",              presence_marker="\"username\""),
    Probe("hackernews",     "https://hn.algolia.com/api/v1/users/{u}",                   presence_marker="\"username\""),
    Probe("huggingface",    "https://huggingface.co/{u}",                                absence_marker="Sorry, we can"),
    Probe("itch",           "https://{u}.itch.io",                                       absence_marker="Nothing here"),
    Probe("keybase",        "https://keybase.io/_/api/1.0/user/lookup.json?usernames={u}", presence_marker="\"basics\""),
    Probe("lobsters",       "https://lobste.rs/~{u}.json",                               presence_marker="\"username\""),
    Probe("mastodon_social","https://mastodon.social/.well-known/webfinger?resource=acct:{u}@mastodon.social", presence_marker="\"subject\""),
    Probe("medium",         "https://medium.com/@{u}",                                   presence_marker="og:url\" content=\"https://medium.com/@"),
    Probe("pastebin",       "https://pastebin.com/u/{u}",                                presence_marker="<title>{u}'s "),
    Probe("reddit",         "https://www.reddit.com/user/{u}/about.json",                presence_marker="\"name\""),
    Probe("soundcloud",     "https://soundcloud.com/{u}",                                presence_marker="og:url\" content=\"https://soundcloud.com/"),
    Probe("steam",          "https://steamcommunity.com/id/{u}",                         absence_marker="the specified profile could not be found"),
    Probe("vimeo",          "https://vimeo.com/{u}",                                     presence_marker="og:type\" content=\"profile"),
    Probe("youtube",        "https://www.youtube.com/@{u}",                              presence_marker="@{u}\""),
]


@dataclass
class ProbeResult:
    site: str
    url: str
    status: int
    found: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _decide(probe: Probe, status: int, body: str) -> tuple[bool, str]:
    """Return (found, note). Note explains the decision."""
    if status == 0:
        return False, "network error"
    if status == 404:
        return False, "HTTP 404"
    if status == 429:
        return False, "rate limited (HTTP 429) — try again later"
    if status >= 400:
        # Some sites use 403/451 for "exists but blocked" — surface but don't claim found
        return False, f"HTTP {status}"
    # status in [200, 399]
    low = (body or "")[:50000].lower()
    if probe.absence_marker and probe.absence_marker.lower() in low:
        return False, f"absence-marker matched: {probe.absence_marker!r}"
    if probe.presence_marker and probe.presence_marker.lower() not in low:
        return False, f"presence-marker missing: {probe.presence_marker!r}"
    return True, f"HTTP {status}"


def probe_one(username: str, probe: Probe, timeout: int, proxy: Optional[str], cache: Optional[Cache]) -> ProbeResult:
    url = probe.url_template.format(u=username)
    # Substitute {u} into presence/absence markers too — some probes need the
    # literal username in the matched substring (e.g., "@5speeddeasil" on YouTube).
    pres = probe.presence_marker.replace("{u}", username) if probe.presence_marker else None
    absc = probe.absence_marker.replace("{u}", username) if probe.absence_marker else None
    effective = Probe(
        name=probe.name,
        url_template=probe.url_template,
        presence_marker=pres,
        absence_marker=absc,
    )
    r = fetch_direct(url, timeout=timeout, proxy=proxy, cache=cache, refresh=True)
    found, note = _decide(effective, r.status, r.text)
    return ProbeResult(site=probe.name, url=url, status=r.status, found=found, note=note)


def enumerate_username(
    username: str,
    sites: Optional[list[str]] = None,
    timeout: int = 15,
    proxy: Optional[str] = None,
    cache: Optional[Cache] = None,
    parallel: int = 8,
) -> list[ProbeResult]:
    """Probe many platforms for `username`. Returns one ProbeResult per site."""
    selected: list[Probe]
    if sites:
        wanted = {s.lower() for s in sites}
        selected = [p for p in PROBES if p.name.lower() in wanted]
        if not selected:
            raise ValueError(
                f"no probes matched {sorted(wanted)}; available: "
                f"{', '.join(sorted(p.name for p in PROBES))}"
            )
    else:
        selected = list(PROBES)

    results: list[Optional[ProbeResult]] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(probe_one, username, p, timeout, proxy, cache): i for i, p in enumerate(selected)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = ProbeResult(
                    site=selected[i].name,
                    url=selected[i].url_template.format(u=username),
                    status=0,
                    found=False,
                    note=f"error: {e}",
                )
    return [r for r in results if r is not None]

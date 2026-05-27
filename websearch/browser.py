"""Browser-rendered fetch via camoufox.

Used for sites that hard-fail under plain `requests` — usually
Cloudflare/Datadome challenge pages, JS-heavy SPAs, or login walls
that require a full headless browser to render. Opt-in via
`fetch --via browser` because every call spins up ~200MB of Chromium-
equivalent and takes 2-10s.

Returns a `FetchResult` shape compatible with the rest of the fetch
pipeline so downstream extraction / caching / smart_truncate all work
without special-casing the source.
"""
from __future__ import annotations

from typing import Optional

from .cache import default as default_cache


def have_camoufox() -> bool:
    try:
        import camoufox  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_browser(
    url: str,
    timeout: int = 30,
    proxy: Optional[str] = None,
    cache=None,
    max_age: Optional[float] = None,
    refresh: bool = False,
    verify: bool = True,
    use_whisper: bool = False,
):
    """Fetch a URL via headless camoufox and return a FetchResult.

    Honors the same cache contract as fetch_direct so repeated browser
    calls aren't re-paying the launch cost. `verify` is accepted for
    signature parity but ignored (camoufox handles TLS internally).
    `use_whisper` is also ignored — browser fetch is for HTML, not video.
    """
    # Imported here so the rest of the module doesn't pay for the
    # core.FetchResult import circularly.
    from .core import FetchResult

    cache = cache if cache is not None else default_cache()

    # Cache lookup: skip the browser launch entirely on cache hit.
    if cache and not refresh:
        hit = cache.get("GET", url, max_age=max_age)
        if hit is not None:
            return FetchResult(
                url=hit.url,
                final_url=hit.final_url,
                status=hit.status,
                content_type=hit.content_type,
                text=hit.body,
                via="cache",
                error=None,
                from_cache=True,
                cache_age_seconds=hit.age_seconds,
            )

    if not have_camoufox():
        return FetchResult(
            url, url, 0, "", "", "browser",
            "camoufox not installed — `pipx inject websearch camoufox` "
            "and run `camoufox fetch` once to download Firefox binaries",
        )

    try:
        from camoufox.sync_api import Camoufox  # type: ignore
    except ImportError as e:
        return FetchResult(
            url, url, 0, "", "", "browser",
            f"camoufox import failed: {e}",
        )

    # Build the Camoufox config. Proxy is optional; camoufox does
    # browser-level anti-fingerprinting on its own.
    kw: dict = {"headless": True}
    if proxy:
        kw["proxy"] = {"server": proxy}

    try:
        with Camoufox(**kw) as browser:
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000)
            # networkidle is more reliable than load for JS-heavy pages,
            # but cap the wait so a never-quiet page can't hang us.
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass  # not all pages reach networkidle; the load is fine
            html = page.content()
            final_url = page.url
            status = 200  # camoufox doesn't expose the HTTP status of the
            # outer navigation cleanly; if we got HTML back, treat as 200.
    except Exception as e:  # noqa: BLE001
        return FetchResult(
            url, url, 0, "", "", "browser",
            f"camoufox fetch failed: {type(e).__name__}: {e}",
        )

    result = FetchResult(
        url=url,
        final_url=final_url,
        status=status,
        content_type="text/html; rendered",
        text=html,
        via="browser",
        error=None,
    )
    if cache and result.text:
        cache.put("GET", url, final_url, status,
                  result.content_type, result.text)
    return result

# websearch

A web search, fetch, and research CLI built to replace AI assistants' built-in `WebFetch` / `WebSearch` tools, which get blocked by anti-bot protection on a lot of sites. Designed for AI agents and humans alike — predictable flags, compact output, smart fallbacks.

## Why

Built-in agent web tools fail in ways that look like success: they 403, get captcha walls, or hit selector drift on Bing/DDG and return empty results without saying so. This tool is opinionated about that:

- Routes through a SOCKS5h DoH stealth proxy so anti-bot pages don't catch it
- Detects block pages (`BLOCK_PATTERNS`) and reports them honestly instead of pretending the page was empty
- Falls back to the Wayback Machine when a live fetch fails, with a `wayback_status=N` line on stderr so you know what happened
- Filters out SEO farms via a reputation module with a blocklist, trusted allowlists by category, and TLD heuristics

## Install

```bash
cd ~/projects/websearch
pipx install -e .
```

After local edits: `pipx install -e . --force`.

The console script lands at `~/.local/bin/websearch`.

## Proxy

Set `WEBSEARCH_PROXY=socks5h://127.0.0.1:1080` (or pass `--proxy`). This is the tool's core advantage — don't run it bare against sites with anti-bot protection.

## Subcommands

```
search    Search the web (DDG, Bing, or auto-fallback)
fetch     Fetch one or more URLs with smart Wayback fallback
text      Shortcut: fetch + extract clean text
research  Multi-query search + fetch top + compact markdown
cite      Generate a markdown citation block from URLs
download  Stream a URL to disk (binary-safe, with progress)
cache     Manage the disk cache (stats / clear)
selftest  Smoke test of search and fetch parsers
```

## Common usage

Research preset — give it a question, get a clean markdown report:

```bash
websearch research "how does QUIC handle congestion control" \
  -q "QUIC BBR vs CUBIC" \
  --depth 5 \
  --max-chars 2000 \
  --exclude reddit.com,x.com,twitter.com
```

Trusted-source-only research:

```bash
websearch research "question" --trust high --prefer academic --depth 5
```

Single-page article extraction:

```bash
websearch fetch "https://example.com/article" --mode article --max-chars 3000
```

Search with grep on fetched bodies:

```bash
websearch search -q "term" --fetch-top 5 --grep "regex" --grep-context 2 --compact
```

## Key flags

### `search`
- `-q QUERY` (repeatable) — multi-query parallel with dedupe
- `--fetch-top N` — search + fetch in one call
- `-e ddg|bing|auto` — engine (default: `auto`, DDG → Bing fallback)
- `--exclude domain1,domain2` (repeatable, suffix-match)
- `--since YYYY-MM-DD|YYYY|d|w|m|y` — date filter (DDG `df`; Bing ignores)
- `--trust any|medium|high` — drops SEO farms (`medium`) or restricts to `.gov`/`.edu`/journals/major news (`high`)
- `--prefer academic|gov|news|reference` — boosts category in ranking
- `--compact` / `-c` — markdown output instead of JSON
- `--grep PATTERN --grep-context N` — filter fetched bodies to matching lines

### `fetch`
- `--mode article|tables|raw` — extraction mode (trafilatura → BS4 fallback)
- `--max-chars N`, `--skip-chars N`, `--refresh` (bypass cache)
- `--grep PATTERN --grep-context N`
- `--insecure` / `-k` for TLS issues
- `--compact` / `-c` for batch fetches

### `research`
Defaults: `--trust medium`, `--max-chars 2500`, compact markdown always on. Multi-query → dedupe → fetch top N → one document.

## Caching

SQLite WAL cache at `~/.cache/websearch/cache.db`, thread-safe. `websearch cache stats` and `websearch cache clear` to manage.

## PDFs

Detected via content-type and magic bytes; extracted with system `pdftotext` (poppler). Install with `brew install poppler` if needed.

## Layout

```
websearch/
├── cli.py          # argparse, subcommand dispatch
├── core.py         # search/fetch engines, Wayback fallback
├── cache.py        # SQLite WAL cache
├── pdfx.py         # PDF detection + pdftotext extraction
├── reputation.py   # blocklist, trusted allowlists, TLD trust heuristics
└── __init__.py
```

## License

Personal project, not currently licensed for redistribution.

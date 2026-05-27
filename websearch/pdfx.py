"""PDF text extraction via system pdftotext (poppler).

We deliberately do NOT add a Python PDF library as a dependency — pdftotext
from poppler is faster, more accurate, and the user already has it for the
PDF work that started this conversation. If poppler isn't installed, we
return a clear marker so callers don't get garbage from HTML parsers.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from typing import Optional


def have_pdftotext() -> bool:
    return shutil.which("pdftotext") is not None


def looks_like_pdf(content_type: str, body_bytes: Optional[bytes] = None, url: str = "") -> bool:
    """Heuristic: is this content a PDF?

    Checks (in order): explicit content-type, magic bytes, URL extension.
    """
    if content_type and "application/pdf" in content_type.lower():
        return True
    if body_bytes and body_bytes[:5] == b"%PDF-":
        return True
    if url and url.lower().split("?")[0].endswith(".pdf"):
        return True
    return False


import re as _re


_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "background"),
    "methods": ("methods", "methodology", "materials and methods", "methods and materials"),
    "results": ("results", "findings"),
    "discussion": ("discussion",),
    "conclusion": ("conclusion", "conclusions", "concluding remarks"),
    "references": ("references", "bibliography", "works cited"),
}

# A line that looks like a section heading: short, mostly alphabetic, may
# be numbered ("1.", "1.", "I.", "II.") or plain.
_HEADING_LINE_RE = _re.compile(
    r"^\s*(?:(?:\d+\.\d*|[IVXLC]+\.)\s+)?([A-Z][A-Za-z &/\-]{2,60})\s*$",
)


def _normalize_aliases(sections: list[str]) -> set[str]:
    wanted: set[str] = set()
    for s in sections:
        key = s.strip().lower()
        if not key:
            continue
        if key in _SECTION_ALIASES:
            wanted.update(_SECTION_ALIASES[key])
        else:
            wanted.add(key)
    return wanted


def extract_sections(text: str, sections: list[str]) -> str:
    """From plaintext (typically pdftotext output), keep only named sections.

    Heading detection is intentionally generous: a line is a heading if it's
    short, starts capitalized, and matches a known section name (or alias).
    Returns the original text if no headings are detected at all.
    """
    if not sections or not text:
        return text
    wanted = _normalize_aliases(sections)
    if not wanted:
        return text

    lines = text.splitlines()
    heading_idx: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        m = _HEADING_LINE_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip().lower()
        if title in wanted or any(title == w or title.startswith(w) for w in wanted):
            heading_idx.append((i, title))
        else:
            # Track any heading-shaped line so we know where the wanted one ends
            heading_idx.append((i, "__other__"))

    if not heading_idx or all(t == "__other__" for _, t in heading_idx):
        return text  # no recognizable structure — return whole document

    parts: list[str] = []
    for pos, (i, title) in enumerate(heading_idx):
        if title == "__other__":
            continue
        end = heading_idx[pos + 1][0] if pos + 1 < len(heading_idx) else len(lines)
        chunk = "\n".join(lines[i:end]).strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts) if parts else text


def extract(body: bytes, max_pages: Optional[int] = None) -> str:
    """Extract text from a PDF byte stream.

    Returns extracted text on success, or a clear marker string on failure
    (so the caller can distinguish "binary I can't read" from "real text").
    """
    if not have_pdftotext():
        return (
            f"[binary PDF, {len(body)} bytes — install poppler "
            f"(`brew install poppler` / `apt install poppler-utils`) "
            f"to extract text]"
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(body)
        tmp.flush()
        cmd = ["pdftotext", "-layout", "-nopgbrk"]
        if max_pages:
            cmd += ["-l", str(max_pages)]
        cmd += [tmp.name, "-"]
        try:
            r = subprocess.run(
                cmd, capture_output=True, timeout=60, check=False
            )
        except subprocess.TimeoutExpired:
            return f"[PDF extraction timed out after 60s, {len(body)} bytes]"
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            return f"[pdftotext failed (rc={r.returncode}): {err[:200]}]"
        return _collapse_pdf_whitespace(r.stdout.decode("utf-8", errors="replace"))


# pdftotext -layout preserves column padding so a centered abstract on a
# two-column paper arrives with 40-80 leading spaces per line. That used to
# eat a large fraction of any --max-chars budget without adding information.
# Strip per-line leading runs of 4+ spaces, then collapse runs of inline
# spaces to a single space. Single/double leading spaces (e.g. indented
# code, tables) are left alone.
_PDF_LEADING_RUN_RE = _re.compile(r"^[ \t]{4,}", _re.MULTILINE)
_PDF_INLINE_RUN_RE = _re.compile(r"[ \t]{3,}")


def _collapse_pdf_whitespace(text: str) -> str:
    if not text:
        return text
    text = _PDF_LEADING_RUN_RE.sub("", text)
    text = _PDF_INLINE_RUN_RE.sub(" ", text)
    return text

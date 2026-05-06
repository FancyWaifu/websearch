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
        return r.stdout.decode("utf-8", errors="replace")

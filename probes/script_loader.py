"""
Script loader: extracts text from PDF or plain text files.

Uses pdfplumber for PDF extraction (pure Python, no system deps).
Falls back to reading as plain text if pdfplumber is unavailable or
the file is not a PDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def load_script(path: str | Path) -> str:
    """
    Extract full text from a PDF or text file.

    Returns the extracted text as a single string.
    Raises FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(path)
    else:
        # Plain text / markdown / any text-readable format
        text = path.read_text(encoding="utf-8", errors="replace")

    print(f"[ScriptLoader] Loaded: {path.name} ({suffix}, {len(text):,} chars)")
    return text


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction. "
            "Install it with: pip install pdfplumber"
        )

    text_parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)


def segment_script(text: str, words_per_segment: int = 20) -> List[str]:
    """
    Split script text into segments of ~20 words each.

    Each segment becomes a talking-point checkpoint for the script tracker.
    """
    if not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    # Chunk into groups of N words
    segments = []
    for i in range(0, len(words), words_per_segment):
        chunk = " ".join(words[i : i + words_per_segment])
        segments.append(chunk)

    # Debug: print all segments
    print(f"[ScriptLoader] Segmented into {len(segments)} chunks ({words_per_segment} words each, {len(words)} total words):")
    for i, seg in enumerate(segments):
        preview = seg[:100]
        print(f"  [{i+1}/{len(segments)}] \"{preview}{'...' if len(seg) > 100 else ''}\"")

    return segments

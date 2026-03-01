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
        return _extract_pdf(path)
    else:
        # Plain text / markdown / any text-readable format
        return path.read_text(encoding="utf-8", errors="replace")


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


def segment_script(text: str, min_length: int = 20) -> List[str]:
    """
    Split extracted script text into logical segments (talking points).

    Segments are split by:
    1. Double newlines (paragraph breaks)
    2. Numbered list items (e.g., "1.", "2.")
    3. Bullet points

    Segments shorter than min_length are merged with the next segment.
    """
    if not text.strip():
        return []

    # Split on double newlines or numbered/bulleted list items
    raw_segments = re.split(r"\n\s*\n|\n(?=\d+[.)]\s)|\n(?=[-•]\s)", text)

    # Clean up whitespace
    segments = []
    for seg in raw_segments:
        cleaned = " ".join(seg.split())  # collapse internal whitespace
        if cleaned:
            segments.append(cleaned)

    # Merge short segments with the next one
    merged: list[str] = []
    buffer = ""
    for seg in segments:
        if buffer:
            buffer = buffer + " " + seg
        else:
            buffer = seg

        if len(buffer) >= min_length:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged

"""
Load knowledge probes from a JSON file.

Probes are a list of strings (prompts or keywords) shown on the HUD when
a cognitive stall is detected. Edit assets/probes.json to customize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_probes(path: str | Path) -> List[str]:
    """
    Load probe strings from a JSON file (array of strings).
    Returns empty list if file is missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        return []
    except (json.JSONDecodeError, OSError):
        return []

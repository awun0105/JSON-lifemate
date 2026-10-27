"""Canonical JSON and text truncation (SPEC 4.0.1)."""

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialise deterministically: sorted keys, compact, non-ASCII kept."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def truncate(text: str, max_chars: int, suffix: str = "…(đã cắt)") -> str:
    """Cut `text` to at most `max_chars` characters plus `suffix` when it is longer."""
    return text if len(text) <= max_chars else text[:max_chars] + suffix

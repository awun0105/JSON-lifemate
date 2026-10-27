"""Near-duplicate grouping of request texts, shared by the split script and the SFT data builder.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _norm(text: str) -> str:
    """NFC and collapsed whitespace (the team lead's normalize_input_text)."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def cluster_near_duplicates(items: Sequence[Mapping[str, Any]], threshold: float) -> list[list[str]]:
    """Group ids whose audio_text is identical or at least `threshold` similar (difflib ratio), so near-duplicate
    requests land in the same fold. Greedy, in id order: deterministic."""
    texts = [(it["id"], _norm(it.get("audio_text") or "")) for it in sorted(items, key=lambda x: x["id"])]
    owner: dict[str, int] = {}
    clusters: list[list[str]] = []
    for i, (sid, t1) in enumerate(texts):
        if sid in owner:
            continue
        owner[sid] = len(clusters)
        group = [sid]
        for sid2, t2 in texts[i + 1 :]:
            if sid2 in owner:
                continue
            same = t1 == t2
            if not same and len(t1) > 10 and len(t2) > 10 and abs(len(t1) - len(t2)) < 15:
                m = difflib.SequenceMatcher(None, t1, t2)
                same = m.quick_ratio() >= threshold and m.ratio() >= threshold
            if same:
                owner[sid2] = len(clusters)
                group.append(sid2)
        clusters.append(group)
    return clusters

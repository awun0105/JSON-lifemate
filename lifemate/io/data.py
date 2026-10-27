"""Load competition samples from `<root>/<split>/<split>.jsonl` (SPEC 4.2)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lifemate.core.types import Sample, Turn


def _turn(raw: dict[str, Any]) -> Turn:
    role = raw["role"]
    if role not in ("user", "assistant"):
        raise ValueError(f"unknown turn role: {role!r}")
    return Turn(
        role=role,
        audio=raw.get("audio"),
        audio_text=raw.get("audio_text"),
        text=raw.get("text"),
        tool_calls=tuple(raw.get("tool_calls") or ()),
        tool_results=tuple(raw.get("tool_results") or ()),
    )


def _sample(raw: dict[str, Any], split: str) -> Sample:
    return Sample(
        id=raw["id"],
        split=split,
        audio=raw["audio"],
        audio_text=raw.get("audio_text"),
        context=raw["context"],
        history=tuple(_turn(t) for t in raw.get("history") or ()),
        label=raw.get("label"),
    )


def load_split(root: Path, split: str, ids: Sequence[str] | None = None) -> list[Sample]:
    """Read every sample of `split` in file order; keep only `ids` when given (unknown ids raise ValueError)."""
    path = Path(root) / split / f"{split}.jsonl"
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(_sample(json.loads(line), split))
    if ids is None:
        return samples
    wanted = set(ids)
    unknown = sorted(wanted - {s.id for s in samples})
    if unknown:
        raise ValueError(f"unknown ids in split {split!r}: {unknown}")
    return [s for s in samples if s.id in wanted]


def read_ids(path: Path) -> list[str]:
    """One id per line; blank lines ignored."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]

"""Đọc mẫu dữ liệu cuộc thi từ `<root>/<split>/<split>.jsonl` (SPEC 4.2).

Nội dung file: `load_split` đọc một split thành danh sách `Sample`, `read_ids` đọc file danh sách id.

Cách xây dựng: mỗi dòng JSON được đổi thành dataclass bất biến (`Sample`, `Turn`) ngay khi đọc, nên các tầng
sau không phải làm việc với dict thô. Khóa bắt buộc thiếu thì báo KeyError ngay; khóa tùy chọn (`audio_text`,
`history`, `label`...) thiếu thì nhận None hoặc tuple rỗng. Thứ tự mẫu giữ đúng thứ tự trong file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lifemate.core.types import Sample, Turn


def _turn(raw: dict[str, Any]) -> Turn:
    """Đổi một lượt hội thoại thô thành `Turn`; báo ValueError khi role không phải user hoặc assistant."""
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
    """Đổi một dòng JSON thô thành `Sample` thuộc `split`."""
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
    """Đọc mọi mẫu của `split` theo thứ tự trong file; khi có `ids` thì chỉ giữ các id đó.

    Tham số:
        root: thư mục dữ liệu.
        split: tên split.
        ids: danh sách id cần giữ; None là giữ tất cả.

    Báo ValueError khi `ids` có id không thuộc split. Kết quả vẫn theo thứ tự file, không theo thứ tự `ids`.
    """
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
    """Đọc file mỗi dòng một id; bỏ qua dòng trống và khoảng trắng hai đầu."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]

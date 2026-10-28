"""Ghi TOML tối giản cho các config sinh ra lúc chạy (bản sao configs/v0.toml với vài khóa đã sửa).

Nội dung: `dump_toml` xuất các bảng cấp cao nhất gồm giá trị đơn, list và bảng inline, đủ cho configs/v0.toml.
Chuyển từ lifemate/train/llm_pipeline.py sang `core` để cả `eval` (lifemate.eval.asr_tune) lẫn `train` cùng dùng
mà không vi phạm quy tắc import giữa các lớp (eval và train không import lẫn nhau).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _dump_toml_value(v: Any) -> str:
    """Chuyển một giá trị Python (bool, số, chuỗi, list, mapping) thành chuỗi TOML; kiểu khác thì ném TypeError."""
    # Xét bool trước int vì bool là lớp con của int.
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_dump_toml_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{k} = {_dump_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"unsupported TOML value: {v!r}")


def dump_toml(data: Mapping[str, Mapping[str, Any]]) -> str:
    """Xuất các bảng cấp cao nhất gồm giá trị đơn, list và bảng inline (đủ cho configs/v0.toml)."""
    out: list[str] = []
    for section, table in data.items():
        out.append(f"[{section}]")
        out += [f"{k} = {_dump_toml_value(v)}" for k, v in table.items()]
        out.append("")
    return "\n".join(out)

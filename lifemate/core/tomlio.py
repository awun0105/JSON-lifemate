"""Ghi file TOML đơn giản cho các config sinh ra lúc chạy (bản sao `configs/v0.toml` đã sửa vài khoá).

Nội dung chính:
- `dump_toml`: xuất các bảng cấp cao nhất, mỗi bảng gồm giá trị đơn, list và bảng inline.

Lưu ý:
- Chỉ hỗ trợ đúng những kiểu mà `configs/v0.toml` cần, không phải bộ ghi TOML đầy đủ.
- Module này từng nằm trong `lifemate/train/llm_pipeline.py`, sau đó chuyển sang `core` để cả `eval`
  (`lifemate.eval.asr_tune`) lẫn `train` cùng dùng, vì `eval` và `train` không được import lẫn nhau.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _dump_toml_value(v: Any) -> str:
    """Chuyển một giá trị Python thành chuỗi TOML.

    Tham số:
        v: bool, số, chuỗi, list hoặc mapping; list và mapping có thể lồng nhau.

    Trả về:
        Chuỗi TOML của `v`; mapping thành bảng inline `{k = v, ...}`.

    Lỗi:
        TypeError: khi `v` thuộc kiểu khác các kiểu trên.
    """
    # Phải xét bool trước int: bool là lớp con của int, xét sau thì `True` sẽ thành `1`.
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        # Chuỗi JSON có dấu nháy kép và cách thoát ký tự trùng với chuỗi cơ bản của TOML.
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_dump_toml_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{k} = {_dump_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"unsupported TOML value: {v!r}")


def dump_toml(data: Mapping[str, Mapping[str, Any]]) -> str:
    """Xuất `data` thành văn bản TOML, mỗi khoá cấp cao nhất là một bảng `[section]`.

    Tham số:
        data: mapping từ tên bảng tới các cặp khoá - giá trị của bảng đó.

    Trả về:
        Văn bản TOML, các bảng cách nhau một dòng trống.
    """
    out: list[str] = []
    for section, table in data.items():
        out.append(f"[{section}]")
        out += [f"{k} = {_dump_toml_value(v)}" for k, v in table.items()]
        out.append("")
    return "\n".join(out)

"""Các bản giả (test double) chạy trong bộ nhớ, thay cho đồng hồ, môi trường tool và tokenizer thật.

Nội dung chính:
- `FakeClock`: đồng hồ chỉ tiến khi gọi `advance`.
- `FakeEnv`: môi trường tool trong bộ nhớ, ghi lại mọi lời gọi.
- `WhitespaceTokenCounter`: đếm token theo khoảng trắng.

Lưu ý:
- Nhờ các bản giả này, test chạy nhanh, tất định, không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from lifemate.core.jsonutil import canonical_json


class FakeClock:
    """Đồng hồ thủ công: trả về `now`, chỉ tiến lên khi gọi `advance`.

    Tham số:
        start: thời điểm ban đầu, tính bằng giây.

    Thuộc tính:
        now: thời điểm hiện tại của đồng hồ.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        """Trả về thời điểm hiện tại của đồng hồ giả."""
        return self.now

    def advance(self, s: float) -> None:
        """Cho đồng hồ tiến thêm `s` giây."""
        self.now += s


class FakeEnv:
    """Môi trường tool (`ToolEnvLike`) trong bộ nhớ; ghi lại mọi lời gọi, kể cả lời gọi bị từ chối, giống `Session`
    thật.

    Tham số:
        allowed: tên các tool hợp lệ.
        required: tên tool tới danh sách tham số bắt buộc.

    Thuộc tính:
        allowed: như tham số cùng tên.
        required: như tham số cùng tên.
        calls: các lời gọi đã ghi.
    """

    def __init__(self, allowed: set[str], required: dict[str, list[str]] | None = None) -> None:
        self.allowed = set(allowed)
        self.required = dict(required or {})
        self.calls: list[dict[str, Any]] = []

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Từ chối tool không có trong `allowed` hoặc lời gọi thiếu tham số bắt buộc."""
        if name not in self.allowed:
            return {"ok": False, "error": f"unknown tool: {name}"}
        missing = [p for p in self.required.get(name, []) if p not in (arguments or {})]
        if missing:
            return {"ok": False, "error": f"missing required: {', '.join(missing)}"}
        return {"ok": True, "error": None}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Ghi lời gọi trước khi kiểm tra, rồi trả về kết quả kiểm tra (khi lỗi) hoặc `done`."""
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments or {})})
        verdict = self.validate(name, arguments)
        return verdict if not verdict["ok"] else {"ok": True, "done": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Trả về bản sao các lời gọi đã ghi."""
        return copy.deepcopy(self.calls)

    def get_state_hash(self) -> str:
        """Trả về SHA-256 của JSON chuẩn tắc các lời gọi, đại diện cho trạng thái môi trường."""
        return hashlib.sha256(canonical_json(self.calls).encode("utf-8")).hexdigest()


class WhitespaceTokenCounter:
    """Đếm token bằng cách tách nội dung từng tin nhắn theo khoảng trắng, cộng thêm 4 cho mỗi tin nhắn."""

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Trả về tổng số token giả của danh sách tin nhắn."""
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages)

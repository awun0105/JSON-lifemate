"""Các bản giả (test double) chạy trong bộ nhớ: FakeClock, FakeEnv, WhitespaceTokenCounter.

Dùng thay đồng hồ thật, môi trường tool thật và tokenizer thật để test chạy nhanh, tất định,
không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from lifemate.core.jsonutil import canonical_json


class FakeClock:
    """Đồng hồ thủ công: trả về `now`, chỉ tiến lên khi gọi `advance`."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        """Trả về thời điểm hiện tại của đồng hồ giả."""
        return self.now

    def advance(self, s: float) -> None:
        """Cho đồng hồ tiến thêm `s` giây."""
        self.now += s


class FakeEnv:
    """ToolEnvLike trong bộ nhớ; ghi lại mọi lời gọi, kể cả lời gọi bị từ chối (giống Session thật)."""

    def __init__(self, allowed: set[str], required: dict[str, list[str]] | None = None) -> None:
        self.allowed = set(allowed)
        self.required = dict(required or {})
        self.calls: list[dict[str, Any]] = []

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Từ chối tool không có trong `allowed` hoặc thiếu tham số bắt buộc."""
        if name not in self.allowed:
            return {"ok": False, "error": f"unknown tool: {name}"}
        missing = [p for p in self.required.get(name, []) if p not in (arguments or {})]
        if missing:
            return {"ok": False, "error": f"missing required: {', '.join(missing)}"}
        return {"ok": True, "error": None}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Ghi lời gọi trước khi kiểm tra, rồi trả kết quả kiểm tra hoặc `done`."""
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments or {})})
        verdict = self.validate(name, arguments)
        return verdict if not verdict["ok"] else {"ok": True, "done": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Trả về bản sao các lời gọi đã ghi."""
        return copy.deepcopy(self.calls)

    def get_state_hash(self) -> str:
        """Băm SHA-256 của JSON chuẩn hoá các lời gọi, đại diện cho trạng thái môi trường."""
        return hashlib.sha256(canonical_json(self.calls).encode("utf-8")).hexdigest()


class WhitespaceTokenCounter:
    """Đếm token bằng cách tách theo khoảng trắng nội dung từng tin nhắn, cộng thêm 4 cho mỗi tin nhắn."""

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Trả về tổng số token giả của danh sách tin nhắn."""
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages)

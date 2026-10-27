"""In-memory test doubles: FakeClock, FakeEnv, WhitespaceTokenCounter."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from lifemate.jsonutil import canonical_json


class FakeClock:
    """Manual clock: returns `now`, moved only by `advance`."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


class FakeEnv:
    """ToolEnvLike in memory; logs every call, rejected ones included (like the real Session)."""

    def __init__(self, allowed: set[str], required: dict[str, list[str]] | None = None) -> None:
        self.allowed = set(allowed)
        self.required = dict(required or {})
        self.calls: list[dict[str, Any]] = []

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.allowed:
            return {"ok": False, "error": f"unknown tool: {name}"}
        missing = [p for p in self.required.get(name, []) if p not in (arguments or {})]
        if missing:
            return {"ok": False, "error": f"missing required: {', '.join(missing)}"}
        return {"ok": True, "error": None}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments or {})})
        verdict = self.validate(name, arguments)
        return verdict if not verdict["ok"] else {"ok": True, "done": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.calls)

    def get_state_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.calls).encode("utf-8")).hexdigest()


class WhitespaceTokenCounter:
    """Whitespace tokens of each message content plus 4 per message."""

    def count(self, messages: list[dict[str, Any]]) -> int:
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages)

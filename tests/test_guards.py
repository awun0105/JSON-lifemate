"""T5, T18: kiểm tra các guard và chuỗi guard mở rộng được qua registry (SPEC 4.9).

Dùng catalog tổng hợp `CATALOG` và `FakeEnv` từ `tests/fakes.py`. Không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

from typing import Any

import pytest

from lifemate.agent.guards import (
    GUARD_REGISTRY,
    AfterAskGuard,
    AllowedToolGuard,
    DuplicateWriteGuard,
    GuardChain,
    SchemaGuard,
    build_guard_chain,
)
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.core.config import ConfigError
from lifemate.core.types import Action, ErrorCode, GuardContext, GuardResult, GuardVerdict
from tests.fakes import FakeEnv


def _tool(name: str, kind: str, required: list[str] | None = None) -> dict[str, Any]:
    """Tạo mô tả tool trong catalog, các tham số bắt buộc đều kiểu string."""
    props = {p: {"type": "string", "description": p} for p in required or []}
    return {
        "name": name,
        "kind": kind,
        "description": f"mô tả {name}",
        "returns": "",
        "parameters": {"type": "object", "properties": props, "required": required or []},
        "app": name.split(".", maxsplit=1)[0],
    }


CATALOG: dict[str, Any] = {
    "apps": [
        {
            "id": "alarm",
            "name_vi": "Báo thức",
            "description": "đặt báo thức",
            "tools": [_tool("alarm.list", "read"), _tool("alarm.delete", "write", ["alarm_id"])],
        },
        {
            "id": "notes",
            "name_vi": "Ghi chú",
            "description": "ghi chú",
            "tools": [_tool("notes.search", "read"), _tool("notes.create", "write", ["title"])],
        },
        {
            "id": "sys",
            "name_vi": "Hệ thống",
            "description": "hệ thống",
            "tools": [_tool(ASK_TOOL, "meta", ["question"])],
        },
    ]
}
INSTALLED = {
    "alarm": {"tools": ["alarm.list", "alarm.delete"]},
    "notes": {"tools": ["notes.search", "notes.create"]},
}
REQUIRED = {"alarm.delete": ["alarm_id"], "notes.create": ["title"], ASK_TOOL: ["question"]}
INDEX = CatalogIndex(CATALOG)
ALLOWED = frozenset(INDEX.allowed_tools(INSTALLED))


def make_env() -> FakeEnv:
    """Tạo `FakeEnv` biết catalog tổng hợp."""
    return FakeEnv(set(ALLOWED), REQUIRED)


def call(name: str, **arguments: Any) -> Action:
    """Tạo action gọi tool `name` với tham số cho trước."""
    return Action(kind="call", name=name, arguments=arguments)


def ctx(env: FakeEnv, calls: tuple[Action, ...] = ()) -> GuardContext:
    """Tạo `GuardContext` với các lần gọi trước đó `calls`."""
    return GuardContext(env=env, allowed=ALLOWED, tool_kind=INDEX.tool_kinds(), calls_so_far=calls)


def test_unknown_tool_blocked() -> None:
    """Tool không được cài bị chặn với mã UNKNOWN_TOOL và không chạm tới môi trường."""
    env = make_env()
    res, warns = build_guard_chain(["allowed_tool", "schema", "duplicate_write"]).check(call("bank.pay"), ctx(env))
    assert res.verdict is GuardVerdict.BLOCK and res.code is ErrorCode.UNKNOWN_TOOL
    assert res.message is not None and "bank.pay" in res.message
    assert warns == [] and env.get_tool_calls() == []


def test_missing_required_blocked() -> None:
    """Thiếu tham số bắt buộc bị chặn với mã INVALID_ARGS và thông báo chính xác."""
    env = make_env()
    res, _ = build_guard_chain(["allowed_tool", "schema"]).check(call("alarm.delete"), ctx(env))
    assert res.verdict is GuardVerdict.BLOCK and res.code is ErrorCode.INVALID_ARGS
    assert res.message == "missing required: alarm_id"
    assert env.get_tool_calls() == []


def test_duplicate_write_warns_only() -> None:
    """Lặp lại lệnh ghi y hệt chỉ sinh cảnh báo, không chặn; tham số khác hoặc lệnh đọc thì không cảnh báo."""
    env = make_env()
    prev = call("alarm.delete", alarm_id="a1")
    chain = build_guard_chain(["allowed_tool", "schema", "duplicate_write"])
    res, warns = chain.check(call("alarm.delete", alarm_id="a1"), ctx(env, (prev,)))
    assert res.verdict is GuardVerdict.PASS
    assert [w.code for w in warns] == [ErrorCode.DUPLICATE_WRITE]
    res, warns = chain.check(call("alarm.delete", alarm_id="a2"), ctx(env, (prev,)))
    assert warns == []
    read = call("alarm.list")
    assert DuplicateWriteGuard().check(read, ctx(env, (read,))).verdict is GuardVerdict.PASS


def test_after_ask_stops() -> None:
    """Sau khi đã hỏi lại người dùng thì mọi lệnh gọi tiếp theo bị dừng."""
    env = make_env()
    asked = (call(ASK_TOOL, question="giờ nào?"),)
    assert AfterAskGuard().check(call("alarm.list"), ctx(env, asked)).verdict is GuardVerdict.STOP
    assert AfterAskGuard().check(call("alarm.list"), ctx(env)).verdict is GuardVerdict.PASS


def test_chain_order_stops_at_first_block() -> None:
    """Chuỗi guard chạy đúng thứ tự, dừng ở guard chặn đầu tiên và vẫn giữ cảnh báo trước đó."""
    seen: list[str] = []

    class Spy:
        """Guard giả ghi lại thứ tự được gọi và trả verdict cố định."""

        def __init__(self, tag: str, verdict: GuardVerdict) -> None:
            self.tag, self.verdict = tag, verdict

        def check(self, action: Action, c: GuardContext) -> GuardResult:
            seen.append(self.tag)
            return GuardResult(
                self.verdict, ErrorCode.DUPLICATE_WRITE if self.verdict is GuardVerdict.WARN else None, None
            )

    chain = GuardChain([Spy("a", GuardVerdict.WARN), Spy("b", GuardVerdict.BLOCK), Spy("c", GuardVerdict.PASS)])
    res, warns = chain.check(call("alarm.list"), ctx(make_env()))
    assert seen == ["a", "b"] and res.verdict is GuardVerdict.BLOCK and len(warns) == 1
    seen.clear()
    res, _ = GuardChain([Spy("a", GuardVerdict.PASS), Spy("c", GuardVerdict.PASS)]).check(
        call("alarm.list"), ctx(make_env())
    )
    assert seen == ["a", "c"] and res.verdict is GuardVerdict.PASS


def test_registry_and_build() -> None:
    """Registry ánh xạ đúng tên sang lớp guard, chuỗi giữ thứ tự config, tên lạ gây `ConfigError`."""
    assert GUARD_REGISTRY["allowed_tool"] is AllowedToolGuard
    assert GUARD_REGISTRY["schema"] is SchemaGuard
    assert GUARD_REGISTRY["duplicate_write"] is DuplicateWriteGuard
    assert GUARD_REGISTRY["after_ask"] is AfterAskGuard
    chain = build_guard_chain(["schema", "allowed_tool"])
    assert [type(g) for g in chain.guards] == [SchemaGuard, AllowedToolGuard]
    with pytest.raises(ConfigError):
        build_guard_chain(["nope"])


def test_guard_chain_open_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """T18: guard chỉ thêm qua registry và config vẫn được gọi, không cần sửa mã chuỗi guard."""
    invoked: list[str | None] = []

    class Custom:
        """Guard tuỳ biến ghi lại tên tool được kiểm tra."""

        def check(self, action: Action, c: GuardContext) -> GuardResult:
            invoked.append(action.name)
            return GuardResult(GuardVerdict.PASS, None, None)

    monkeypatch.setitem(GUARD_REGISTRY, "custom", Custom)
    build_guard_chain(["allowed_tool", "custom"]).check(call("alarm.list"), ctx(make_env()))
    assert invoked == ["alarm.list"]

"""Chuỗi guard kiểm tra mỗi lời gọi tool trước khi gửi tới env (SPEC 4.9).

Nội dung chính:
- `AllowedToolGuard` (G1): chặn tool không được phép.
- `SchemaGuard` (G2): chặn tham số mà `env.validate` từ chối.
- `DuplicateWriteGuard` (G3): cảnh báo khi một lời gọi `write` bị lặp lại y hệt.
- `AfterAskGuard` (G4): dừng nếu đã hỏi lại người dùng; mặc định không bật.
- `GuardChain`: chạy các guard theo thứ tự và trả về quyết định cuối.
- `GUARD_REGISTRY`, `build_guard_chain`: dựng chuỗi guard từ tên trong `[agent].guards`.

Cách hoạt động:
- Dùng mẫu Chain of Responsibility: các guard chạy theo thứ tự trong config và dừng ở kết quả BLOCK hoặc STOP
  đầu tiên. Các kết quả WARN gặp trên đường được gom lại và trả về cùng quyết định.
- BLOCK: lời gọi không được gửi tới env. Vòng lặp agent đưa thông báo lỗi lại cho model và tính một lần thử lại.
- WARN: lời gọi vẫn được gửi đi, chỉ ghi thêm mã lỗi.

Lưu ý:
- Thông báo trong `GuardResult` được đưa thẳng cho model đọc, nên viết bằng tiếng Việt.
"""

from __future__ import annotations

from collections.abc import Sequence

from lifemate.agent.tooldocs import ASK_TOOL
from lifemate.core.config import ConfigError
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Action, ErrorCode, Guard, GuardContext, GuardResult, GuardVerdict

PASS = GuardResult(verdict=GuardVerdict.PASS, code=None, message=None)


class AllowedToolGuard:
    """G1: tool được gọi phải nằm trong tập tool được phép của mẫu."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """Trả về BLOCK với mã `unknown_tool` khi tool không tồn tại hoặc app chưa cài, ngược lại trả về PASS."""
        if action.name in ctx.allowed:
            return PASS
        return GuardResult(
            GuardVerdict.BLOCK, ErrorCode.UNKNOWN_TOOL, f"tool không tồn tại hoặc chưa cài: {action.name}"
        )


class SchemaGuard:
    """G2: tham số của lời gọi phải được `env.validate` chấp nhận."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """Trả về BLOCK với mã `invalid_args` kèm nguyên văn lỗi từ `env.validate`, hoặc PASS khi tham số hợp lệ."""
        verdict = ctx.env.validate(action.name or "", action.arguments or {})
        if verdict.get("ok"):
            return PASS
        return GuardResult(GuardVerdict.BLOCK, ErrorCode.INVALID_ARGS, str(verdict.get("error")))


class DuplicateWriteGuard:
    """G3: cảnh báo (không bao giờ chặn) khi một lời gọi `write` bị lặp lại y hệt."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """Trả về WARN với mã `duplicate_write` khi cùng lời gọi `write` đã chạy trước đó, ngược lại PASS."""
        # Chỉ xét tool loại `write`: gọi lại y hệt một tool `read` không gây hại gì.
        if ctx.tool_kind.get(action.name or "") != "write":
            return PASS
        # So sánh qua JSON chuẩn tắc, để thứ tự khoá trong `arguments` không ảnh hưởng kết quả.
        key = canonical_json(action.arguments or {})
        for prev in ctx.calls_so_far:
            if prev.name == action.name and canonical_json(prev.arguments or {}) == key:
                return GuardResult(
                    GuardVerdict.WARN, ErrorCode.DUPLICATE_WRITE, f"gọi lại write giống hệt: {action.name}"
                )
        return PASS


class AfterAskGuard:
    """G4 (mặc định không bật): không gọi thêm tool nào sau `sys.ask_missing_info`."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """Trả về STOP khi trước đó đã có lời gọi hỏi lại người dùng, ngược lại PASS."""
        if any(prev.name == ASK_TOOL for prev in ctx.calls_so_far):
            return GuardResult(GuardVerdict.STOP, None, "đã hỏi lại người dùng")
        return PASS


class GuardChain:
    """Chạy các guard theo thứ tự và dừng ở kết quả BLOCK hoặc STOP đầu tiên.

    Tham số:
        guards: danh sách guard theo thứ tự chạy; được lưu thành tuple để không bị sửa.
    """

    def __init__(self, guards: Sequence[Guard]) -> None:
        self.guards: tuple[Guard, ...] = tuple(guards)

    def check(self, action: Action, ctx: GuardContext) -> tuple[GuardResult, list[GuardResult]]:
        """Kiểm tra một `Action` qua cả chuỗi guard.

        Tham số:
            action: hành động LLM muốn thực hiện.
            ctx: thông tin guard được phép xem.

        Trả về:
            Cặp (kết quả quyết định, danh sách WARN gặp trên đường). Kết quả quyết định là PASS khi không guard nào
            chặn hay dừng.
        """
        warnings: list[GuardResult] = []
        for guard in self.guards:
            res = guard.check(action, ctx)
            if res.verdict in (GuardVerdict.BLOCK, GuardVerdict.STOP):
                return res, warnings
            if res.verdict is GuardVerdict.WARN:
                warnings.append(res)
        return PASS, warnings


GUARD_REGISTRY: dict[str, type] = {
    "allowed_tool": AllowedToolGuard,
    "schema": SchemaGuard,
    "duplicate_write": DuplicateWriteGuard,
    "after_ask": AfterAskGuard,
}


def build_guard_chain(names: Sequence[str]) -> GuardChain:
    """Tạo chuỗi guard từ các tên trong `[agent].guards`, giữ đúng thứ tự trong config.

    Tham số:
        names: tên guard, mỗi tên là một khoá của `GUARD_REGISTRY`.

    Trả về:
        `GuardChain` chứa các guard đã tạo.

    Lỗi:
        ConfigError: khi có tên không nằm trong `GUARD_REGISTRY`.
    """
    guards: list[Guard] = []
    for name in names:
        cls = GUARD_REGISTRY.get(name)
        if cls is None:
            raise ConfigError(f"agent.guards: unknown guard {name!r}")
        guards.append(cls())
    return GuardChain(guards)

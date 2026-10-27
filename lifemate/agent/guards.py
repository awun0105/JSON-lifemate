"""Chuỗi guard được kiểm tra trước mỗi lời gọi tool (SPEC 4.9).

Nội dung: bốn guard G1-G4, lớp `GuardChain` và hàm `build_guard_chain` dựng chuỗi từ cấu hình.

Cách xây dựng:
- Mẫu Chain of Responsibility: guard chạy theo thứ tự cấu hình, dừng ở kết quả BLOCK hoặc STOP đầu tiên;
  các kết quả WARN được gom lại và trả về cùng quyết định.
- BLOCK không gửi lời gọi tới env: vòng lặp agent đưa thông báo lỗi (tiếng Việt) lại cho mô hình và tiêu
  một lần thử lại. WARN vẫn cho lời gọi đi qua, chỉ ghi mã lỗi.
- Thông báo trong `GuardResult` là văn bản mô hình đọc được, nên viết bằng tiếng Việt.
- Guard được đăng ký theo tên trong `GUARD_REGISTRY` để chọn qua `[agent].guards`.
"""

from __future__ import annotations

from collections.abc import Sequence

from lifemate.agent.tooldocs import ASK_TOOL
from lifemate.core.config import ConfigError
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Action, ErrorCode, Guard, GuardContext, GuardResult, GuardVerdict

PASS = GuardResult(verdict=GuardVerdict.PASS, code=None, message=None)


class AllowedToolGuard:
    """G1: tool phải nằm trong tập tool được phép của mẫu."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """BLOCK với mã `unknown_tool` khi tên tool không được phép (không có hoặc app chưa cài)."""
        if action.name in ctx.allowed:
            return PASS
        return GuardResult(
            GuardVerdict.BLOCK, ErrorCode.UNKNOWN_TOOL, f"tool không tồn tại hoặc chưa cài: {action.name}"
        )


class SchemaGuard:
    """G2: `env.validate` phải chấp nhận các tham số."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """BLOCK với mã `invalid_args`, kèm nguyên văn lỗi từ bộ kiểm tra của env."""
        verdict = ctx.env.validate(action.name or "", action.arguments or {})
        if verdict.get("ok"):
            return PASS
        return GuardResult(GuardVerdict.BLOCK, ErrorCode.INVALID_ARGS, str(verdict.get("error")))


class DuplicateWriteGuard:
    """G3: cảnh báo (không bao giờ chặn) khi một lời gọi `write` bị lặp lại y hệt."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """WARN với mã `duplicate_write` khi cùng lời gọi write đã được thực hiện trước đó."""
        # Chỉ xét tool loại write: gọi lại tool read giống hệt là vô hại.
        if ctx.tool_kind.get(action.name or "") != "write":
            return PASS
        # So sánh qua JSON chuẩn hóa để thứ tự khóa trong arguments không ảnh hưởng kết quả.
        key = canonical_json(action.arguments or {})
        for prev in ctx.calls_so_far:
            if prev.name == action.name and canonical_json(prev.arguments or {}) == key:
                return GuardResult(
                    GuardVerdict.WARN, ErrorCode.DUPLICATE_WRITE, f"gọi lại write giống hệt: {action.name}"
                )
        return PASS


class AfterAskGuard:
    """G4 (không bật mặc định): không gọi thêm tool nào sau `sys.ask_missing_info`."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """STOP khi đã có một lời gọi hỏi lại người dùng."""
        if any(prev.name == ASK_TOOL for prev in ctx.calls_so_far):
            return GuardResult(GuardVerdict.STOP, None, "đã hỏi lại người dùng")
        return PASS


class GuardChain:
    """Chain of Responsibility: chạy các guard theo thứ tự, dừng ở BLOCK hoặc STOP đầu tiên.

    Args:
        guards: danh sách guard, được giữ dưới dạng tuple bất biến.
    """

    def __init__(self, guards: Sequence[Guard]) -> None:
        self.guards: tuple[Guard, ...] = tuple(guards)

    def check(self, action: Action, ctx: GuardContext) -> tuple[GuardResult, list[GuardResult]]:
        """Kiểm tra một hành động qua cả chuỗi.

        Returns:
            Cặp (kết quả quyết định, danh sách WARN gặp trên đường). Kết quả quyết định là PASS khi không
            guard nào chặn hay dừng.
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
    """Tạo các guard có tên trong `[agent].guards`, đúng thứ tự cấu hình.

    Raises:
        ConfigError: khi có tên guard không nằm trong `GUARD_REGISTRY`.
    """
    guards: list[Guard] = []
    for name in names:
        cls = GUARD_REGISTRY.get(name)
        if cls is None:
            raise ConfigError(f"agent.guards: unknown guard {name!r}")
        guards.append(cls())
    return GuardChain(guards)

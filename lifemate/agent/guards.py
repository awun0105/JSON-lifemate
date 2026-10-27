"""Guard chain checked before every tool call (SPEC 4.9)."""

from __future__ import annotations

from collections.abc import Sequence

from lifemate.agent.tooldocs import ASK_TOOL
from lifemate.core.config import ConfigError
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Action, ErrorCode, Guard, GuardContext, GuardResult, GuardVerdict

PASS = GuardResult(verdict=GuardVerdict.PASS, code=None, message=None)


class AllowedToolGuard:
    """G1: the tool must be in the sample's allowed set."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """BLOCK with `unknown_tool` when the name is not allowed."""
        if action.name in ctx.allowed:
            return PASS
        return GuardResult(
            GuardVerdict.BLOCK, ErrorCode.UNKNOWN_TOOL, f"tool không tồn tại hoặc chưa cài: {action.name}"
        )


class SchemaGuard:
    """G2: `env.validate` must accept the arguments."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """BLOCK with `invalid_args` and the validator's error text."""
        verdict = ctx.env.validate(action.name or "", action.arguments or {})
        if verdict.get("ok"):
            return PASS
        return GuardResult(GuardVerdict.BLOCK, ErrorCode.INVALID_ARGS, str(verdict.get("error")))


class DuplicateWriteGuard:
    """G3: warn (never block) on an identical repeated `write` call."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """WARN with `duplicate_write` when the same write call was already made."""
        if ctx.tool_kind.get(action.name or "") != "write":
            return PASS
        key = canonical_json(action.arguments or {})
        for prev in ctx.calls_so_far:
            if prev.name == action.name and canonical_json(prev.arguments or {}) == key:
                return GuardResult(
                    GuardVerdict.WARN, ErrorCode.DUPLICATE_WRITE, f"gọi lại write giống hệt: {action.name}"
                )
        return PASS


class AfterAskGuard:
    """G4 (not default): no tool call after `sys.ask_missing_info`."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult:
        """STOP when an ask call was already made."""
        if any(prev.name == ASK_TOOL for prev in ctx.calls_so_far):
            return GuardResult(GuardVerdict.STOP, None, "đã hỏi lại người dùng")
        return PASS


class GuardChain:
    """Chain of Responsibility: run guards in order, stop at the first BLOCK or STOP."""

    def __init__(self, guards: Sequence[Guard]) -> None:
        self.guards: tuple[Guard, ...] = tuple(guards)

    def check(self, action: Action, ctx: GuardContext) -> tuple[GuardResult, list[GuardResult]]:
        """Deciding result (PASS if none decided) and the WARN results met on the way."""
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
    """Instantiate the guards named in `[agent].guards`, in config order."""
    guards: list[Guard] = []
    for name in names:
        cls = GUARD_REGISTRY.get(name)
        if cls is None:
            raise ConfigError(f"agent.guards: unknown guard {name!r}")
        guards.append(cls())
    return GuardChain(guards)

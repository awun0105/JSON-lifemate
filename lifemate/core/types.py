"""Shared value objects, enums and Protocols (SPEC 4.0, 4.2, 4.4, 4.7, 4.8, 4.9, 4.12, 4.13)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol


class StopReason(Enum):
    """Why the agent loop ended for a sample."""

    DONE = "done"
    ASKED = "asked"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PARSE_ERROR = "parse_error"
    RETRIES_EXHAUSTED = "retries_exhausted"
    EXCEPTION = "exception"


class ErrorCode(Enum):
    """Error codes recorded in steps and in the sample error list."""

    PARSE_ERROR = "parse_error"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    DUPLICATE_WRITE = "duplicate_write"
    EXCEPTION = "exception"


class GuardVerdict(Enum):
    """Outcome of a guard check."""

    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Turn:
    """One turn of the conversation history."""

    role: Literal["user", "assistant"]
    audio: str | None
    audio_text: str | None
    text: str | None
    tool_calls: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Sample:
    """One competition sample; `label` is read only by FakeLLM and the scorer."""

    id: str
    split: str
    audio: str
    audio_text: str | None
    context: dict[str, Any]
    history: tuple[Turn, ...]
    label: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Text of the final turn and of the user turns in history."""

    final: str
    history: tuple[str | None, ...]
    seconds: float


@dataclass(frozen=True, slots=True)
class Generation:
    """One LLM completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    timed_out: bool = False  # aborted at the sample deadline; `text` is partial and must not be parsed


@dataclass(frozen=True, slots=True)
class Action:
    """A parsed LLM decision: call a tool or finish."""

    kind: Literal["call", "done"]
    name: str | None
    arguments: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the trace."""
        return {"kind": self.kind, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ParseError:
    """LLM output that could not be parsed into an Action."""

    message: str
    raw: str


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Result of one guard check."""

    verdict: GuardVerdict
    code: ErrorCode | None
    message: str | None


@dataclass(frozen=True, slots=True)
class GuardContext:
    """What a guard may look at."""

    env: ToolEnvLike
    allowed: frozenset[str]
    tool_kind: Mapping[str, str]
    calls_so_far: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One agent step, as written to trace.jsonl."""

    i: int
    prompt_tokens: int
    completion_tokens: int
    llm_s: float
    raw: str
    action: Action | None
    guard: GuardVerdict | None
    warnings: tuple[ErrorCode, ...] = ()
    error: ErrorCode | None = None
    result: dict[str, Any] | None = None
    tool_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the trace, enums as `.value`."""
        return {
            "i": self.i,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_s": self.llm_s,
            "raw": self.raw,
            "action": self.action.to_dict() if self.action else None,
            "guard": self.guard.value if self.guard else None,
            "warnings": [w.value for w in self.warnings],
            "error": self.error.value if self.error else None,
            "result": self.result,
            "tool_s": self.tool_s,
        }


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Outcome of one sample, as written to trace.jsonl."""

    id: str
    stop_reason: StopReason
    errors: tuple[ErrorCode, ...]
    transcript: Transcript | None
    steps: tuple[StepRecord, ...]
    tool_calls: tuple[dict[str, Any], ...]
    state_hash: str
    timing: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    prompt_shrink_level: int = 0
    transcript_source: str = "gold"
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the trace, enums as `.value`."""
        t = self.transcript
        return {
            "id": self.id,
            "stop_reason": self.stop_reason.value,
            "errors": [e.value for e in self.errors],
            "prompt_shrink_level": self.prompt_shrink_level,
            "transcript": None
            if t is None
            else {"final": t.final, "history": list(t.history), "source": self.transcript_source},
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": list(self.tool_calls),
            "state_hash": self.state_hash,
            "timing": self.timing,
            "tokens": self.tokens,
            "traceback": self.traceback,
        }


class ToolEnvLike(Protocol):
    """One zaic_tools environment for one sample."""

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


class LLMBackend(Protocol):
    """Text generation constrained by a JSON schema."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """One completion; stop and return `timed_out=True` once `time_left_s` seconds have passed (None: no limit)."""
        ...

    def warmup(self) -> None: ...


class TokenCounter(Protocol):
    """Counts prompt tokens of a message list."""

    def count(self, messages: list[dict[str, Any]]) -> int: ...


class Transcriber(Protocol):
    """Turns sample audio into text."""

    def transcribe(self, sample: Sample) -> Transcript: ...
    def warmup(self) -> None: ...


class Clock(Protocol):
    """Monotonic seconds."""

    def __call__(self) -> float: ...


class Guard(Protocol):
    """One link of the guard chain."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult: ...


class AgentObserver(Protocol):
    """Receives agent events (tracer, null observer)."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None: ...
    def on_step(self, step: StepRecord) -> None: ...
    def on_stop(self, result: SampleResult) -> None: ...

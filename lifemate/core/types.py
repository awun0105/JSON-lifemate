"""Đối tượng giá trị, enum và Protocol dùng chung (SPEC 4.0, 4.2, 4.4, 4.7, 4.8, 4.9, 4.12, 4.13).

Nội dung file:
- Enum: `StopReason` (lý do dừng một mẫu), `ErrorCode` (mã lỗi), `GuardVerdict` (kết luận của guard).
- Dataclass giá trị: dữ liệu vào (`Turn`, `Sample`, `Transcript`), kết quả LLM (`Generation`, `Action`,
  `ParseError`), guard (`GuardResult`, `GuardContext`) và bản ghi trace (`StepRecord`, `SampleResult`).
- Protocol: giao diện của môi trường công cụ, backend LLM, bộ đếm token, bộ chuyển giọng nói, đồng hồ, guard
  và observer.

Cách xây dựng:
- Module này không import gì trong lifemate, nên mọi tầng (io, backends, agent) dùng được mà không vòng import.
- Dataclass `frozen=True, slots=True`: giá trị bất biến, an toàn khi truyền qua các tầng và khi ghi trace.
- Phụ thuộc giữa các tầng đi qua Protocol (kiểu cấu trúc), nên test thay backend thật bằng bản giả dễ dàng.
- `to_dict` đổi enum thành `.value` để trace.jsonl chỉ chứa kiểu JSON thuần.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol


class StopReason(Enum):
    """Lý do vòng lặp agent kết thúc với một mẫu."""

    DONE = "done"
    ASKED = "asked"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PARSE_ERROR = "parse_error"
    RETRIES_EXHAUSTED = "retries_exhausted"
    EXCEPTION = "exception"


class ErrorCode(Enum):
    """Mã lỗi ghi trong từng bước và trong danh sách lỗi của mẫu."""

    PARSE_ERROR = "parse_error"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    DUPLICATE_WRITE = "duplicate_write"
    EXCEPTION = "exception"


class GuardVerdict(Enum):
    """Kết luận của một lần kiểm tra guard: cho qua, chặn, cảnh báo hoặc dừng hẳn."""

    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Turn:
    """Một lượt trong lịch sử hội thoại (của user hoặc assistant)."""

    role: Literal["user", "assistant"]
    audio: str | None
    audio_text: str | None
    text: str | None
    tool_calls: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Sample:
    """Một mẫu dữ liệu của cuộc thi; chỉ FakeLLM và bộ chấm điểm được đọc `label`."""

    id: str
    split: str
    audio: str
    audio_text: str | None
    context: dict[str, Any]
    history: tuple[Turn, ...]
    label: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Văn bản của lượt cuối và của các lượt user trong lịch sử; `seconds` là thời gian chuyển giọng nói."""

    final: str
    history: tuple[str | None, ...]
    seconds: float


@dataclass(frozen=True, slots=True)
class Generation:
    """Một lần sinh văn bản của LLM, kèm số token và thời gian."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    timed_out: bool = False  # bị dừng tại hạn chót của mẫu; `text` chỉ là một phần, không được parse


@dataclass(frozen=True, slots=True)
class Action:
    """Quyết định của LLM sau khi parse: gọi một công cụ (`call`) hoặc kết thúc (`done`)."""

    kind: Literal["call", "done"]
    name: str | None
    arguments: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Tuần tự hóa thành dict để ghi trace."""
        return {"kind": self.kind, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ParseError:
    """Đầu ra LLM không parse được thành Action; giữ lại `raw` để gỡ lỗi."""

    message: str
    raw: str


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Kết quả một lần kiểm tra guard; `code` và `message` là None khi cho qua."""

    verdict: GuardVerdict
    code: ErrorCode | None
    message: str | None


@dataclass(frozen=True, slots=True)
class GuardContext:
    """Những gì guard được phép xem: môi trường, danh sách công cụ cho phép, loại công cụ, các lần gọi trước."""

    env: ToolEnvLike
    allowed: frozenset[str]
    tool_kind: Mapping[str, str]
    calls_so_far: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Một bước của agent, đúng như được ghi vào trace.jsonl."""

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
        """Tuần tự hóa thành dict để ghi trace, enum được đổi thành `.value`."""
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
    """Kết quả của một mẫu, đúng như được ghi vào trace.jsonl."""

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
        """Tuần tự hóa thành dict để ghi trace, enum được đổi thành `.value`."""
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
    """Một môi trường zaic_tools cho một mẫu: kiểm tra, gọi công cụ, lấy danh sách lần gọi và hash trạng thái."""

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


class LLMBackend(Protocol):
    """Backend sinh văn bản bị ràng buộc theo JSON schema."""

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Sinh một lần; dừng và trả về `timed_out=True` khi đã hết `time_left_s` giây (None: không giới hạn)."""
        ...

    def warmup(self) -> None: ...


class TokenCounter(Protocol):
    """Đếm số token prompt của một danh sách tin nhắn."""

    def count(self, messages: list[dict[str, Any]]) -> int: ...


class Transcriber(Protocol):
    """Chuyển audio của mẫu thành văn bản."""

    def transcribe(self, sample: Sample) -> Transcript: ...
    def warmup(self) -> None: ...


class Clock(Protocol):
    """Đồng hồ trả về số giây đơn điệu tăng (thay được bằng đồng hồ giả trong test)."""

    def __call__(self) -> float: ...


class Guard(Protocol):
    """Một mắt xích trong chuỗi guard, kiểm tra từng Action trước khi thực thi."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult: ...


class AgentObserver(Protocol):
    """Nhận sự kiện của agent (tracer, observer rỗng)."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None: ...
    def on_step(self, step: StepRecord) -> None: ...
    def on_stop(self, result: SampleResult) -> None: ...

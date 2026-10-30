"""Khai báo các kiểu dữ liệu, enum và Protocol mà mọi phần của lifemate dùng chung (SPEC 4.0, 4.2, 4.7–4.9, 4.12, 4.13).

Nội dung chính:
- `StopReason`, `ErrorCode`, `GuardVerdict`: lý do dừng một mẫu, mã lỗi, và kết luận của guard.
- `Turn`, `Sample`, `Transcript`: dữ liệu đầu vào của một mẫu.
- `Generation`, `Action`, `ParseError`: đầu ra của LLM trước và sau khi parse.
- `GuardResult`, `GuardContext`: đầu ra và đầu vào của guard.
- `StepRecord`, `SampleResult`: bản ghi một bước và một mẫu trong `trace.jsonl`.
- `ToolEnvLike`, `LLMBackend`, `TokenCounter`, `Transcriber`, `Clock`, `Guard`, `AgentObserver`: giao diện
  (Protocol) của các thành phần thay thế được.

Cách hoạt động:
- Module này không import gì khác trong lifemate, nên mọi tầng (`io`, `backends`, `agent`) đều dùng được mà
  không bị import vòng.
- Các dataclass đều `frozen=True, slots=True`: giá trị không đổi, an toàn khi truyền qua các tầng và khi ghi trace.
- Các tầng phụ thuộc nhau qua Protocol (kiểu cấu trúc), nên test dễ thay backend thật bằng bản giả.
- `to_dict` đổi enum thành `.value`, để `trace.jsonl` chỉ chứa kiểu JSON thuần.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol


class StopReason(Enum):
    """Lý do vòng lặp agent dừng với một mẫu.

    `TIMEOUT` nghĩa là đã tới `deadline_s` và bước LLM đang chạy bị ngắt. `EXCEPTION` là lỗi ngoài dự kiến, do
    `Runner` bắt.
    """

    DONE = "done"
    ASKED = "asked"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PARSE_ERROR = "parse_error"
    RETRIES_EXHAUSTED = "retries_exhausted"
    EXCEPTION = "exception"


class ErrorCode(Enum):
    """Mã lỗi ghi trong từng bước (`StepRecord.error`) và trong danh sách lỗi của mẫu (`SampleResult.errors`).

    `UNKNOWN_TOOL` là tool không được phép, `INVALID_ARGS` là `env.validate` từ chối tham số, `DUPLICATE_WRITE` là
    gọi lại y hệt một lời gọi `write` trước đó (chỉ cảnh báo).
    """

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
    """Một lượt trong lịch sử hội thoại, của user hoặc của assistant.

    Thuộc tính:
        role: `"user"` hoặc `"assistant"`.
        audio: lượt user: đường dẫn file audio, tương đối trong split.
        audio_text: lượt user: transcript gốc; chỉ tập train có.
        text: lượt assistant: câu trả lời bằng chữ.
        tool_calls: lượt assistant: các lời gọi tool, có thể rỗng.
        tool_results: lượt assistant: kết quả của các lời gọi tool, có thể rỗng.
    """

    role: Literal["user", "assistant"]
    audio: str | None
    audio_text: str | None
    text: str | None
    tool_calls: tuple[dict[str, Any], ...]
    tool_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Sample:
    """Một mẫu dữ liệu của cuộc thi.

    Thuộc tính:
        id: mã mẫu, ví dụ `train_0874`.
        split: tên split chứa mẫu.
        audio: đường dẫn file audio của lượt cuối.
        audio_text: transcript gốc của lượt cuối; chỉ tập train có.
        context: ngữ cảnh đi kèm, giữ nguyên văn từ file jsonl.
        history: các lượt hội thoại trước lượt cuối.
        label: đáp án; chỉ tập train có. Chỉ `FakeLLM` và bộ chấm được đọc trường này, agent không được đọc.
    """

    id: str
    split: str
    audio: str
    audio_text: str | None
    context: dict[str, Any]
    history: tuple[Turn, ...]
    label: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Văn bản của lượt cuối và của các lượt user trong lịch sử, sau bước chuyển giọng nói thành chữ.

    Thuộc tính:
        final: văn bản lượt cuối.
        history: văn bản từng lượt trong lịch sử; `None` ở lượt không phải của user.
        seconds: thời gian chuyển giọng nói thành chữ, tính bằng giây.
        raw: chữ ASR trước khi lọc ảo giác (xem chú thích bên dưới).
    """

    final: str
    history: tuple[str | None, ...]
    seconds: float
    # Chữ ASR trước khi lọc ảo giác, theo thứ tự file: các lượt user trong lịch sử trước, lượt cuối sau.
    # Rỗng khi không chạy ASR (dùng transcript gold). Chỉ `eval_asr` đọc trường này để đếm câu bịa; agent chỉ dùng
    # `final` và `history`.
    raw: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Generation:
    """Kết quả một lần sinh văn bản của LLM, kèm số token và thời gian.

    Thuộc tính:
        text: văn bản LLM sinh ra.
        prompt_tokens: số token của prompt.
        completion_tokens: số token sinh ra.
        seconds: thời gian sinh, tính bằng giây.
        timed_out: `True` khi bị ngắt tại hạn chót của mẫu.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    timed_out: bool = False  # bị ngắt tại hạn chót của mẫu; khi đó `text` chỉ là một phần nên không được parse


@dataclass(frozen=True, slots=True)
class Action:
    """Quyết định của LLM sau khi parse: gọi một tool (`call`) hoặc kết thúc (`done`).

    Thuộc tính:
        kind: `"call"` hoặc `"done"`.
        name: tên tool khi `kind = "call"`, ngược lại là `None`.
        arguments: tham số của tool khi `kind = "call"`, ngược lại là `None`.
    """

    kind: Literal["call", "done"]
    name: str | None
    arguments: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Chuyển thành dict để ghi trace."""
        return {"kind": self.kind, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True, slots=True)
class ParseError:
    """Đầu ra LLM không parse được thành `Action`.

    Thuộc tính:
        message: mô tả lỗi parse.
        raw: văn bản gốc của LLM, giữ lại để gỡ lỗi.
    """

    message: str
    raw: str


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Kết quả một lần kiểm tra guard.

    Thuộc tính:
        verdict: kết luận của guard.
        code: mã lỗi; `None` khi cho qua.
        message: giải thích cho LLM hoặc cho trace; `None` khi cho qua.
    """

    verdict: GuardVerdict
    code: ErrorCode | None
    message: str | None


@dataclass(frozen=True, slots=True)
class GuardContext:
    """Những thông tin guard được phép xem khi kiểm tra một `Action`.

    Thuộc tính:
        env: môi trường tool của mẫu.
        allowed: tên các tool được phép gọi.
        tool_kind: tên tool tới loại tool (ví dụ `read` hoặc `write`).
        calls_so_far: các `Action` đã thực thi trước đó trong mẫu.
    """

    env: ToolEnvLike
    allowed: frozenset[str]
    tool_kind: Mapping[str, str]
    calls_so_far: tuple[Action, ...]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Một bước của agent, đúng như được ghi vào `trace.jsonl`.

    Thuộc tính:
        i: số thứ tự bước, bắt đầu từ 0.
        prompt_tokens: số token của prompt ở bước này.
        completion_tokens: số token LLM sinh ra.
        llm_s: thời gian LLM chạy, tính bằng giây.
        raw: văn bản gốc của LLM.
        action: `Action` đã parse; `None` khi parse lỗi.
        guard: kết luận của guard; `None` khi bước không gọi tool (parse lỗi, `done`, hết giờ).
        warnings: các mã cảnh báo của guard (không chặn lời gọi).
        error: mã lỗi của bước; `None` khi không lỗi.
        result: kết quả tool trả về; `None` khi không gọi tool.
        tool_s: thời gian chạy tool, tính bằng giây.
    """

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
        """Chuyển thành dict để ghi trace; enum được đổi thành `.value`."""
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
    """Kết quả của một mẫu, đúng như được ghi vào `trace.jsonl`.

    Thuộc tính:
        id: mã mẫu.
        stop_reason: lý do dừng.
        errors: các mã lỗi gặp trong mẫu.
        transcript: transcript đã dùng; `None` khi mẫu lỗi trước khi có transcript.
        steps: các bước của agent.
        tool_calls: các lời gọi tool mà môi trường ghi nhận.
        state_hash: hash trạng thái môi trường sau mẫu, dùng để chấm điểm.
        timing: thời gian từng giai đoạn (`asr_s`, `llm_s`, `tool_s`, `total_s`), tính bằng giây.
        tokens: số token prompt lớn nhất (`max_prompt`) và tổng số token sinh ra (`completion_total`).
        prompt_shrink_level: mức rút gọn prompt đã áp dụng; 0 là không rút gọn.
        transcript_source: nguồn transcript: `"gold"`, `"asr"` hoặc thuộc tính `source` của transcriber.
        traceback: traceback khi mẫu dừng vì exception.
    """

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
        """Chuyển thành dict để ghi trace; enum được đổi thành `.value`."""
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
    """Môi trường `zaic_tools` của một mẫu.

    Gồm bốn thao tác: `validate` kiểm tra tham số, `call` gọi tool, `get_tool_calls` trả về các lời gọi đã ghi nhận,
    `get_state_hash` trả về hash trạng thái để chấm điểm.
    """

    def validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
    def get_tool_calls(self) -> list[dict[str, Any]]: ...
    def get_state_hash(self) -> str: ...


class LLMBackend(Protocol):
    """Backend sinh văn bản theo ràng buộc JSON schema.

    `warmup` chạy vài lần sinh thử trước khi đo, để lần đo đầu tiên không bị chậm vì khởi động.
    """

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None,
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Sinh văn bản một lần cho `messages`, đầu ra tuân theo `schema`.

        Tham số:
            messages: danh sách tin nhắn của prompt.
            schema: JSON schema ràng buộc đầu ra.
            sample_id: mã mẫu, dùng cho log.
            time_left_s: số giây còn lại của mẫu; `None` là không giới hạn.

        Trả về:
            `Generation`; `timed_out=True` khi hết `time_left_s` giây và phải ngắt giữa chừng.
        """
        ...

    def warmup(self) -> None: ...


class TokenCounter(Protocol):
    """Đếm số token prompt của một danh sách tin nhắn."""

    def count(self, messages: list[dict[str, Any]]) -> int: ...


class Transcriber(Protocol):
    """Chuyển audio của mẫu thành văn bản; `warmup` chạy thử trước khi đo."""

    def transcribe(self, sample: Sample) -> Transcript: ...
    def warmup(self) -> None: ...


class Clock(Protocol):
    """Đồng hồ trả về số giây tăng đơn điệu; test thay bằng đồng hồ giả để điều khiển thời gian."""

    def __call__(self) -> float: ...


class Guard(Protocol):
    """Một khâu trong chuỗi guard, kiểm tra từng `Action` trước khi thực thi."""

    def check(self, action: Action, ctx: GuardContext) -> GuardResult: ...


class AgentObserver(Protocol):
    """Nhận sự kiện của agent: bắt đầu mẫu, mỗi bước, và khi dừng (ví dụ tracer ghi `trace.jsonl`)."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None: ...
    def on_step(self, step: StepRecord) -> None: ...
    def on_stop(self, result: SampleResult) -> None: ...

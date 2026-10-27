"""Vòng lặp agent cho một mẫu (SPEC 4.10).

Nội dung: lớp `Agent` chạy vòng LLM -> parse -> guard -> tool cho một mẫu, các hằng chuỗi phản hồi gửi lại
mô hình (`PARSE_ERROR_MSG`, `BLOCK_PREFIX`, `RESULT_PREFIX`) và hàm `transcript_source`.

Cách xây dựng:
- Luồng dữ liệu: ASR -> `PromptBuilder.start` dựng hội thoại đầu -> mỗi bước gọi LLM với JSON schema của mẫu
  -> `parse` -> guard -> env -> kết quả được nối vào hội thoại dưới dạng tin nhắn user -> lặp đến khi dừng.
- Hội thoại chỉ nối thêm (append-only): tiền tố prompt không bao giờ đổi giữa các bước, nhờ đó backend
  tái dùng được prefix cache.
- Các hằng chuỗi là một phần prompt và phải giống hệt từng byte giữa lúc sinh dữ liệu huấn luyện và lúc
  chạy thật; không sửa nội dung chúng.
- `Budget` quyết định có chạy bước tiếp; hạn chót được backend thực thi bằng cách ngắt bước đang chạy.
- Lỗi parse và BLOCK của guard tiêu một lần thử lại; hết lượt thì dừng với lý do tương ứng.
- Agent không bao giờ bắt `Exception`: lỗi lập trình phải nổi lên, không bị nuốt thành kết quả rỗng.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from lifemate.agent.budget import Budget
from lifemate.agent.guards import GuardChain
from lifemate.agent.observers import NullObserver
from lifemate.agent.prompt import Conversation, PromptBuilder, truncate_result
from lifemate.agent.protocol import output_schema, parse
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import (
    Action,
    AgentObserver,
    Clock,
    ErrorCode,
    GuardContext,
    GuardVerdict,
    LLMBackend,
    ParseError,
    Sample,
    SampleResult,
    StepRecord,
    StopReason,
    ToolEnvLike,
    Transcriber,
)

__all__ = ["Agent", "NullObserver", "transcript_source"]

log = structlog.get_logger(__name__)

# Ba chuỗi dưới đây là văn bản mô hình đọc được; phải trùng khớp với dữ liệu huấn luyện (SFT).
PARSE_ERROR_MSG = "LỖI: đầu ra không phải JSON hợp lệ, hãy trả về đúng một JSON"
BLOCK_PREFIX = "LỖI (chưa gọi tool): "
RESULT_PREFIX = "KẾT QUẢ: "


def transcript_source(transcriber: Transcriber) -> str:
    """Thuộc tính `source` của transcriber nếu có; nếu không thì "gold" cho GoldTranscriber, còn lại "asr"."""
    default = "gold" if isinstance(transcriber, GoldTranscriber) else "asr"
    return str(getattr(transcriber, "source", default))


@dataclass(slots=True)
class _Run:
    """Trạng thái thay đổi được của vòng lặp cho một mẫu, dùng chung với hàm xử lý lời gọi `_call`."""

    env: ToolEnvLike
    allowed: frozenset[str]
    conv: Conversation
    budget: Budget
    clock: Clock
    errors: list[ErrorCode]
    calls: list[Action] = field(default_factory=list)


class Agent:
    """Chạy vòng LLM / guard / tool cho một mẫu; không bao giờ bắt `Exception`.

    Args:
        llm: backend sinh văn bản.
        transcriber: chuyển audio của mẫu thành transcript (ASR hoặc gold).
        prompts: dựng hội thoại đầu và tập tool được phép.
        guards: chuỗi guard kiểm tra trước mỗi lời gọi tool.
        index: chỉ mục catalog, dùng để biết loại của từng tool.
        cfg: cấu hình agent (số bước, thử lại, hạn chót, độ dài kết quả).
        llm_cfg: cấu hình LLM (độ dài ngữ cảnh, số token sinh tối đa).
        observer: nhận sự kiện của vòng lặp; mặc định là `NullObserver`.
    """

    def __init__(
        self,
        llm: LLMBackend,
        transcriber: Transcriber,
        prompts: PromptBuilder,
        guards: GuardChain,
        index: CatalogIndex,
        cfg: AgentConfig,
        llm_cfg: LLMConfig,
        observer: AgentObserver | None = None,
    ) -> None:
        self.llm = llm
        self.transcriber = transcriber
        self.prompts = prompts
        self.guards = guards
        self.index = index
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.observer: AgentObserver = observer if observer is not None else NullObserver()
        self._tool_kind = index.tool_kinds()

    def _call(self, action: Action, base: dict[str, Any], run: _Run) -> tuple[StepRecord, StopReason | None, float]:
        """Cho một lời gọi qua chuỗi guard và, nếu không bị chặn, gửi tới env.

        Args:
            action: hành động loại `call` đã parse.
            base: các trường chung của `StepRecord` (chỉ số bước, token, thời gian LLM, văn bản thô).
            run: trạng thái vòng lặp của mẫu.

        Returns:
            Bộ ba (bản ghi bước, lý do dừng hoặc None, số giây tool chạy).
        """
        ctx = GuardContext(env=run.env, allowed=run.allowed, tool_kind=self._tool_kind, calls_so_far=tuple(run.calls))
        decision, warns = self.guards.check(action, ctx)
        warn_codes = tuple(w.code for w in warns if w.code is not None)
        run.errors.extend(warn_codes)
        if decision.verdict is GuardVerdict.BLOCK:
            # Bị chặn: lời gọi không tới env, mô hình nhận thông báo lỗi để sửa ở bước sau.
            if decision.code is not None:
                run.errors.append(decision.code)
            run.conv.append_user(BLOCK_PREFIX + (decision.message or ""))
            stop = None if run.budget.retry() else StopReason.RETRIES_EXHAUSTED
            step = StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes, error=decision.code)
            return step, stop, 0.0
        if decision.verdict is GuardVerdict.STOP:
            # STOP kết thúc mẫu như `done`; lời gọi này không được gửi.
            return StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes), StopReason.DONE, 0.0
        name, arguments = action.name or "", action.arguments or {}
        t0 = run.clock()
        result = run.env.call(name, arguments)
        spent = run.clock() - t0
        run.calls.append(action)
        # Kết quả bị cắt theo `max_result_chars` để hội thoại không phình quá ngữ cảnh.
        run.conv.append_user(RESULT_PREFIX + truncate_result(result, self.cfg.max_result_chars))
        verdict = GuardVerdict.WARN if warns else GuardVerdict.PASS
        step = StepRecord(**base, action=action, guard=verdict, warnings=warn_codes, result=result, tool_s=spent)
        # Sau khi hỏi lại người dùng thì mẫu kết thúc: câu trả lời nằm ở lượt sau, ngoài mẫu này.
        return step, (StopReason.ASKED if name == ASK_TOOL else None), spent

    def run(
        self,
        sample: Sample,
        env: ToolEnvLike,
        *,
        clock: Clock = time.monotonic,
        train_mode: bool = False,
    ) -> SampleResult:
        """Chạy vòng lặp cho một mẫu và trả về kết quả.

        Args:
            sample: mẫu cần xử lý.
            env: môi trường tool của mẫu; ghi lại các lời gọi và trạng thái.
            clock: đồng hồ đơn điệu, thay được trong test.
            train_mode: True khi chạy trên dữ liệu huấn luyện; các app held-out bị loại khỏi tập tool.

        Returns:
            `SampleResult` gồm lý do dừng, lỗi (không trùng), các bước, lời gọi tool, hash trạng thái,
            thời gian và số token.

        Raises:
            HeldOutLeak: khi `train_mode` và lịch sử của mẫu gọi tới app held-out (ném từ `PromptBuilder`).
        """
        # Budget tạo trước ASR nên thời gian chuyển giọng nói cũng tính vào hạn chót.
        budget = Budget(self.cfg, self.llm_cfg, clock)
        transcript = self.transcriber.transcribe(sample)
        conv, shrink_level = self.prompts.start(sample, transcript, train_mode=train_mode)
        allowed = frozenset(self.prompts.allowed_tools(sample, train_mode))
        schema = output_schema(allowed)
        self.observer.on_sample_start(sample, transcript)

        steps: list[StepRecord] = []
        errors: list[ErrorCode] = []
        run = _Run(env=env, allowed=allowed, conv=conv, budget=budget, clock=clock, errors=errors)
        llm_s = tool_s = 0.0
        max_prompt = completion_total = 0
        stop: StopReason | None = None

        while stop is None:
            max_prompt = max(max_prompt, conv.prompt_tokens())
            reason = budget.before_step(conv.prompt_tokens())
            if reason is not None:
                stop = reason
                break
            gen = self.llm.generate(conv.as_list(), schema, sample_id=sample.id, time_left_s=budget.remaining)
            budget.after_step(gen.seconds)
            llm_s += gen.seconds
            completion_total += gen.completion_tokens
            max_prompt = max(max_prompt, gen.prompt_tokens)
            if gen.timed_out:
                # Hết hạn giữa bước: văn bản dở dang không phải một hành động. Các lời gọi đã thực hiện vẫn giữ.
                steps.append(
                    StepRecord(
                        i=len(steps),
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        llm_s=gen.seconds,
                        raw=gen.text,
                        action=None,
                        guard=None,
                    )
                )
                self.observer.on_step(steps[-1])
                stop = StopReason.TIMEOUT
                break
            conv.reconcile(gen.prompt_tokens)  # ghi log prompt_tokens_mismatch khi hai số đếm khác nhau
            # Nối nguyên văn đầu ra (kể cả khi hỏng) để hội thoại khớp đúng những gì mô hình đã sinh.
            conv.append_assistant(gen.text, gen.completion_tokens)
            base: dict[str, Any] = {
                "i": len(steps),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "llm_s": gen.seconds,
                "raw": gen.text,
            }
            parsed = parse(gen.text)
            if isinstance(parsed, ParseError):
                step = StepRecord(**base, action=None, guard=None, error=ErrorCode.PARSE_ERROR)
                errors.append(ErrorCode.PARSE_ERROR)
                conv.append_user(PARSE_ERROR_MSG)
                if not budget.retry():
                    stop = StopReason.PARSE_ERROR
            elif parsed.kind == "done":
                step = StepRecord(**base, action=parsed, guard=None)
                stop = StopReason.DONE
            else:
                step, stop, spent = self._call(parsed, base, run)
                tool_s += spent
            steps.append(step)
            self.observer.on_step(step)

        sample_result = SampleResult(
            id=sample.id,
            stop_reason=stop,
            # Bỏ mã lỗi trùng nhưng giữ thứ tự xuất hiện đầu tiên.
            errors=tuple(dict.fromkeys(errors)),
            transcript=transcript,
            steps=tuple(steps),
            tool_calls=tuple(env.get_tool_calls()),
            state_hash=env.get_state_hash(),
            timing={"asr_s": transcript.seconds, "llm_s": llm_s, "tool_s": tool_s, "total_s": budget.elapsed},
            tokens={"max_prompt": max_prompt, "completion_total": completion_total},
            prompt_shrink_level=shrink_level,
            transcript_source=transcript_source(self.transcriber),
        )
        self.observer.on_stop(sample_result)
        return sample_result

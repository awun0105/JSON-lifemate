"""Vòng lặp agent ở định dạng ChatML (tool calling có sẵn của Qwen), cho `[agent].format = "chatml"`.

Nội dung chính:
- `ChatMLAgent`: cùng giao diện và cùng hành vi với `lifemate.agent.agent.Agent` (ngân sách, guard, hạn giờ, kết quả),
  chỉ khác định dạng hội thoại: dựng bằng `lifemate.agent.chatml`, đọc lời gọi từ khối `<tool_call>`.

Cách hoạt động (mỗi bước):
1. `Budget.before_step` kiểm tra số bước, độ dài context, hạn giờ.
2. Gọi LLM không ép schema (`schema=None`); danh sách tool đi kèm qua chat template.
3. `parse_reply`:
   - không có `<tool_call>`: mô hình trả lời bằng lời, mẫu kết thúc (`DONE`);
   - có lời gọi: nối một tin `assistant` có `tool_calls` (tham số viết lại bằng `json.dumps` như dữ liệu của lead),
     rồi với từng lời gọi chạy chuỗi guard. BLOCK: không gửi tới env, tin `tool` là `{"ok": false, "error": <lý do>}`
     (đúng dạng lượt "phục hồi lỗi" trong dữ liệu của lead) và tốn một lượt thử lại. PASS/WARN: `env.call`, tin `tool`
     là kết quả. Gọi `sys.ask_missing_info` thì dừng (`ASKED`);
   - khối `<tool_call>` hỏng: nối nguyên văn đầu ra và một tin `user` báo lỗi, tốn một lượt thử lại.

Lưu ý:
- Tin nhắn giữ trường `tools` trong tin `system` như dữ liệu của lead; backend LLM và bộ đếm token tách trường này ra để
  đưa vào chat template (`lifemate.core.chat.split_tools`).
- Mỗi lời gọi trong một lượt có id `call_curr_<k>`, `k` đếm tiếp qua các lượt, giống `build_messages` của lead.
- Agent không bắt `Exception`; `Runner.run_one` là ranh giới lỗi theo mẫu.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lifemate.agent.agent import PARSE_ERROR_MSG, transcript_source
from lifemate.agent.budget import Budget
from lifemate.agent.chatml import ChatMLPrompt, arguments_text, parse_reply, tool_content
from lifemate.agent.guards import GuardChain
from lifemate.agent.observers import NullObserver
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import (
    Action,
    AgentObserver,
    Clock,
    ErrorCode,
    Generation,
    GuardContext,
    GuardVerdict,
    LLMBackend,
    ParseError,
    Sample,
    SampleResult,
    StepRecord,
    StopReason,
    TokenCounter,
    ToolEnvLike,
    Transcriber,
)


@dataclass(slots=True)
class _Run:
    """Trạng thái thay đổi được của vòng lặp với một mẫu.

    Thuộc tính:
        env: môi trường tool của mẫu.
        allowed: tên các tool được phép.
        messages: hội thoại hiện tại (chỉ nối thêm).
        budget: ngân sách của mẫu.
        clock: đồng hồ đo thời gian chạy tool.
        steps: bản ghi các bước; một bước có nhiều lời gọi thì mỗi lời gọi một bản ghi cùng chỉ số `i`.
        errors: các mã lỗi gặp tới giờ (có thể trùng).
        sent: các lời gọi đã gửi tới env.
        n_steps: số bước LLM đã chạy.
        n_calls: số lời gọi đã nhận từ mô hình, để đặt id `call_curr_<k>`.
        tool_s: tổng số giây chạy tool.
    """

    env: ToolEnvLike
    allowed: frozenset[str]
    messages: list[dict[str, Any]]
    budget: Budget
    clock: Clock
    steps: list[StepRecord] = field(default_factory=list)
    errors: list[ErrorCode] = field(default_factory=list)
    sent: list[Action] = field(default_factory=list)
    n_steps: int = 0
    n_calls: int = 0
    tool_s: float = 0.0


class ChatMLAgent:
    """Chạy vòng LLM / guard / tool cho một mẫu ở định dạng ChatML; không bao giờ bắt `Exception`.

    Tham số:
        llm: backend sinh văn bản.
        transcriber: chuyển audio của mẫu thành chữ.
        prompts: dựng hội thoại đầu tiên và tập tool được phép.
        guards: chuỗi guard kiểm tra trước mỗi lời gọi tool.
        index: chỉ mục catalog, dùng để biết loại của từng tool.
        counter: bộ đếm token của prompt.
        cfg: config agent.
        llm_cfg: config LLM.
        observer: nhận sự kiện của vòng lặp; mặc định là `NullObserver`.
    """

    def __init__(
        self,
        llm: LLMBackend,
        transcriber: Transcriber,
        prompts: ChatMLPrompt,
        guards: GuardChain,
        index: CatalogIndex,
        counter: TokenCounter,
        cfg: AgentConfig,
        llm_cfg: LLMConfig,
        observer: AgentObserver | None = None,
    ) -> None:
        self.llm = llm
        self.transcriber = transcriber
        self.prompts = prompts
        self.guards = guards
        self.index = index
        self.counter = counter
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.observer: AgentObserver = observer if observer is not None else NullObserver()
        self._tool_kind = index.tool_kinds()

    def _record(self, run: _Run, step: StepRecord) -> None:
        """Lưu một bản ghi bước và báo cho observer."""
        run.steps.append(step)
        self.observer.on_step(step)

    def _call(self, cid: str, action: Action, base: dict[str, Any], run: _Run) -> StopReason | None:
        """Cho một lời gọi qua chuỗi guard, gửi tới env nếu không bị chặn, và nối tin `tool` tương ứng.

        Trả về:
            Lý do dừng (`RETRIES_EXHAUSTED`, `DONE` khi guard STOP, `ASKED`) hoặc `None` để chạy tiếp.
        """
        ctx = GuardContext(env=run.env, allowed=run.allowed, tool_kind=self._tool_kind, calls_so_far=tuple(run.sent))
        decision, warns = self.guards.check(action, ctx)
        warn_codes = tuple(w.code for w in warns if w.code is not None)
        run.errors.extend(warn_codes)
        name, arguments = action.name or "", action.arguments or {}
        if decision.verdict is GuardVerdict.BLOCK:
            # Bị chặn: lời gọi không tới env; mô hình đọc lỗi trong tin `tool` và sửa ở bước sau.
            if decision.code is not None:
                run.errors.append(decision.code)
            error = {"ok": False, "error": decision.message or ""}
            run.messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": tool_content(error)})
            step = StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes, error=decision.code)
            self._record(run, step)
            return None if run.budget.retry() else StopReason.RETRIES_EXHAUSTED
        if decision.verdict is GuardVerdict.STOP:
            self._record(run, StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes))
            return StopReason.DONE
        t0 = run.clock()
        result = run.env.call(name, arguments)
        spent = run.clock() - t0
        run.tool_s += spent
        run.sent.append(action)
        run.messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": tool_content(result)})
        verdict = GuardVerdict.WARN if warns else GuardVerdict.PASS
        self._record(
            run, StepRecord(**base, action=action, guard=verdict, warnings=warn_codes, result=result, tool_s=spent)
        )
        # Hỏi lại người dùng xong thì mẫu kết thúc: câu trả lời nằm ở lượt sau, ngoài phạm vi mẫu này.
        return StopReason.ASKED if name == ASK_TOOL else None

    def _turn(self, gen: Generation, run: _Run) -> StopReason | None:
        """Xử lý một lượt trả lời đã sinh xong (không hết giờ); trả lý do dừng hoặc `None`."""
        base: dict[str, Any] = {
            "i": run.n_steps - 1,
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "llm_s": gen.seconds,
            "raw": gen.text,
        }
        parsed = parse_reply(gen.text)
        if isinstance(parsed, ParseError):
            run.errors.append(ErrorCode.PARSE_ERROR)
            run.messages.append({"role": "assistant", "content": gen.text})
            run.messages.append({"role": "user", "content": PARSE_ERROR_MSG})
            self._record(run, StepRecord(**base, action=None, guard=None, error=ErrorCode.PARSE_ERROR))
            return None if run.budget.retry() else StopReason.PARSE_ERROR
        if parsed is None:
            # Câu kết bằng lời: lượt của mẫu xong.
            run.messages.append({"role": "assistant", "content": gen.text})
            self._record(run, StepRecord(**base, action=Action(kind="done", name=None, arguments=None), guard=None))
            return StopReason.DONE
        ids = [f"call_curr_{run.n_calls + k}" for k in range(len(parsed))]
        run.n_calls += len(parsed)
        calls = [
            {
                "id": cid,
                "type": "function",
                "function": {"name": a.name or "", "arguments": arguments_text(a.arguments)},
            }
            for cid, a in zip(ids, parsed, strict=True)
        ]
        run.messages.append({"role": "assistant", "tool_calls": calls})
        for cid, action in zip(ids, parsed, strict=True):
            stop = self._call(cid, action, base, run)
            if stop is not None:
                return stop
        return None

    def run(
        self,
        sample: Sample,
        env: ToolEnvLike,
        *,
        clock: Clock = time.monotonic,
        train_mode: bool = False,
    ) -> SampleResult:
        """Chạy vòng lặp cho một mẫu và trả về kết quả (cùng ý nghĩa các trường với `Agent.run`)."""
        budget = Budget(self.cfg, self.llm_cfg, clock)  # bấm giờ trước ASR: thời gian ASR tính vào hạn giờ
        transcript = self.transcriber.transcribe(sample)
        messages, shrink_level = self.prompts.start(sample, transcript, train_mode=train_mode)
        allowed = frozenset(self.prompts.allowed_tools(sample, train_mode))
        self.observer.on_sample_start(sample, transcript)
        run = _Run(env=env, allowed=allowed, messages=messages, budget=budget, clock=clock)
        llm_s = 0.0
        max_prompt = completion_total = 0
        stop: StopReason | None = None
        while stop is None:
            prompt_tokens = self.counter.count(run.messages)
            max_prompt = max(max_prompt, prompt_tokens)
            stop = budget.before_step(prompt_tokens)
            if stop is not None:
                break
            gen = self.llm.generate(run.messages, None, sample_id=sample.id, time_left_s=budget.remaining)
            budget.after_step(gen.seconds)
            run.n_steps += 1
            llm_s += gen.seconds
            completion_total += gen.completion_tokens
            max_prompt = max(max_prompt, gen.prompt_tokens)
            if gen.timed_out:
                # Hết giờ giữa bước: văn bản dở dang không phải một hành động; các lời gọi đã gửi vẫn được nộp.
                step = StepRecord(
                    i=run.n_steps - 1,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    llm_s=gen.seconds,
                    raw=gen.text,
                    action=None,
                    guard=None,
                )
                self._record(run, step)
                stop = StopReason.TIMEOUT
                break
            stop = self._turn(gen, run)

        sample_result = SampleResult(
            id=sample.id,
            stop_reason=stop,
            errors=tuple(dict.fromkeys(run.errors)),
            transcript=transcript,
            steps=tuple(run.steps),
            tool_calls=tuple(env.get_tool_calls()),
            state_hash=env.get_state_hash(),
            timing={"asr_s": transcript.seconds, "llm_s": llm_s, "tool_s": run.tool_s, "total_s": budget.elapsed},
            tokens={"max_prompt": max_prompt, "completion_total": completion_total},
            prompt_shrink_level=shrink_level,
            transcript_source=transcript_source(self.transcriber),
        )
        self.observer.on_stop(sample_result)
        return sample_result

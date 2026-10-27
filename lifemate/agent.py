"""Agent loop for one sample (SPEC 4.10)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from lifemate.asr import GoldTranscriber
from lifemate.budget import Budget
from lifemate.config import AgentConfig, LLMConfig
from lifemate.guards import GuardChain
from lifemate.prompt import Conversation, PromptBuilder, truncate_result
from lifemate.protocol import output_schema, parse
from lifemate.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.types import (
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
    Transcript,
)

log = structlog.get_logger(__name__)

PARSE_ERROR_MSG = "LỖI: đầu ra không phải JSON hợp lệ, hãy trả về đúng một JSON"
BLOCK_PREFIX = "LỖI (chưa gọi tool): "
RESULT_PREFIX = "KẾT QUẢ: "


class NullObserver:
    """Observer that ignores every event."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Ignore."""

    def on_step(self, step: StepRecord) -> None:
        """Ignore."""

    def on_stop(self, result: SampleResult) -> None:
        """Ignore."""


def transcript_source(transcriber: Transcriber) -> str:
    """`source` attribute of the transcriber if any; "gold" for GoldTranscriber, else "asr"."""
    default = "gold" if isinstance(transcriber, GoldTranscriber) else "asr"
    return str(getattr(transcriber, "source", default))


@dataclass(slots=True)
class _Run:
    """Mutable per-sample loop state shared with the call handler."""

    env: ToolEnvLike
    allowed: frozenset[str]
    conv: Conversation
    budget: Budget
    clock: Clock
    errors: list[ErrorCode]
    calls: list[Action] = field(default_factory=list)


class Agent:
    """Runs the LLM / guard / tool loop for one sample; never catches Exception."""

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
        """Guard one call and, unless blocked, send it to the env; returns (step, stop reason, tool seconds)."""
        ctx = GuardContext(env=run.env, allowed=run.allowed, tool_kind=self._tool_kind, calls_so_far=tuple(run.calls))
        decision, warns = self.guards.check(action, ctx)
        warn_codes = tuple(w.code for w in warns if w.code is not None)
        run.errors.extend(warn_codes)
        if decision.verdict is GuardVerdict.BLOCK:
            if decision.code is not None:
                run.errors.append(decision.code)
            run.conv.append_user(BLOCK_PREFIX + (decision.message or ""))
            stop = None if run.budget.retry() else StopReason.RETRIES_EXHAUSTED
            step = StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes, error=decision.code)
            return step, stop, 0.0
        if decision.verdict is GuardVerdict.STOP:
            return StepRecord(**base, action=action, guard=decision.verdict, warnings=warn_codes), StopReason.DONE, 0.0
        name, arguments = action.name or "", action.arguments or {}
        t0 = run.clock()
        result = run.env.call(name, arguments)
        spent = run.clock() - t0
        run.calls.append(action)
        run.conv.append_user(RESULT_PREFIX + truncate_result(result, self.cfg.max_result_chars))
        verdict = GuardVerdict.WARN if warns else GuardVerdict.PASS
        step = StepRecord(**base, action=action, guard=verdict, warnings=warn_codes, result=result, tool_s=spent)
        return step, (StopReason.ASKED if name == ASK_TOOL else None), spent

    def run(
        self,
        sample: Sample,
        env: ToolEnvLike,
        *,
        step_estimate_s: float | None = None,
        clock: Clock = time.monotonic,
        train_mode: bool = False,
    ) -> SampleResult:
        """Run the loop for one sample and return its result."""
        budget = Budget(self.cfg, self.llm_cfg, clock, step_estimate_s=step_estimate_s)
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
            gen = self.llm.generate(conv.as_list(), schema, sample_id=sample.id)
            budget.after_step(gen.seconds)
            llm_s += gen.seconds
            completion_total += gen.completion_tokens
            max_prompt = max(max_prompt, gen.prompt_tokens)
            conv.reconcile(gen.prompt_tokens)  # logs prompt_tokens_mismatch on a difference
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

"""Agent loop end to end with FakeLLM / FakeEnv (SPEC 4.10; T5, T10 at Agent level)."""

from __future__ import annotations

import dataclasses
from typing import Any

from lifemate.agent.agent import Agent, NullObserver
from lifemate.agent.guards import build_guard_chain
from lifemate.agent.prompt import PromptBuilder
from lifemate.backends.asr import GoldTranscriber
from lifemate.backends.llm import FakeLLM
from lifemate.core.types import ErrorCode, GuardVerdict, Sample, SampleResult, StepRecord, StopReason, Transcript
from tests.fakes import FakeClock, FakeEnv, WhitespaceTokenCounter
from tests.test_budget import AGENT_CFG, LLM_CFG
from tests.test_guards import ASK_TOOL, INDEX, INSTALLED, make_env

TEMPLATE = 'Quy tắc: trả {"action":"done"} khi xong.\n\n{tool_docs}\n'
LLM = dataclasses.replace(LLM_CFG, max_model_len=100_000)


def sample(sid: str, calls: list[dict[str, Any]]) -> Sample:
    return Sample(
        id=sid,
        split="train",
        audio="a.wav",
        audio_text="xoá báo thức a1",
        context={"current_time": "2025-01-01T08:00:00", "installed_apps": INSTALLED},
        history=(),
        label={"tool_calls": calls},
    )


class Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.steps: list[StepRecord] = []

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        self.events.append("start")

    def on_step(self, step: StepRecord) -> None:
        self.events.append("step")
        self.steps.append(step)

    def on_stop(self, result: SampleResult) -> None:
        self.events.append("stop")


def make_agent(llm: FakeLLM, observer: Recorder | None = None, cfg: Any = AGENT_CFG) -> Agent:
    prompts = PromptBuilder(
        TEMPLATE,
        INDEX,
        WhitespaceTokenCounter(),
        max_model_len=LLM.max_model_len,
        max_tokens=LLM.max_tokens,
        max_result_chars=cfg.max_result_chars,
    )
    chain = build_guard_chain(cfg.guards)
    return Agent(llm, GoldTranscriber(), prompts, chain, INDEX, cfg, LLM, observer)


GOLD = [{"name": "alarm.list", "arguments": {}}, {"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}]


def test_replay_two_calls_done() -> None:
    s = sample("s1", GOLD)
    rec = Recorder()
    env = make_env()
    res = make_agent(FakeLLM([s]), rec).run(s, env, clock=FakeClock())
    assert res.stop_reason is StopReason.DONE
    assert list(res.tool_calls) == GOLD
    assert res.errors == () and res.transcript_source == "gold"
    assert rec.events == ["start", "step", "step", "step", "stop"]
    assert set(res.timing) == {"asr_s", "llm_s", "tool_s", "total_s"}
    assert res.tokens["completion_total"] > 0 and res.tokens["max_prompt"] > 0


def test_ask_stops_even_if_llm_has_more() -> None:
    calls = [{"name": ASK_TOOL, "arguments": {"question": "báo thức nào?"}}, GOLD[1]]
    s = sample("s2", calls)
    env = make_env()
    res = make_agent(FakeLLM([s])).run(s, env, clock=FakeClock())
    assert res.stop_reason is StopReason.ASKED
    assert list(res.tool_calls) == calls[:1]


def test_parse_error_then_recovery() -> None:
    s = sample("s3", GOLD)
    llm = FakeLLM([s], broken_outputs={"s3": ["không phải json", "{oops"]})
    res = make_agent(llm).run(s, make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.DONE and list(res.tool_calls) == GOLD
    assert res.errors == (ErrorCode.PARSE_ERROR,)
    assert [st.error for st in res.steps[:2]] == [ErrorCode.PARSE_ERROR] * 2


def test_parse_error_exhausts_retries() -> None:
    s = sample("s4", GOLD)
    llm = FakeLLM([s], broken_outputs={"s4": ["x", "y", "z"]})
    res = make_agent(llm).run(s, make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.PARSE_ERROR and res.tool_calls == ()


def test_blocked_call_never_reaches_env() -> None:
    calls = [{"name": "bank.pay", "arguments": {}}, {"name": "alarm.delete", "arguments": {}}, GOLD[0]]
    s = sample("s5", calls)
    env = make_env()
    res = make_agent(FakeLLM([s])).run(s, env, clock=FakeClock())
    assert list(res.tool_calls) == [GOLD[0]] and env.calls == [GOLD[0]]
    assert res.errors == (ErrorCode.UNKNOWN_TOOL, ErrorCode.INVALID_ARGS)
    assert [st.guard for st in res.steps[:2]] == [GuardVerdict.BLOCK] * 2
    assert res.stop_reason is StopReason.DONE


def test_blocks_exhaust_retries() -> None:
    s = sample("s6", [{"name": "bank.pay", "arguments": {}}] * 3)
    res = make_agent(FakeLLM([s])).run(s, make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.RETRIES_EXHAUSTED and res.tool_calls == ()


def test_duplicate_write_proceeds() -> None:
    s = sample("s7", [GOLD[1], GOLD[1]])
    res = make_agent(FakeLLM([s])).run(s, make_env(), clock=FakeClock())
    assert list(res.tool_calls) == [GOLD[1], GOLD[1]]
    assert res.errors == (ErrorCode.DUPLICATE_WRITE,)
    assert res.steps[1].guard is GuardVerdict.WARN


def test_timeout_at_agent_level() -> None:
    """T10: 5 s steps, deadline 29 -> 5 steps finish (25 s), the 6th starts and is cut at 29 s;
    the 5 calls already made are kept and a result is still produced."""
    calls = [{"name": "notes.search", "arguments": {"q": str(i)}} for i in range(8)]
    s = sample("s8", calls)
    clock = FakeClock()
    rec = Recorder()
    res = make_agent(FakeLLM([s], delay_s=5, clock=clock), rec).run(s, make_env(), clock=clock)
    assert res.stop_reason is StopReason.TIMEOUT
    assert len(res.steps) == 6 and res.steps[-1].action is None and clock() == 29
    assert list(res.tool_calls) == calls[:5]
    assert rec.events[-1] == "stop"


def test_context_overflow_stops() -> None:
    s = sample("s9", GOLD)
    agent = make_agent(FakeLLM([s]))
    agent.llm_cfg = dataclasses.replace(LLM, max_model_len=150)
    res = agent.run(s, make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.CONTEXT_OVERFLOW and res.steps == ()


def test_null_observer_default() -> None:
    s = sample("s10", GOLD)
    agent = make_agent(FakeLLM([s]))
    assert isinstance(agent.observer, NullObserver)
    assert isinstance(make_env(), FakeEnv)

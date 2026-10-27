"""T10 at Budget level: predictive deadline, steps, context, retries (SPEC 4.10)."""

from __future__ import annotations

import dataclasses

from lifemate.agent.budget import Budget
from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import StopReason
from tests.fakes import FakeClock

AGENT_CFG = AgentConfig(
    max_steps=10,
    max_retries=2,
    hard_limit_s=28.0,
    initial_step_estimate_s=5.0,
    max_result_chars=4000,
    system_prompt="",
    guards=["allowed_tool", "schema", "duplicate_write"],
)
LLM_CFG = LLMConfig(
    backend="fake",
    model="",
    revision="",
    dtype="auto",
    max_model_len=1000,
    gpu_memory_utilization=0.9,
    seed=0,
    max_tokens=100,
    batch_invariant=False,
    warmup_runs=0,
)


def test_timeout_before_sixth_step() -> None:
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock, step_estimate_s=None)
    ran = 0
    while b.before_step(10) is None:
        clock.advance(5)
        b.after_step(5)
        ran += 1
    assert ran == 5 and b.elapsed == 25
    assert b.before_step(10) is StopReason.TIMEOUT


def test_last_step_larger_than_estimate() -> None:
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock, step_estimate_s=1.0)
    clock.advance(15)
    b.after_step(15)
    assert b.before_step(10) is StopReason.TIMEOUT  # 15 + max(15, 1) > 28


def test_max_steps() -> None:
    b = Budget(dataclasses.replace(AGENT_CFG, max_steps=2), LLM_CFG, FakeClock())
    b.after_step(0)
    assert b.before_step(10) is None
    b.after_step(0)
    assert b.steps == 2 and b.before_step(10) is StopReason.MAX_STEPS


def test_context_overflow() -> None:
    b = Budget(AGENT_CFG, LLM_CFG, FakeClock())
    assert b.before_step(900) is None
    assert b.before_step(901) is StopReason.CONTEXT_OVERFLOW


def test_retry_counting() -> None:
    b = Budget(AGENT_CFG, LLM_CFG, FakeClock())
    assert [b.retry(), b.retry(), b.retry()] == [True, True, False]

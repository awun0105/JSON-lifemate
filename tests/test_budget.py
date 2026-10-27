"""T10 at Budget level: real deadline, steps, context, retries (SPEC 4.10)."""

from __future__ import annotations

import dataclasses

from lifemate.agent.budget import Budget
from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import StopReason
from tests.fakes import FakeClock

AGENT_CFG = AgentConfig(
    max_steps=10,
    max_retries=2,
    deadline_s=29.0,
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


def test_steps_run_until_the_real_deadline() -> None:
    """Catches: stopping early on a predicted step time; any step may start while time is left."""
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock)
    clock.advance(28.5)
    assert b.before_step(10) is None and b.remaining == 0.5
    clock.advance(0.5)
    assert b.before_step(10) is StopReason.TIMEOUT and b.remaining == 0.0


def test_slow_first_step_does_not_block_later_steps() -> None:
    """Catches the Kaggle T4 failure: a 14 s first step made the old predictor refuse a 3 s second step."""
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock)
    clock.advance(16)  # ASR 2 s + first step 14 s
    b.after_step(14)
    assert b.before_step(10) is None


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

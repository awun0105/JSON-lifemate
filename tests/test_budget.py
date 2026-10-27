"""Kiểm thử T10 ở mức Budget: hạn chót thật, số bước, ngữ cảnh, số lần thử lại (SPEC 4.10).

Dùng FakeClock để điều khiển thời gian; AGENT_CFG và LLM_CFG ở đây được các file test khác dùng lại.
Không cần dữ liệu cuộc thi hay tokenizer.
"""

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
    """Bắt lỗi: dừng sớm theo thời gian bước dự đoán; bước nào cũng được bắt đầu khi còn thời gian."""
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock)
    clock.advance(28.5)
    assert b.before_step(10) is None and b.remaining == 0.5
    clock.advance(0.5)
    assert b.before_step(10) is StopReason.TIMEOUT and b.remaining == 0.0


def test_slow_first_step_does_not_block_later_steps() -> None:
    """Bắt lỗi từng gặp trên Kaggle T4: bước đầu 14 s khiến bộ dự đoán cũ từ chối bước thứ hai chỉ 3 s."""
    clock = FakeClock()
    b = Budget(AGENT_CFG, LLM_CFG, clock)
    clock.advance(16)  # ASR 2 s + bước đầu 14 s
    b.after_step(14)
    assert b.before_step(10) is None


def test_max_steps() -> None:
    """Bảo vệ: đủ `max_steps` bước thì dừng với MAX_STEPS."""
    b = Budget(dataclasses.replace(AGENT_CFG, max_steps=2), LLM_CFG, FakeClock())
    b.after_step(0)
    assert b.before_step(10) is None
    b.after_step(0)
    assert b.steps == 2 and b.before_step(10) is StopReason.MAX_STEPS


def test_context_overflow() -> None:
    """Bảo vệ: prompt cộng `max_tokens` vượt `max_model_len` (1000) thì dừng với CONTEXT_OVERFLOW."""
    b = Budget(AGENT_CFG, LLM_CFG, FakeClock())
    # 900 + 100 token sinh = 1000 vừa khít; 901 vượt một token.
    assert b.before_step(900) is None
    assert b.before_step(901) is StopReason.CONTEXT_OVERFLOW


def test_retry_counting() -> None:
    """Bảo vệ: `max_retries=2` cho phép đúng hai lần thử lại, lần thứ ba bị từ chối."""
    b = Budget(AGENT_CFG, LLM_CFG, FakeClock())
    assert [b.retry(), b.retry(), b.retry()] == [True, True, False]

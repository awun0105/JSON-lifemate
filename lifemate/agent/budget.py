"""Per-sample step, retry, context and deadline budget (SPEC 4.10).

The deadline is real, not predicted: a step is started whenever time is left, and the LLM backend aborts
the running step when the deadline arrives (`LLMBackend.generate(time_left_s=...)`). Tool calls already
sent to the env stay in the submission.
"""

from __future__ import annotations

from lifemate.core.config import AgentConfig, LLMConfig
from lifemate.core.types import Clock, StopReason


class Budget:
    """Starts timing on construction (before ASR); decides whether the next step may start."""

    def __init__(self, cfg: AgentConfig, llm_cfg: LLMConfig, clock: Clock) -> None:
        self._cfg = cfg
        self._llm_cfg = llm_cfg
        self._clock = clock
        self._start = clock()
        self.steps = 0
        self.retries = 0

    @property
    def elapsed(self) -> float:
        """Seconds since the budget started."""
        return self._clock() - self._start

    @property
    def remaining(self) -> float:
        """Seconds left before the deadline (never negative)."""
        return max(0.0, self._cfg.deadline_s - self.elapsed)

    def before_step(self, prompt_tokens: int) -> StopReason | None:
        """Reason to stop before the next step, or None to go on."""
        if self.steps >= self._cfg.max_steps:
            return StopReason.MAX_STEPS
        if prompt_tokens + self._llm_cfg.max_tokens > self._llm_cfg.max_model_len:
            return StopReason.CONTEXT_OVERFLOW
        if self.remaining <= 0:
            return StopReason.TIMEOUT
        return None

    def after_step(self, seconds: float) -> None:
        """Record one finished (or aborted) step."""
        self.steps += 1

    def retry(self) -> bool:
        """Use one retry; False when `cfg.max_retries` are already used."""
        if self.retries >= self._cfg.max_retries:
            return False
        self.retries += 1
        return True

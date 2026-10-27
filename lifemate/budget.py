"""Per-sample step, retry, context and predictive time budget (SPEC 4.10)."""

from __future__ import annotations

from lifemate.config import AgentConfig, LLMConfig
from lifemate.types import Clock, StopReason


class Budget:
    """Starts timing on construction (before ASR); decides whether the next step may run."""

    def __init__(
        self, cfg: AgentConfig, llm_cfg: LLMConfig, clock: Clock, *, step_estimate_s: float | None = None
    ) -> None:
        self._cfg = cfg
        self._llm_cfg = llm_cfg
        self._clock = clock
        self._start = clock()
        self._estimate = step_estimate_s if step_estimate_s is not None else cfg.initial_step_estimate_s
        self._last_step_s = 0.0
        self.steps = 0
        self.retries = 0

    @property
    def elapsed(self) -> float:
        """Seconds since the budget started."""
        return self._clock() - self._start

    def before_step(self, prompt_tokens: int) -> StopReason | None:
        """Reason to stop before the next step, or None to go on."""
        if self.steps >= self._cfg.max_steps:
            return StopReason.MAX_STEPS
        if prompt_tokens + self._llm_cfg.max_tokens > self._llm_cfg.max_model_len:
            return StopReason.CONTEXT_OVERFLOW
        if self.elapsed + max(self._last_step_s, self._estimate) > self._cfg.hard_limit_s:
            return StopReason.TIMEOUT
        return None

    def after_step(self, seconds: float) -> None:
        """Record one finished step and its duration."""
        self.steps += 1
        self._last_step_s = seconds

    def retry(self) -> bool:
        """Use one retry; False when `cfg.max_retries` are already used."""
        if self.retries >= self._cfg.max_retries:
            return False
        self.retries += 1
        return True

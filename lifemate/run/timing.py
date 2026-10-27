"""Timing decorators for LLM backends and transcribers (SPEC 4.0, Decorator)."""

from __future__ import annotations

import time
from typing import Any

from lifemate.backends.asr import GoldTranscriber
from lifemate.core.types import Generation, LLMBackend, Sample, Transcriber, Transcript


class TimedLLM:
    """Wraps an LLMBackend; records wall seconds of every `generate` call."""

    def __init__(self, inner: LLMBackend) -> None:
        self.inner = inner
        self.durations: list[float] = []

    @property
    def calls(self) -> int:
        """Number of generate calls."""
        return len(self.durations)

    @property
    def seconds(self) -> float:
        """Total generate seconds."""
        return sum(self.durations)

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Delegate and time."""
        t0 = time.perf_counter()
        try:
            return self.inner.generate(messages, schema, sample_id=sample_id, time_left_s=time_left_s)
        finally:
            self.durations.append(time.perf_counter() - t0)

    def warmup(self) -> None:
        """Delegate (not recorded in `durations`)."""
        self.inner.warmup()


class TimedTranscriber:
    """Wraps a Transcriber; accumulates seconds and call count of `transcribe`."""

    def __init__(self, inner: Transcriber) -> None:
        self.inner = inner
        self.calls = 0
        self.seconds = 0.0
        self.source: str = str(getattr(inner, "source", "gold" if isinstance(inner, GoldTranscriber) else "asr"))

    def transcribe(self, sample: Sample) -> Transcript:
        """Delegate and time."""
        t0 = time.perf_counter()
        try:
            return self.inner.transcribe(sample)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t0

    def warmup(self) -> None:
        """Delegate."""
        self.inner.warmup()

"""Lớp bọc đo thời gian cho backend LLM và transcriber (SPEC 4.0, mẫu Decorator).

Hai lớp bọc giữ nguyên giao diện của đối tượng bên trong và chỉ ghi thời gian thực (wall clock) của mỗi lần gọi;
thời gian được ghi trong `finally`, nên lần gọi lỗi vẫn được tính. Lời gọi `warmup` không được tính.
"""

from __future__ import annotations

import time
from typing import Any

from lifemate.backends.asr import GoldTranscriber
from lifemate.core.types import Generation, LLMBackend, Sample, Transcriber, Transcript


class TimedLLM:
    """Bọc một LLMBackend; ghi số giây thực của mọi lần gọi `generate`."""

    def __init__(self, inner: LLMBackend) -> None:
        self.inner = inner
        self.durations: list[float] = []

    @property
    def calls(self) -> int:
        """Số lần gọi generate."""
        return len(self.durations)

    @property
    def seconds(self) -> float:
        """Tổng số giây của các lần generate."""
        return sum(self.durations)

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Chuyển lời gọi xuống đối tượng bên trong và đo thời gian."""
        t0 = time.perf_counter()
        try:
            return self.inner.generate(messages, schema, sample_id=sample_id, time_left_s=time_left_s)
        finally:
            self.durations.append(time.perf_counter() - t0)

    def warmup(self) -> None:
        """Chuyển lời gọi xuống (không ghi vào `durations`)."""
        self.inner.warmup()


class TimedTranscriber:
    """Bọc một Transcriber; cộng dồn số giây và số lần gọi `transcribe`.

    `source` lấy từ thuộc tính của đối tượng bên trong; khi thiếu thì suy ra "gold" hoặc "asr" theo kiểu.
    """

    def __init__(self, inner: Transcriber) -> None:
        self.inner = inner
        self.calls = 0
        self.seconds = 0.0
        self.source: str = str(getattr(inner, "source", "gold" if isinstance(inner, GoldTranscriber) else "asr"))

    def transcribe(self, sample: Sample) -> Transcript:
        """Chuyển lời gọi xuống đối tượng bên trong và đo thời gian."""
        t0 = time.perf_counter()
        try:
            return self.inner.transcribe(sample)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t0

    def warmup(self) -> None:
        """Chuyển lời gọi xuống đối tượng bên trong."""
        self.inner.warmup()

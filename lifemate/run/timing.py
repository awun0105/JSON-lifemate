"""Lớp bọc đo thời gian cho backend LLM và transcriber, theo mẫu Decorator (SPEC 4.0).

Nội dung chính:
- `TimedLLM`: bọc một `LLMBackend`, ghi thời gian từng lần `generate`.
- `TimedTranscriber`: bọc một `Transcriber`, cộng dồn thời gian và số lần `transcribe`.

Cách hoạt động:
- Hai lớp bọc giữ nguyên giao diện của đối tượng bên trong, chỉ ghi thêm thời gian thực (wall clock) của mỗi
  lần gọi.
- Thời gian được ghi trong `finally`, nên lần gọi bị lỗi vẫn được tính.

Lưu ý:
- Lời gọi `warmup` không được tính vào thời gian.
"""

from __future__ import annotations

import time
from typing import Any

from lifemate.backends.asr import GoldTranscriber
from lifemate.core.types import Generation, LLMBackend, Sample, Transcriber, Transcript


class TimedLLM:
    """Bọc một `LLMBackend` và ghi số giây thực của mọi lần gọi `generate`.

    Tham số:
        inner: backend LLM được bọc.

    Thuộc tính:
        inner: như tham số cùng tên.
        durations: thời gian của từng lần `generate`, tính bằng giây.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self.inner = inner
        self.durations: list[float] = []

    @property
    def calls(self) -> int:
        """Số lần đã gọi `generate`."""
        return len(self.durations)

    @property
    def seconds(self) -> float:
        """Tổng số giây của các lần gọi `generate`."""
        return sum(self.durations)

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None,
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Chuyển lời gọi xuống backend bên trong và ghi lại thời gian chạy."""
        t0 = time.perf_counter()
        try:
            return self.inner.generate(messages, schema, sample_id=sample_id, time_left_s=time_left_s)
        finally:
            self.durations.append(time.perf_counter() - t0)

    def prefill(self, messages: list[dict[str, Any]]) -> float:
        """Chuyển xuống `prefill` của backend bên trong nếu có (đọc trước phần đầu prompt); không có thì trả 0."""
        inner = getattr(self.inner, "prefill", None)
        return float(inner(messages)) if inner is not None else 0.0

    def warmup(self) -> None:
        """Chuyển lời gọi xuống backend bên trong; không ghi vào `durations`."""
        self.inner.warmup()


class TimedTranscriber:
    """Bọc một `Transcriber`, cộng dồn số giây và số lần gọi `transcribe`.

    Tham số:
        inner: transcriber được bọc.

    Thuộc tính:
        inner: như tham số cùng tên.
        calls: số lần đã gọi `transcribe`.
        seconds: tổng số giây của các lần gọi `transcribe`.
        source: lấy từ thuộc tính `source` của `inner`; không có thì suy ra `"gold"` hoặc `"asr"` theo kiểu.
    """

    def __init__(self, inner: Transcriber) -> None:
        self.inner = inner
        self.calls = 0
        self.seconds = 0.0
        self.source: str = str(getattr(inner, "source", "gold" if isinstance(inner, GoldTranscriber) else "asr"))

    def transcribe(self, sample: Sample) -> Transcript:
        """Chuyển lời gọi xuống transcriber bên trong và ghi lại thời gian chạy."""
        t0 = time.perf_counter()
        try:
            return self.inner.transcribe(sample)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t0

    def warmup(self) -> None:
        """Chuyển lời gọi xuống transcriber bên trong."""
        self.inner.warmup()

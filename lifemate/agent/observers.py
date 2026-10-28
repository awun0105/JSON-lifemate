"""Các observer nhận sự kiện từ vòng lặp agent: bắt đầu mẫu, từng bước, kết thúc mẫu (SPEC 4.0).

Nội dung chính:
- `NullObserver`: bỏ qua mọi sự kiện; là observer mặc định.
- `Tracer`: ghi mỗi mẫu đã xong thành một dòng JSON.
- `CompositeObserver`: chuyển một sự kiện tới nhiều observer theo thứ tự.

Cách hoạt động:
- Dùng mẫu Observer: agent chỉ phát sự kiện, observer không bao giờ tác động ngược lại vòng lặp.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import AgentObserver, Sample, SampleResult, StepRecord, Transcript


class NullObserver:
    """Observer bỏ qua mọi sự kiện."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Bỏ qua."""

    def on_step(self, step: StepRecord) -> None:
        """Bỏ qua."""

    def on_stop(self, result: SampleResult) -> None:
        """Bỏ qua."""


class Tracer:
    """Gom các bước của mẫu đang chạy và ghi `SampleResult.to_dict()` thành một dòng JSON vào cuối file `path`.

    Tracer không bao giờ ảnh hưởng tới agent. File `trace.jsonl` chính do `RunStore` của Runner ghi; Tracer chỉ
    dùng cho một luồng ghi phụ, ví dụ trace gỡ lỗi giữ cả những mẫu mà Runner sau đó ghi đè.

    Tham số:
        path: file JSONL đích, mở ở chế độ ghi nối thêm.

    Thuộc tính:
        current_id: id mẫu đang chạy; `None` khi không có mẫu nào.
        steps: các bước đã gom của mẫu đang chạy.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.current_id: str | None = None
        self.steps: list[StepRecord] = []

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Bắt đầu gom bước cho một mẫu mới."""
        self.current_id = sample.id
        self.steps = []

    def on_step(self, step: StepRecord) -> None:
        """Gom thêm một bước."""
        self.steps.append(step)

    def on_stop(self, result: SampleResult) -> None:
        """Ghi mẫu đã xong thành một dòng, rồi flush và fsync ngay."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(result.to_dict()) + "\n")
            # fsync để dòng vừa ghi không bị mất khi tiến trình bị dừng đột ngột.
            f.flush()
            os.fsync(f.fileno())
        self.current_id = None
        self.steps = []


class CompositeObserver:
    """Chuyển mọi sự kiện tới nhiều observer, đúng theo thứ tự trong danh sách.

    Tham số:
        observers: các observer nhận sự kiện.
    """

    def __init__(self, observers: Sequence[AgentObserver]) -> None:
        self.observers: tuple[AgentObserver, ...] = tuple(observers)

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Chuyển sự kiện bắt đầu mẫu tới từng observer."""
        for o in self.observers:
            o.on_sample_start(sample, transcript)

    def on_step(self, step: StepRecord) -> None:
        """Chuyển sự kiện một bước tới từng observer."""
        for o in self.observers:
            o.on_step(step)

    def on_stop(self, result: SampleResult) -> None:
        """Chuyển sự kiện kết thúc mẫu tới từng observer."""
        for o in self.observers:
            o.on_stop(result)

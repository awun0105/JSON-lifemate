"""Các observer của agent (SPEC 4.0): NullObserver, Tracer, CompositeObserver.

Nội dung: ba cài đặt của giao thức `AgentObserver` (bắt đầu mẫu, từng bước, kết thúc mẫu).

Cách xây dựng: mẫu Observer. Agent chỉ phát sự kiện, observer không bao giờ tác động ngược lại vòng lặp.
`NullObserver` là mặc định không làm gì, `Tracer` ghi mỗi mẫu thành một dòng JSON, `CompositeObserver`
chuyển một sự kiện tới nhiều observer theo thứ tự.
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
    """Giữ tạm các bước của mẫu hiện tại và nối `SampleResult.to_dict()` thành một dòng JSON vào `path`.

    Không bao giờ ảnh hưởng tới agent. `RunStore` của Runner mới là nơi ghi `trace.jsonl`; Tracer dùng cho
    một luồng ghi bổ sung (ví dụ trace gỡ lỗi giữ cả những mẫu mà Runner sau đó thay thế).

    Args:
        path: file JSONL đích, được mở ở chế độ nối thêm.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.current_id: str | None = None
        self.steps: list[StepRecord] = []

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Bắt đầu giữ tạm cho một mẫu mới."""
        self.current_id = sample.id
        self.steps = []

    def on_step(self, step: StepRecord) -> None:
        """Giữ tạm một bước."""
        self.steps.append(step)

    def on_stop(self, result: SampleResult) -> None:
        """Ghi mẫu đã xong thành một dòng, flush và fsync ngay."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(result.to_dict()) + "\n")
            # fsync để dòng đã ghi không mất khi tiến trình bị dừng đột ngột giữa chừng.
            f.flush()
            os.fsync(f.fileno())
        self.current_id = None
        self.steps = []


class CompositeObserver:
    """Chuyển mọi sự kiện tới nhiều observer, theo đúng thứ tự."""

    def __init__(self, observers: Sequence[AgentObserver]) -> None:
        self.observers: tuple[AgentObserver, ...] = tuple(observers)

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Chuyển tiếp."""
        for o in self.observers:
            o.on_sample_start(sample, transcript)

    def on_step(self, step: StepRecord) -> None:
        """Chuyển tiếp."""
        for o in self.observers:
            o.on_step(step)

    def on_stop(self, result: SampleResult) -> None:
        """Chuyển tiếp."""
        for o in self.observers:
            o.on_stop(result)

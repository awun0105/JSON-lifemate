"""Agent observers (SPEC 4.0): NullObserver, Tracer, CompositeObserver."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from lifemate.jsonutil import canonical_json
from lifemate.types import AgentObserver, Sample, SampleResult, StepRecord, Transcript


class NullObserver:
    """Observer that ignores every event."""

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Ignore."""

    def on_step(self, step: StepRecord) -> None:
        """Ignore."""

    def on_stop(self, result: SampleResult) -> None:
        """Ignore."""


class Tracer:
    """Buffers the current sample's steps and appends `SampleResult.to_dict()` as one JSON line to `path`.

    Never influences the agent. The Runner's `RunStore` is the writer of `trace.jsonl`; a Tracer is for an
    extra stream (e.g. a debug trace that also keeps samples the Runner later replaces).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.current_id: str | None = None
        self.steps: list[StepRecord] = []

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Start buffering a new sample."""
        self.current_id = sample.id
        self.steps = []

    def on_step(self, step: StepRecord) -> None:
        """Buffer one step."""
        self.steps.append(step)

    def on_stop(self, result: SampleResult) -> None:
        """Write the finished sample as one line, flushed and fsynced."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(result.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.current_id = None
        self.steps = []


class CompositeObserver:
    """Fans every event out to several observers in order."""

    def __init__(self, observers: Sequence[AgentObserver]) -> None:
        self.observers: tuple[AgentObserver, ...] = tuple(observers)

    def on_sample_start(self, sample: Sample, transcript: Transcript) -> None:
        """Forward."""
        for o in self.observers:
            o.on_sample_start(sample, transcript)

    def on_step(self, step: StepRecord) -> None:
        """Forward."""
        for o in self.observers:
            o.on_step(step)

    def on_stop(self, result: SampleResult) -> None:
        """Forward."""
        for o in self.observers:
            o.on_stop(result)

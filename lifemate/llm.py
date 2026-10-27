"""LLM backends (SPEC 4.8). v0 ships FakeLLM; VLLMBackend lands in task #4."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from lifemate.jsonutil import canonical_json
from lifemate.types import Generation, Sample

_DONE = canonical_json({"action": "done"})


class FakeLLM:
    """Replays each sample's `label.tool_calls`, then `{"action":"done"}` forever.

    `broken_outputs[sample_id]` raw strings are yielded first. With a `clock` (FakeClock-like, has `.advance`)
    each call advances it by `delay_s` instead of sleeping.
    """

    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        delay_s: float = 0.0,
        broken_outputs: Mapping[str, Sequence[str]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.clock = clock
        self._queues: dict[str, list[str]] = {}
        broken = broken_outputs or {}
        for s in samples:
            calls = (s.label or {}).get("tool_calls") or []
            replies = [
                canonical_json({"action": "call", "name": c["name"], "arguments": c.get("arguments") or {}})
                for c in calls
            ]
            self._queues[s.id] = list(broken.get(s.id, ())) + replies
        for sid, raws in broken.items():
            self._queues.setdefault(sid, list(raws))
        self._cursor: dict[str, int] = {}

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any], *, sample_id: str) -> Generation:
        """Next scripted output for `sample_id`."""
        queue = self._queues.get(sample_id, [])
        i = self._cursor.get(sample_id, 0)
        text = queue[i] if i < len(queue) else _DONE
        self._cursor[sample_id] = i + 1
        if self.delay_s > 0:
            advance = getattr(self.clock, "advance", None)
            if advance is not None:
                advance(self.delay_s)
            elif self.clock is None:
                time.sleep(self.delay_s)
        prompt = " ".join(str(m.get("content") or "") for m in messages)
        return Generation(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            seconds=self.delay_s,
        )

    def warmup(self) -> None:
        """Nothing to warm up."""


class VLLMBackend:
    """PLACEHOLDER: implemented in task #4 (SPEC 4.8)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("task #4")

    def generate(self, messages: list[dict[str, Any]], schema: dict[str, Any], *, sample_id: str) -> Generation:
        """Not implemented."""
        raise NotImplementedError("task #4")

    def warmup(self) -> None:
        """Not implemented."""
        raise NotImplementedError("task #4")

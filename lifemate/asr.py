"""Transcribers (SPEC 4.4, 6.1)."""

from __future__ import annotations

import time

from lifemate.types import Sample, Transcript


class GoldTranscriber:
    """Returns the gold `audio_text` (train only); raises ValueError when any user turn lacks it."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Gold text of the final turn and of every user turn in history (None for assistant turns)."""
        start = time.perf_counter()
        if sample.audio_text is None:
            raise ValueError(f"{sample.id}: final turn has no audio_text (gold transcripts exist only in train)")
        history: list[str | None] = []
        for i, turn in enumerate(sample.history):
            if turn.role == "user":
                if turn.audio_text is None:
                    raise ValueError(f"{sample.id}: history turn {i} has no audio_text")
                history.append(turn.audio_text)
            else:
                history.append(None)
        return Transcript(final=sample.audio_text, history=tuple(history), seconds=time.perf_counter() - start)

    def warmup(self) -> None:
        """Nothing to warm up."""


class ASRTranscriber:
    """Placeholder for the real ASR model."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("after v0: SPEC 6.1")

    def transcribe(self, sample: Sample) -> Transcript:
        """Not implemented."""
        raise NotImplementedError("after v0: SPEC 6.1")

    def warmup(self) -> None:
        """Not implemented."""
        raise NotImplementedError("after v0: SPEC 6.1")

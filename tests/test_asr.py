import dataclasses

import pytest

from lifemate.asr import GoldTranscriber
from lifemate.types import Sample, Turn


def _turn(role: str, audio_text: str | None) -> Turn:
    return Turn(role=role, audio=None, audio_text=audio_text, text=None, tool_calls=(), tool_results=())  # type: ignore[arg-type]


SAMPLE = Sample(
    id="s",
    split="train",
    audio="a.wav",
    audio_text="cuối",
    context={},
    history=(_turn("user", "đầu"), _turn("assistant", None)),
    label=None,
)


def test_gold_ok() -> None:
    t = GoldTranscriber()
    t.warmup()
    tr = t.transcribe(SAMPLE)
    assert tr.final == "cuối" and tr.history == ("đầu", None) and tr.seconds >= 0


def test_gold_missing_final() -> None:
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, audio_text=None))


def test_gold_missing_history() -> None:
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, history=(_turn("user", None),)))

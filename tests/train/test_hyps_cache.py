"""Test `lifemate/train/hyps_cache.py`: gom dòng chữ từng câu thành mục cache đúng định dạng `asr_cache`."""

from __future__ import annotations

from lifemate.core.types import Sample, Transcript, Turn
from lifemate.train.asr_cache import CachedTranscriber
from lifemate.train.hyps_cache import cache_from_rows


def _sample() -> Sample:
    history = (
        Turn(role="user", audio="a0.wav", audio_text="mai có báo thức nào", text=None, tool_calls=(), tool_results=()),
        Turn(role="assistant", audio=None, audio_text=None, text="Có hai báo thức.", tool_calls=(), tool_results=()),
        Turn(role="user", audio="a2.wav", audio_text="xoá cái sớm nhất", text=None, tool_calls=(), tool_results=()),
    )
    return Sample(id="s1", split="train", audio="f.wav", audio_text="ok", context={}, history=history, label=None)


def test_cache_matches_asr_cache_format_and_replays() -> None:
    rows = [
        {"id": "s1", "turn": "0", "hyp": "mai có báo thức nào"},
        {"id": "s1", "turn": "final", "hyp": "ừ 5 giờ"},
    ]
    cache, missing = cache_from_rows([_sample()], rows)
    assert cache == {"s1": {"final": "ừ 5 giờ", "history": ["mai có báo thức nào", None, ""]}}
    assert missing == 1  # lượt 2 không có dòng chữ
    t = CachedTranscriber(cache).transcribe(_sample())
    assert isinstance(t, Transcript) and t.final == "ừ 5 giờ" and t.history[1] is None

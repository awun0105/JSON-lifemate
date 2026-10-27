"""Team lead's SFT transcript mix: share of samples on ASR, gold fallback when ASR loses an entity, stable per id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.core.types import Sample, Turn
from lifemate.train.asr_cache import CachedTranscriber, to_entry
from lifemate.train.sft_data import MixedTranscriber, _transcriber


def _sample(i: int, amount: int = 500000) -> Sample:
    label = {"tool_calls": [{"name": "ewallet.transfer", "arguments": {"amount": amount, "contact_id": "c"}}]}
    hist = (Turn(role="user", audio=None, audio_text="xin chào", text=None, tool_calls=(), tool_results=()),)  # type: ignore[arg-type]
    return Sample(
        id=f"s{i:04d}",
        split="train",
        audio="a.wav",
        audio_text=f"chuyển {amount}",
        context={},
        history=hist,
        label=label,
    )


def _cache(samples: list[Sample], final: str) -> CachedTranscriber:
    return CachedTranscriber({s.id: {"final": final.format(s=s), "history": ["xin chao"]} for s in samples})


def test_ratio_and_stability() -> None:
    samples = [_sample(i) for i in range(2000)]
    asr = _cache(samples, "chuyển 500000 nha")
    mix = MixedTranscriber(asr, 0.3, seed=0)
    finals = [mix.transcribe(s).final for s in samples]
    share = sum(f == "chuyển 500000 nha" for f in finals) / len(finals)
    assert 0.27 < share < 0.33
    again = MixedTranscriber(asr, 0.3, seed=0)
    assert [again.transcribe(s).final for s in samples] == finals


def test_falls_back_to_gold_when_entity_lost() -> None:
    samples = [_sample(i) for i in range(200)]
    mix = MixedTranscriber(_cache(samples, "chuyển năm trăm nghìn"), 1.0)
    assert all(mix.transcribe(s).final == s.audio_text for s in samples)
    assert set(mix.chosen.values()) == {"gold_fallback"}


def test_cache_roundtrip_and_missing_cache(tmp_path: Path) -> None:
    s = _sample(1)
    path = tmp_path / "c.json"
    path.write_text(json.dumps({s.id: to_entry(_cache([s], "x").transcribe(s))}), encoding="utf-8")
    t = CachedTranscriber.from_file(path).transcribe(s)
    assert t.final == "x" and t.history == ("xin chao",)
    with pytest.raises(FileNotFoundError, match="asr_cache"):
        _transcriber("mix", tmp_path / "nope.json")

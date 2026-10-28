"""Test cách trộn transcript ASR và gold cho dữ liệu SFT, theo quyết định của trưởng nhóm.

Nội dung chính:
- Tỉ lệ mẫu dùng transcript ASR và tính ổn định của lựa chọn theo id.
- Quay về văn bản gốc (gold) khi ASR làm mất thực thể.
- Ghi và đọc lại cache ASR.

Lưu ý:
- Mẫu giả dựng bằng `_sample`, cache ASR giả bằng `CachedTranscriber`; không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.core.types import Sample, Turn
from lifemate.train.asr_cache import CachedTranscriber, to_entry
from lifemate.train.sft_data import MixedTranscriber, _transcriber


def _sample(i: int, amount: int = 500000) -> Sample:
    """Tạo mẫu chuyển tiền giả với số tiền `amount`, có một lượt lịch sử của user."""
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
    """Tạo cache ASR giả: câu cuối lấy theo mẫu `final`, lịch sử luôn là "xin chao"."""
    return CachedTranscriber({s.id: {"final": final.format(s=s), "history": ["xin chao"]} for s in samples})


def test_ratio_and_stability() -> None:
    """Khoảng 30% mẫu dùng ASR khi ratio=0.3, và cùng seed thì mỗi mẫu luôn nhận cùng lựa chọn."""
    samples = [_sample(i) for i in range(2000)]
    asr = _cache(samples, "chuyển 500000 nha")
    mix = MixedTranscriber(asr, 0.3, seed=0)
    finals = [mix.transcribe(s).final for s in samples]
    share = sum(f == "chuyển 500000 nha" for f in finals) / len(finals)
    # 2000 mẫu nên tỉ lệ thực tế dao động quanh 0.3.
    assert 0.27 < share < 0.33
    again = MixedTranscriber(asr, 0.3, seed=0)
    assert [again.transcribe(s).final for s in samples] == finals


def test_falls_back_to_gold_when_entity_lost() -> None:
    """ASR viết số tiền bằng chữ (mất thực thể 500000) thì mẫu luôn dùng văn bản gốc."""
    samples = [_sample(i) for i in range(200)]
    mix = MixedTranscriber(_cache(samples, "chuyển năm trăm nghìn"), 1.0)
    assert all(mix.transcribe(s).final == s.audio_text for s in samples)
    assert set(mix.chosen.values()) == {"gold_fallback"}


def test_cache_roundtrip_and_missing_cache(tmp_path: Path) -> None:
    """Cache ghi ra file rồi đọc lại giữ nguyên nội dung; chế độ mix thiếu file cache thì báo lỗi rõ ràng."""
    s = _sample(1)
    path = tmp_path / "c.json"
    path.write_text(json.dumps({s.id: to_entry(_cache([s], "x").transcribe(s))}), encoding="utf-8")
    t = CachedTranscriber.from_file(path).transcribe(s)
    assert t.final == "x" and t.history == ("xin chao",)
    with pytest.raises(FileNotFoundError, match="asr_cache"):
        _transcriber("mix", tmp_path / "nope.json")

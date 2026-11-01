"""Test phần không cần mô hình của VieGem: cắt đoạn (`split_quiet`), kiểm tra `generate_kwargs`, từ gợi ý."""

from __future__ import annotations

import numpy as np
import pytest

from lifemate.backends.asr_viegem import ViegemASR, split_quiet
from lifemate.core.config import ASRConfig
from lifemate.core.types import Sample, Turn
from lifemate.eval.viegem_bench import bias_words


def test_split_quiet_cuts_long_audio_at_the_quietest_point() -> None:
    sr = 16000
    x = np.ones(sr * 30, dtype=np.float32)
    x[sr * 18 : sr * 18 + sr // 10] = 0.0  # im lặng ở giây 18
    segs = split_quiet(x, sr, max_s=20.0)
    assert len(segs) == 2 and 17.9 * sr < segs[0][1] < 18.2 * sr and segs[-1][1] == len(x)
    assert split_quiet(np.ones(sr * 5, dtype=np.float32), sr) == [(0, sr * 5)]


def test_unknown_generate_kwargs_rejected_before_loading() -> None:
    cfg = ASRConfig(backend="asr", model="m", revision="r", engine="viegem", generate_kwargs={"beam": 3})
    with pytest.raises(ValueError, match="unknown generate_kwargs"):
        ViegemASR(cfg)


def test_bias_words_are_english_words_from_sample_context() -> None:
    hist = (Turn(role="assistant", audio=None, audio_text=None, text="Đã gửi qua Zalo cho Olivia.", tool_calls=(),
                 tool_results=()),)  # fmt: skip
    ctx = {"installed_apps": ["chat_ott"], "user_profile": {"city": "Hà Nội"}}
    s = Sample(id="s", split="train", audio="a.wav", audio_text=None, context=ctx, history=hist, label=None)
    catalog = {"chat_ott": {"name_vi": "Nhắn tin", "description": "Gửi tin qua Messenger"}}
    words = bias_words(s, catalog)
    assert {"zalo", "olivia", "messenger"} <= set(words) and "nội" not in words

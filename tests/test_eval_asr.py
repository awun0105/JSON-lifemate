"""eval_asr metrics (SPEC 6.4.2)."""

from typing import Any

import pytest

from lifemate.backends.asr import GoldTranscriber
from lifemate.core.types import Sample, Transcript, Turn
from lifemate.eval.eval_asr import cer, entity_miss_rate, evaluate, normalize_text, percentile, wer


def _sample(sid: str, text: str, calls: list[dict[str, Any]], history: tuple[Turn, ...] = ()) -> Sample:
    return Sample(
        id=sid,
        split="train",
        audio=f"audio/{sid}/{sid}.wav",
        audio_text=text,
        context={},
        history=history,
        label={"tool_calls": calls},
    )


def test_normalize() -> None:
    assert normalize_text("  Chào, BẠN!  nhé. ") == "chào bạn nhé"


def test_wer_cer_known() -> None:
    assert wer("a b c d", "a x c") == pytest.approx(0.5)
    assert wer("Xin chào", "xin chào!") == 0.0
    assert cer("abcd", "abed") == pytest.approx(0.25)
    assert wer("", "") == 0.0


def test_entity_miss() -> None:
    s = _sample(
        "e", "chuyển cho Huy 150000", [{"name": "bank.transfer", "arguments": {"contact": "Huy", "amount": 150000}}]
    )
    missed, total, rate, details = entity_miss_rate([s], [Transcript("chuyển cho hùng 150000", (), 0.0)])
    assert (missed, total, rate) == (1, 2, 0.5)
    assert details[0].missed == ("huy",)


def test_entity_skips_ask_and_absent() -> None:
    s = _sample(
        "e",
        "đặt bàn",
        [{"name": "sys.ask_missing_info", "arguments": {"q": "đặt"}}, {"name": "x", "arguments": {"n": "Hà Nội"}}],
    )
    assert entity_miss_rate([s], [Transcript("", (), 0.0)])[:2] == (0, 0)


def test_percentile() -> None:
    xs = [float(i) for i in range(1, 21)]
    assert percentile(xs, 95) == 19.0
    assert percentile(xs, 50) == 10.0
    assert percentile([], 95) == 0.0


def test_evaluate_gold() -> None:
    u = Turn(role="user", audio=None, audio_text="trước đó", text=None, tool_calls=(), tool_results=())
    a = Turn(role="assistant", audio=None, audio_text=None, text="ok", tool_calls=(), tool_results=())
    samples = [
        _sample("a", "chuyển cho Huy 150000", [{"name": "t", "arguments": {"contact": "Huy", "amount": 150000}}]),
        _sample("b", "thời tiết Hà Nội", [{"name": "w", "arguments": {"city": "Hà Nội"}}], (u, a)),
        _sample("c", "bật đèn", []),
    ]
    rep = evaluate(samples, GoldTranscriber())
    assert rep.wer == 0.0 and rep.cer == 0.0
    assert (rep.entity_missed, rep.entity_total) == (0, 3)
    assert rep.n_utterances == 4
    text = rep.to_text()
    for key in ("WER", "CER", "entity_miss_rate", "p95"):
        assert key in text
    assert rep.to_dict()["entity_misses"] == []

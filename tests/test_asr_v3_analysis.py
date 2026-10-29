"""Test phần phân tích offline của benchmark ASR lần 3: nhãn tự động, độ tin cậy, các cách quyết định, hồi quy logistic,
hai phương án."""

from __future__ import annotations

from lifemate.eval import asr_v3_analysis as analysis
from lifemate.eval.asr_merge import Region


def _rows(hyp: str, ref: str) -> list[dict[str, str]]:
    return [{"id": "s", "turn": "final", "ref": ref, "hyp": hyp, "raw": hyp}]


def test_regions_get_automatic_labels() -> None:
    """Lấy large-v3 ít lỗi hơn thì nhãn 1, nhiều lỗi hơn thì nhãn 0."""
    good = analysis.utterance_regions(
        _rows("nhắn giá lô cho mẹ", "nhắn Zalo cho mẹ"), _rows("Nhắn Zalo cho mẹ.", "nhắn Zalo cho mẹ")
    )
    bad = analysis.utterance_regions(_rows("mai đi khám", "mai đi khám"), _rows("Mail đi khám.", "mai đi khám"))
    assert good[0].labels == [1] and bad[0].labels == [0]


def test_reliability_uses_laplace_rule() -> None:
    """(đúng + 1) / (viết ra + 2): từ chưa gặp tra ngoài là 0,5, từ 1/1 là 2/3."""
    rel = analysis.reliability({"sjc": (20, 19), "mail": (1, 1)})
    assert abs(rel["sjc"] - 20 / 22) < 1e-9 and abs(rel["mail"] - 2 / 3) < 1e-9


def test_f_and_g_deciders() -> None:
    """F đòi ngữ cảnh với từ câu mồi có độ tin cậy thấp; G chọn câu có điểm âm thanh cao hơn."""
    utts = analysis.utterance_regions(
        _rows("mai đi khám", "mai đi khám"),
        _rows("Mail đi khám.", "mai đi khám"),
        [{"id": "s", "turn": "final", "part": 0, "score_a": -1.0, "score_b": -5.0}],
    )
    u = utts[0]
    k = u.candidates[0]
    region = u.parts[k]
    assert isinstance(region, Region)
    prompt = frozenset({"mail"})
    c0 = analysis.c0_decider({"s": frozenset()}, prompt)
    f_low = analysis.f_decider({"s": frozenset()}, prompt, {"mail": 0.2}, 0.5)
    g = analysis.g_decider(c0)
    assert c0(u, k, region) and not f_low(u, k, region) and not g(u, k, region)
    assert analysis.decision_accuracy(analysis.decisions(utts, f_low)) == 1.0


def test_logistic_learns_separable_data() -> None:
    """Hồi quy logistic có phạt L2 tách được dữ liệu một chiều và cho xác suất đúng chiều."""
    x = [[float(i)] for i in range(-5, 6) if i != 0]
    y = [int(v[0] > 0) for v in x]
    model = analysis.fit_logistic(x, y, l2=0.1)
    assert analysis.predict(model, [3.0]) > 0.9 and analysis.predict(model, [-3.0]) < 0.1


def test_alternatives_format_and_counts() -> None:
    """Đoạn không chắc được viết `{chốt | còn lại}`; bản chốt nhầm thì tính là cứu được."""
    utts = analysis.utterance_regions(
        _rows("nhắn giá lô cho mẹ", "nhắn Zalo cho mẹ"), _rows("Nhắn Zalo cho mẹ.", "nhắn Zalo cho mẹ")
    )
    texts, counts = analysis.alternatives(utts, lambda u, k, r: False, lambda u, k, r: True)
    assert texts[("s", "final")] == "nhắn {giá lô | Zalo} cho mẹ"
    assert counts["uncertain"] == 1 and counts["rescuable"] == 1

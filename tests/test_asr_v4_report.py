"""Test các hàm không cần dữ liệu của `lifemate/eval/asr_v4_report.py` (benchmark ASR lần 4)."""

from __future__ import annotations

from lifemate.eval.asr_v4_report import digit_stats, digitized


def test_digitized_rows_convert_hyp_and_raw() -> None:
    rows = digitized([{"id": "s", "turn": "final", "ref": "5 giờ", "hyp": "năm giờ", "raw": "năm giờ"}])
    assert rows[0]["hyp"] == rows[0]["raw"] == "5 giờ"


def test_digit_stats_compare_against_digitized_reference() -> None:
    """Đáp án "tám giờ" (bằng chữ) được đổi thành "8 giờ" trước khi so, nên "8" của ASR không bị tính là chữ số lạ."""
    rows = [
        {"ref": "Nhắc tôi tám giờ.", "hyp": "nhắc tôi 8 giờ"},
        {"ref": "Để 30 phần trăm.", "hyp": "để 80 phần trăm"},
        {"ref": "ok", "hyp": "ok"},
    ]
    out = digit_stats(rows)
    assert (out["ref_with_digits"], out["hyp_with_digits"]) == (1, 1)
    assert (out["ref_digits"], out["ref_digits_found"], out["digits_not_in_ref"]) == (2, 1, 1)
    assert round(out["wer_digitized_ref"], 3) == round(1 / 9, 3)

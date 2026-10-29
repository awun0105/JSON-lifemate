"""Test `lifemate.eval.asr_merge`: gióng hàng và ghép chữ của model chính với từ tiếng Anh của model phụ.

Lưu ý:
- Câu thử dựa theo lỗi thật của benchmark lần 1 (RNNT nghe "check" thành "chắc", Whisper bịa câu phụ đề).
"""

from lifemate.eval.asr_merge import align_words, merge_hyp_rows, merge_transcripts


def test_align_pairs() -> None:
    """Gióng hàng trả đủ mọi phần tử của hai dãy theo thứ tự, ghép đúng các từ khớp."""
    pairs = align_words(["a", "b", "c"], ["a", "x", "c", "d"])
    assert pairs == [(0, 0), (1, 1), (2, 2), (None, 3)]


def test_merge_takes_only_new_english() -> None:
    """Chỉ thay vùng mà model phụ có từ tiếng Anh mới; phần tiếng Việt và vùng không có tiếng Anh giữ model chính."""
    assert (
        merge_transcripts("chắc ví dùm còn trên ba triệu", "Check ví dùm, còn trên 3 triệu.")
        == "Check ví dùm còn trên ba triệu"
    )
    assert merge_transcripts("ơi chuyển sang kênh s tp giùm em", "Ơi, chuyển sang kênh S Plus Sport giùm em") == (
        "ơi chuyển sang kênh s Plus Sport giùm em"
    )
    # Model phụ sai phần tiếng Việt ("lán"): không lấy, vì vùng đó không có từ tiếng Anh mới.
    assert merge_transcripts("nhắn zalo cho chị lan", "Nhắn Zalo cho chị lán") == "nhắn zalo cho chị lan"
    assert merge_transcripts("bật đèn", "") == "bật đèn"


def test_merge_hyp_rows_skips_filtered_secondary() -> None:
    """Câu của model phụ bị bộ lọc ảo giác cắt (raw khác hyp) thì bỏ, giữ model chính."""
    primary = [{"id": "s", "turn": "final", "ref": "r", "hyp": "chắc ví", "raw": "chắc ví"}]
    bad = [{"id": "s", "turn": "final", "hyp": "Check ví", "raw": "Check ví. Hãy subscribe cho kênh"}]
    good = [{"id": "s", "turn": "final", "hyp": "Check ví", "raw": "Check ví"}]
    assert merge_hyp_rows(primary, bad)[0]["hyp"] == "chắc ví"
    assert merge_hyp_rows(primary, good)[0]["hyp"] == "Check ví"

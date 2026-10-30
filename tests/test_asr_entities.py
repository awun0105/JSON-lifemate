"""Test bộ sửa tên riêng theo danh sách của mẫu (`lifemate/backends/asr_entities.py`, thử nghiệm L2 lần 4)."""

from __future__ import annotations

from lifemate.backends.asr_entities import Thresholds, candidates_from_strings, correct

CANDS = candidates_from_strings(
    [
        "Olivia Brown",
        "Harbor View Hotel",
        "SJC",
        "VivuFood",
        "Danh bạ",
        "Mô tả rất dài của một tool nào đó trong catalog",
        "Lan",
    ]
)


def test_candidates_skip_long_and_short_strings() -> None:
    texts = [c.text for c in CANDS]
    assert "Olivia Brown" in texts and "Harbor View Hotel" in texts and "Danh bạ" in texts
    assert "Lan" not in texts  # dưới MIN_CHARS
    assert all(c.n_words <= 4 for c in CANDS)


def test_correct_replaces_near_misses_only_in_non_vietnamese_windows() -> None:
    text, reps = correct("giao việc cho olivia broad rồi gọi harbour view hotel", CANDS)
    assert text == "giao việc cho Olivia Brown rồi gọi Harbor View Hotel"
    assert [(r.source, r.target) for r in reps] == [
        ("olivia broad", "Olivia Brown"),
        ("harbour view hotel", "Harbor View Hotel"),
    ]


def test_correct_leaves_vietnamese_and_exact_matches_alone() -> None:
    assert correct("tìm số của lan trong danh bạ", CANDS) == ("tìm số của lan trong danh bạ", [])
    assert correct("gửi cho Olivia Brown", CANDS)[1] == []


def test_correct_respects_distance_thresholds_and_ambiguity() -> None:
    # "SJC" là chữ viết tắt in hoa: được phép lệch 1 ("sjk"), còn "scg" lệch 2 thì không.
    assert correct("giá vàng sjk hôm nay", CANDS)[0] == "giá vàng SJC hôm nay"
    assert correct("giá vàng scg hôm nay", CANDS)[1] == []
    cands = candidates_from_strings(["Robot hút bụi", "vouchers[]", "playlist_id"])
    assert [c.text for c in cands] == ["Robot hút bụi"]
    assert correct("cho robot hút bụi cả nhà", cands)[1] == []
    cands = candidates_from_strings(["Nexora Labs", "Nexora Lab"])
    # Hai ứng viên cách nhau 1, cụm "nexora labz" gần cả hai: không sửa vì không tách được.
    assert correct("demo cho nexora labz", cands)[1] == []
    strict = Thresholds(steps=((4, 0),))
    assert correct("giao cho olivia broad", CANDS, strict)[1] == []

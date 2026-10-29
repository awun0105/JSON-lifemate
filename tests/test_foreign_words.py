"""Test `lifemate.eval.foreign_words` (tìm từ tiếng Anh chêm trong câu) và `make_splits.en_mixed_ids`.

Lưu ý:
- Câu thử viết tay, không cần dữ liệu cuộc thi.
"""

from lifemate.eval.foreign_words import find_foreign_words, folded_word_set, has_foreign_word
from lifemate.eval.make_splits import en_mixed_ids


def _groups(text: str, seen: set[str] | None = None) -> list[tuple[str, str]]:
    """Trả về các cặp (từ, nhóm) tìm được trong `text`."""
    return [(w.text, w.group) for w in find_foreign_words(text, seen or set())]


def test_groups_and_vietnamese_skipped() -> None:
    """Từ có dấu và âm tiết Việt không dấu bị bỏ qua; viết tắt, tên, từ thường, từ mượn xếp đúng nhóm."""
    assert _groups("Nhắn Zalo cho Lan, giá vàng SJC, xăng RON95 nha em.") == [
        ("Zalo", "name"),
        ("SJC", "acronym"),
        ("RON", "acronym"),
    ]
    assert _groups("Mở tivi kênh VTV3, mở đoạn chat với Sarah ở gara.") == [
        ("tivi", "loanword"),
        ("VTV", "acronym"),
        ("chat", "common"),
        ("Sarah", "name"),
        ("gara", "loanword"),
    ]
    # Chữ cái đơn ("S" trong "S+ Sport") bị bỏ; tên ghép viết hoa giữa từ là tên.
    assert _groups("Chuyển sang S+ Sport, điểm MinhMart.") == [("Sport", "name"), ("MinhMart", "name")]
    assert not has_foreign_word("Đặt báo thức năm giờ sáng mai cho anh nha.")


def test_sentence_initial_capital_uses_lowercase_seen() -> None:
    """Từ viết hoa đứng đầu câu là từ thường khi từng gặp viết thường trong kho, ngược lại là tên."""
    assert _groups("Check ví giùm. Sarah gọi chưa?", {"check"}) == [("Check", "common"), ("Sarah", "name")]
    assert _groups("Check ví giùm.") == [("Check", "name")]


def test_folded_word_set_splits_digits_and_folds() -> None:
    """`folded_word_set` bỏ dấu và tách chữ số giống cách `find_foreign_words` tách, để "RON95" khớp được "RON"."""
    assert folded_word_set("Xăng RON95, nhắn Zalo!") == {"xang", "ron", "nhan", "zalo"}


def test_en_mixed_ids_checks_final_and_user_history() -> None:
    """Mẫu được chọn khi lượt cuối hoặc một lượt user trong lịch sử có từ tiếng Anh; lượt assistant không tính."""
    items = [
        {"id": "a", "audio_text": "Bật đèn phòng khách.", "history": [{"role": "user", "audio_text": "Check mail"}]},
        {"id": "b", "audio_text": "Book xe nha.", "history": []},
        {"id": "c", "audio_text": "Tắt quạt.", "history": [{"role": "assistant", "text": "OK Zalo"}]},
    ]
    assert en_mixed_ids(items) == ["a", "b"]


def test_hits_count_occurrences_and_lenient_spellings() -> None:
    """Từ lặp chỉ đúng bằng số lần có trong chữ ASR; chế độ nới lỏng chấp nhận "TV" cho "tivi" và tên ghép bị tách."""
    from lifemate.eval.foreign_words import match_words_in_transcript

    words = find_foreign_words("Check mail rồi check lịch, mở tivi, điểm MinhMart.")
    assert [w.folded for w in words] == ["check", "mail", "check", "tivi", "minhmart"]
    hyp = "check mail rồi chắc lịch mở TV điểm Minh Mart"
    assert match_words_in_transcript(words, hyp) == [True, True, False, False, False]
    assert match_words_in_transcript(words, hyp, lenient=True) == [True, True, False, True, True]


def test_sentence_start_on_decomposed_text() -> None:
    """Chuỗi Unicode dạng NFD (dấu tách rời) vẫn xác định đúng vị trí đầu câu."""
    import unicodedata

    text = unicodedata.normalize("NFD", "Ừ, mở nhạc. Check ví nhé.")
    assert _groups(text, {"check"}) == [("Check", "common")]


def test_equivalents_both_ways_and_split_spelling() -> None:
    """Cách viết tương đương tra theo hai chiều; `contains_word` chấp nhận cả từ bị tách ("ti vi" cho "tivi")."""
    from lifemate.eval.foreign_words import contains_word, folded_words, match_words_in_transcript

    assert match_words_in_transcript(find_foreign_words("Tối nay TV có gì?"), "tối nay tivi có gì", lenient=True) == [
        True
    ]
    assert match_words_in_transcript(find_foreign_words("Tối nay TV có gì?"), "tối nay tivi có gì") == [False]
    # Hai cách viết của cùng một từ trong một câu chỉ dùng được một lần xuất hiện trong chữ ASR.
    assert match_words_in_transcript(find_foreign_words("Mở TV, tắt tivi."), "mở tivi tắt đèn", lenient=True) == [
        True,
        False,
    ]
    assert contains_word("tivi", folded_words("Tắt cái ti vi đi con."))
    assert contains_word("tivi", folded_words("Ơi TV có gì hay?"))
    assert not contains_word("tivi", folded_words("Mở VTV1 cho cô."))

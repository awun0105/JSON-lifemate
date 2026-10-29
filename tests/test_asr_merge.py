"""Test `lifemate.eval.asr_merge`: gióng hàng và ghép chữ của model chính với từ tiếng Anh của model phụ.

Lưu ý:
- Câu thử dựa theo lỗi thật của benchmark lần 1 (RNNT nghe "check" thành "chắc", Whisper bịa câu phụ đề).
"""

from lifemate.core.types import Sample, Turn
from lifemate.eval.asr_merge import (
    CONTEXT_POLICIES,
    align_words,
    english_filter,
    merge_hyp_rows,
    merge_transcripts,
    sample_context_words,
)


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


def test_english_filter_policies() -> None:
    """Bốn cách lọc bước 3b: "all" không lọc; các cách còn lại đòi từ có trong ngữ cảnh ở đúng nhóm từ."""
    context, prompt = frozenset({"zalo"}), frozenset({"mail", "zalo"})
    assert english_filter("all", context, prompt) is None
    words = ("zalo", "mail", "toyota")
    # "zalo": có trong ngữ cảnh. "mail": của câu mồi, không có trong ngữ cảnh. "toyota": ngoài câu mồi, ngoài ngữ cảnh.
    expected = {
        "context": [True, False, False],
        "context_prompt": [True, False, True],
        "context_other": [True, True, False],
    }
    for policy in CONTEXT_POLICIES[1:]:
        accept = english_filter(policy, context, prompt)
        assert accept is not None and [accept(w) for w in words] == expected[policy]


def test_merge_with_filter_keeps_primary_for_rejected_word() -> None:
    """Từ tiếng Anh mới bị bộ lọc từ chối thì vùng đó giữ model chính; từ được nhận thì vẫn lấy model phụ."""
    only_zalo = english_filter("context", frozenset({"zalo"}), frozenset())
    assert merge_transcripts("mai đi khám", "Mail đi khám", only_zalo) == "mai đi khám"
    assert merge_transcripts("nhắn giá lô cho mẹ", "Nhắn Zalo cho mẹ", only_zalo) == "nhắn Zalo cho mẹ"


def test_sample_context_words_uses_text_inputs_only() -> None:
    """Từ điển ngữ cảnh lấy catalog của app đã cài, ngữ cảnh người dùng và lượt trợ lý; không lấy lượt người dùng."""
    sample = Sample(
        id="s",
        split="train",
        audio="a.wav",
        audio_text=None,
        context={"installed_apps": {"chat_app": {}}, "user_profile": {"home": "Riverside"}},
        history=(
            Turn(role="user", audio="u1.wav", audio_text="nhắn Spotify", text=None, tool_calls=(), tool_results=()),
            Turn(role="assistant", audio=None, audio_text=None, text="Gửi cho Sarah?", tool_calls=(), tool_results=()),
        ),
        label=None,
    )
    words = sample_context_words(sample, {"chat_app": {"name_vi": "Zalo"}, "other_app": {"name_vi": "Grab"}})
    assert {"zalo", "chat", "riverside", "sarah"} <= words
    assert "grab" not in words and "spotify" not in words
    # Khoá JSON (`name_vi`, `user_profile`, `home`) không vào từ điển; chỉ giá trị.
    assert not {"name", "vi", "user", "profile", "home"} & words

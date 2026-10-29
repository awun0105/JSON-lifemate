"""Test `lifemate.eval.foreign_words_audit`: so quy tắc tìm từ ngoại với nhãn rà tay, chấm lại theo nhãn rà tay.

Lưu ý:
- Dùng câu tự viết, không cần dữ liệu cuộc thi.
"""

import json
from pathlib import Path

from lifemate.core.types import Sample, Turn
from lifemate.eval.foreign_words_audit import audit_rule, english_recall, labelled_words, load_hand_labels


def _sample(final: str, history_user: str) -> Sample:
    """Mẫu có một lượt người dùng trong lịch sử (chỉ số 0), một lượt trợ lý (chỉ số 1) và lượt cuối."""
    return Sample(
        id="s",
        split="train",
        audio="a.wav",
        audio_text=final,
        context={},
        history=(
            Turn(role="user", audio="u.wav", audio_text=history_user, text=None, tool_calls=(), tool_results=()),
            Turn(role="assistant", audio=None, audio_text=None, text="Dạ", tool_calls=(), tool_results=()),
        ),
        label=None,
    )


def test_load_and_labelled_words(tmp_path: Path) -> None:
    """Nhãn ghi thứ tự từ trong câu (đếm từ 0); `labelled_words` trả từ ở đúng vị trí, theo thứ tự câu."""
    path = tmp_path / "labels.jsonl"
    rows = [{"id": "s", "turn": "final", "word_index": 2, "group": "common"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    labels = load_hand_labels(path)
    assert labels == {("s", "final"): {2: "common"}}
    words = labelled_words("Nhắn họp team chiều nay", labels[("s", "final")])
    assert [(w.text, w.folded, w.group) for w in words] == [("team", "team", "common")]


def test_audit_counts_false_positives_and_misses() -> None:
    """"hihi" quy tắc nhận nhưng nhãn không có (nhận nhầm); "team" nhãn có nhưng quy tắc bỏ sót."""
    sample = _sample("Nhắn Zalo hihi", "họp team nha")
    labels = {("s", "final"): {1: "name"}, ("s", "0"): {1: "common"}}
    result = audit_rule([sample], labels)
    assert (result["rule_words"], result["hand_words"], result["both"]) == (2, 2, 1)
    assert result["false_positive_words"] == {"hihi": 1} and result["missed_words"] == {"team": 1}
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["group_agreement"] == 1.0


def test_english_recall_by_rule_and_by_hand() -> None:
    """Cùng chữ ASR, chấm theo quy tắc và theo nhãn rà tay; câu thiếu chữ ASR coi như rỗng."""
    sample = _sample("Nhắn Zalo cho mẹ", "họp team nha")
    labels = {("s", "final"): {1: "name"}, ("s", "0"): {1: "common"}}
    rows = [{"id": "s", "turn": "final", "hyp": "nhắn zalo cho mẹ"}]
    result = english_recall([sample], rows, labels)
    assert (result["rule"]["words"], result["rule"]["hits"]) == (1, 1)
    assert (result["hand"]["words"], result["hand"]["hits"]) == (2, 1)

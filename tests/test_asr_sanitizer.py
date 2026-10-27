"""ASR sanitizer (adapted from the team lead's repo) and the entity check used by the SFT data mix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.backends.asr_sanitizer import entities_preserved, is_pure_hallucination, sanitize, truncate_loops

DATA = Path(__file__).resolve().parent.parent / "data" / "train+public_test" / "train" / "train.jsonl"


def test_subtitle_hallucinations_removed() -> None:
    """Protects: YouTube-style boilerplate is cut; a history turn made only of it becomes empty."""
    mixed = "Hôm nay mệt quá! Cảm ơn các bạn đã theo dõi và hãy subscribe cho kênh lalaschool"
    assert sanitize(mixed, is_history=False) == "Hôm nay mệt quá!"
    pure = "Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ những video hấp dẫn"
    assert is_pure_hallucination(pure) and sanitize(pure, is_history=True) == ""


@pytest.mark.parametrize(
    "text",
    ["À đăng ký cho anh cái khóa Thiết kế dịch vụ", "hủy gói đăng ký Xemflix giúp tôi", "Chuyển sang kênh HTV7"],
)
def test_real_requests_kept(text: str) -> None:
    """Catches the source's false positive: "đăng ký <anything>" was cut, wiping subscription requests."""
    assert sanitize(text, is_history=False) == text


def test_loops_collapsed_but_spoken_digits_kept() -> None:
    """Catches the source's other false positive: "không, không không, không" in a spoken phone number."""
    assert truncate_loops("tôi muốn tôi muốn tôi muốn tôi muốn đặt xe") == "tôi muốn đặt xe"
    assert sanitize("số " + "22 " * 25, is_history=False) == "số 22"
    phone = "gọi số không chín không tám, không không không, không hai"
    assert sanitize(phone, is_history=False) == phone


@pytest.mark.data
def test_gold_transcripts_untouched() -> None:
    """Protects: the sanitizer changes none of the 5.036 gold transcripts (final and history turns) of train."""
    if not DATA.is_file():
        pytest.skip("data package missing")
    changed = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        turns = [(it["audio_text"], False)] + [
            (h["audio_text"], True) for h in it["history"] if h.get("role") == "user" and h.get("audio_text")
        ]
        changed += [it["id"] for t, hist in turns if sanitize(t, is_history=hist) != " ".join(t.split())]
    assert changed == []


def test_entities_preserved() -> None:
    """Names and long numbers must survive; IDs and times are not required (diacritics and case ignored)."""
    calls = [
        {"name": "sms.send", "arguments": {"contact_id": "c_1", "content": "Về muộn"}},
        {"name": "banking.transfer", "arguments": {"amount": 2500000, "beneficiary_id": "b"}},
    ]
    assert entities_preserved(calls, "nhắn VE MUON cho chị, chuyển 2500000 nhé")
    assert not entities_preserved(calls, "nhắn về muộn, chuyển hai triệu rưỡi")
    assert entities_preserved([{"name": "sys.ask_missing_info", "arguments": {"tool_name": "x"}}], "")

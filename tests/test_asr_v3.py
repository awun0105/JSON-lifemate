"""Test phần không cần GPU của benchmark ASR lần 3: câu mồi theo mẫu, vùng khác nhau, câu ứng viên, bộ nhớ GPU."""

from __future__ import annotations

from lifemate.backends.asr_prompt import build_sample_prompts
from lifemate.core.types import Sample, Turn
from lifemate.eval.asr_merge import assemble, candidate_regions, merge_transcripts, rule_decision, split_regions
from lifemate.eval.asr_v3 import utterance_keys
from lifemate.eval.candidate_scoring import candidate_pairs, decide_g, scoring_text
from lifemate.eval.gpu_memory import parse_memory_csv


def _turn(role: str, text: str | None = None, audio_text: str | None = None) -> Turn:
    return Turn(role=role, audio=None, audio_text=audio_text, text=text, tool_calls=(), tool_results=())


def _sample(history: tuple[Turn, ...]) -> Sample:
    return Sample(
        id="s",
        split="train",
        audio=None,
        audio_text="cuối",
        context={"installed_apps": {"chat_app": {}, "bike_rental": {}}},
        history=history,
        label=None,
    )


CATALOG = {"chat_app": {"name_vi": "Zalo"}}


def test_prompts_follow_audio_order_and_previous_assistant_turn() -> None:
    """Mỗi câu người dùng một câu mồi, đúng thứ tự file; lời trợ lý chỉ thêm vào câu đứng ngay sau lượt trợ lý."""
    sample = _sample(
        (
            _turn("user", audio_text="u1"),
            _turn("assistant", text="Gửi cho Sarah hay Sara?"),
            _turn("user", audio_text="u2"),
        )
    )
    assert build_sample_prompts(sample, CATALOG, "FIX", "E0") == ["FIX", "FIX", "FIX"]
    assert build_sample_prompts(sample, CATALOG, "FIX", "E1") == ["FIX", "FIX Gửi cho Sarah hay Sara?", "FIX"]
    e2 = build_sample_prompts(sample, CATALOG, "FIX", "E2")
    assert e2[0] == "FIX Ứng dụng đã cài: Zalo, bike rental."
    assert e2[1].endswith("Ứng dụng đã cài: Zalo, bike rental. Gửi cho Sarah hay Sara?")
    # Lượt người dùng (chỉ có âm thanh lúc thi) không bao giờ vào câu mồi.
    assert not any("u1" in p or "u2" in p for p in e2)


def test_utterance_keys_match_hyps_turns() -> None:
    """Khoá lượt giống `hyps.jsonl` của `eval_asr`: chỉ số lượt người dùng trong lịch sử, rồi "final"."""
    sample = _sample((_turn("user", audio_text="u1"), _turn("assistant", text="a"), _turn("user", audio_text="u2")))
    assert utterance_keys(sample) == [("0", "u1"), ("2", "u2"), ("final", "cuối")]


def test_regions_reassemble_to_same_merge() -> None:
    """Tách vùng rồi ghép lại bằng luật cho đúng kết quả của `merge_transcripts`; giữ hết thì ra câu chính."""
    primary, secondary = "chắc ví dùm còn trên ba triệu", "Check ví dùm, còn trên 3 triệu."
    parts = split_regions(primary, secondary)
    assert assemble(parts, lambda _k, _r: False) == primary
    assert assemble(parts, lambda _k, r: rule_decision(r)) == merge_transcripts(primary, secondary)
    regions = candidate_regions(parts)
    assert len(regions) == 1  # chỉ vùng "chắc" / "Check" có từ tiếng Anh mới


def test_candidate_pairs_differ_only_in_one_region() -> None:
    """Hai câu ứng viên cùng kiểu viết và chỉ khác nhau ở đúng vùng cần quyết định."""
    pairs = candidate_pairs("nhắn giá lô cho mẹ", "Nhắn Zalo cho mẹ.")
    assert len(pairs) == 1
    assert pairs[0].text_a == "nhắn giá lô cho mẹ"
    assert pairs[0].text_b == "nhắn zalo cho mẹ"
    assert candidate_pairs("mai đi khám", "") == []
    assert scoring_text("Nhắn, ZALO!") == "nhắn zalo"
    assert decide_g(-4.7, -1.1) and not decide_g(-1.0, -1.0)


def test_parse_memory_csv_skips_noise() -> None:
    """Đọc được số MiB mỗi dòng, bỏ dòng lỗi."""
    assert parse_memory_csv("1200\n 8000 \nN/A\n\n7,1\n") == [1200, 8000, 7]

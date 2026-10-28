"""Kiểm thử GoldTranscriber và việc chọn dtype cho ASR.

Dùng Sample dựng tay, không cần mô hình, dữ liệu cuộc thi hay tokenizer.
"""

import dataclasses

import pytest

from lifemate.backends.asr import GoldTranscriber, resolve_asr_dtype
from lifemate.core.types import Sample, Turn


def _turn(role: str, audio_text: str | None) -> Turn:
    """Tạo lượt hội thoại chỉ có `audio_text`."""
    return Turn(role=role, audio=None, audio_text=audio_text, text=None, tool_calls=(), tool_results=())  # type: ignore[arg-type]


SAMPLE = Sample(
    id="s",
    split="train",
    audio="a.wav",
    audio_text="cuối",
    context={},
    history=(_turn("user", "đầu"), _turn("assistant", None)),
    label=None,
)


def test_gold_ok() -> None:
    """Bảo vệ: GoldTranscriber trả transcript gold của câu cuối và lịch sử (lượt assistant là None)."""
    t = GoldTranscriber()
    t.warmup()
    tr = t.transcribe(SAMPLE)
    assert tr.final == "cuối" and tr.history == ("đầu", None) and tr.seconds >= 0


def test_gold_missing_final() -> None:
    """Bảo vệ: thiếu transcript gold của câu cuối thì báo lỗi ValueError."""
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, audio_text=None))


def test_gold_missing_history() -> None:
    """Bảo vệ: lượt user trong lịch sử thiếu transcript gold thì báo lỗi ValueError."""
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, history=(_turn("user", None),)))


@pytest.mark.parametrize(
    ("cfg", "on_gpu", "cc", "want"),
    [
        ("auto", True, (8, 0), "bfloat16"),  # A30
        ("auto", True, (7, 5), "float16"),  # T4: không có bfloat16 nhanh
        ("auto", False, None, "float32"),  # CPU
        ("float32", True, (8, 0), "float32"),  # giá trị đặt rõ được ưu tiên
        ("float16", True, (8, 6), "float16"),
    ],
)
def test_resolve_asr_dtype(cfg: str, on_gpu: bool, cc: tuple[int, int] | None, want: str) -> None:
    """Bảo vệ: mặc định ASR không bao giờ chạy float32 trên GPU (từng hết bộ nhớ khi chạy cạnh vLLM trên T4)."""
    assert resolve_asr_dtype(cfg, on_gpu, cc) == want


@pytest.mark.parametrize(("cfg", "on_gpu", "cc"), [("bfloat16", True, (7, 5)), ("float16", False, None)])
def test_resolve_asr_dtype_rejects(cfg: str, on_gpu: bool, cc: tuple[int, int] | None) -> None:
    """Bắt lỗi: bfloat16 trên GPU đời trước sm80, hoặc dtype nửa độ chính xác trên CPU."""
    with pytest.raises(ValueError, match=r"asr\.dtype"):
        resolve_asr_dtype(cfg, on_gpu, cc)


def test_hf_generate_kwargs_maps_shared_config() -> None:
    """Bảo vệ: cấu hình chung kiểu faster-whisper dịch đúng sang transformers; mặc định vẫn tham lam như cũ."""
    from lifemate.backends.asr import hf_generate_kwargs

    assert hf_generate_kwargs({"language": "vi", "task": "transcribe"}) == (
        {"language": "vi", "task": "transcribe", "num_beams": 1, "do_sample": False},
        "",
    )
    shared = {
        "language": "vi",
        "temperature": 0.0,
        "beam_size": 5,
        "best_of": 5,
        "condition_on_previous_text": False,
        "initial_prompt": "Nhắn Zalo cho Lan.",
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "vad_filter": False,
    }
    kwargs, prompt = hf_generate_kwargs(shared)
    assert prompt == "Nhắn Zalo cho Lan."
    assert kwargs == {
        "language": "vi",
        "temperature": 0.0,
        "condition_on_prev_tokens": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "do_sample": False,
    }
    with pytest.raises(ValueError, match="vad_filter"):
        hf_generate_kwargs({"vad_filter": True})

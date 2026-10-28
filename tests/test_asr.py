"""Test `GoldTranscriber` và các hàm chọn tham số cho ASR (dtype, `generate_kwargs`).

Lưu ý:
- Dùng `Sample` dựng tay; không cần model, dữ liệu cuộc thi hay tokenizer.
"""

import dataclasses
from pathlib import Path
from typing import Any

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
    """`GoldTranscriber` trả transcript gold của câu cuối và lịch sử; lượt assistant là `None`."""
    t = GoldTranscriber()
    t.warmup()
    tr = t.transcribe(SAMPLE)
    assert tr.final == "cuối" and tr.history == ("đầu", None) and tr.seconds >= 0


def test_gold_missing_final() -> None:
    """Thiếu transcript gold của câu cuối thì báo `ValueError`."""
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, audio_text=None))


def test_gold_missing_history() -> None:
    """Lượt user trong lịch sử thiếu transcript gold thì báo `ValueError`."""
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, history=(_turn("user", None),)))


@pytest.mark.parametrize(
    ("cfg", "on_gpu", "cc", "want"),
    [
        ("auto", True, (8, 0), "bfloat16"),  # A30
        ("auto", True, (7, 5), "float16"),  # T4: không có bfloat16 nhanh
        ("auto", False, None, "float32"),  # CPU
        ("float32", True, (8, 0), "float32"),  # giá trị ghi rõ trong config được ưu tiên
        ("float16", True, (8, 6), "float16"),
    ],
)
def test_resolve_asr_dtype(cfg: str, on_gpu: bool, cc: tuple[int, int] | None, want: str) -> None:
    """Mặc định ASR không bao giờ chạy float32 trên GPU (từng hết bộ nhớ khi chạy cạnh vLLM trên T4)."""
    assert resolve_asr_dtype(cfg, on_gpu, cc) == want


@pytest.mark.parametrize(("cfg", "on_gpu", "cc"), [("bfloat16", True, (7, 5)), ("float16", False, None)])
def test_resolve_asr_dtype_rejects(cfg: str, on_gpu: bool, cc: tuple[int, int] | None) -> None:
    """Báo lỗi khi chọn bfloat16 trên GPU dưới sm80, hoặc dtype nửa độ chính xác trên CPU."""
    with pytest.raises(ValueError, match=r"asr\.dtype"):
        resolve_asr_dtype(cfg, on_gpu, cc)


def test_hf_generate_kwargs_maps_shared_config() -> None:
    """Config chung kiểu faster-whisper được dịch đúng sang transformers; mặc định vẫn giải mã tham lam như cũ."""
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


def test_nemo_nonfinite_rows_rerun_in_float32(tmp_path: Path) -> None:
    """Engine NeMo: câu ra NaN khi chạy autocast được chạy lại không autocast; câu tốt giữ nguyên (cần torch)."""
    torch = pytest.importorskip("torch")
    import wave as _wave

    from lifemate.backends.asr import NemoCTCASR
    from lifemate.core.config import ASRConfig

    paths = []
    for i, n in enumerate((1600, 800)):
        p = tmp_path / f"{i}.wav"
        with _wave.open(str(p), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\x01\x00" * n)
        paths.append(p)
    eng = object.__new__(NemoCTCASR)
    eng._cfg = ASRConfig(
        backend="asr",
        engine="nemo_ctc",
        model="m",
        revision="",
        device="cpu",
        dtype="float16",
        batch_size=8,
        generate_kwargs={},
        sample_rate=16000,
        sanitize=False,
    )
    eng._torch, eng._device, eng.dtype, eng._autocast = torch, torch.device("cpu"), "float16", True
    calls: list[tuple[int, bool]] = []

    def fake_forward(x: Any, lens: Any, autocast: bool) -> tuple[Any, Any]:
        calls.append((x.shape[0], autocast))
        lp = torch.zeros(x.shape[0], 4, 3)
        if autocast:
            lp[1, 0, 0] = float("nan")  # câu thứ hai tràn số
        return lp, torch.full((x.shape[0],), 4)

    class _Decoding:
        def ctc_decoder_predictions_tensor(self, lp: Any, decoder_lengths: Any) -> list[str]:
            return ["nan" if bool(torch.isnan(r).any()) else "ok" for r in lp]

    eng._forward = fake_forward  # type: ignore[method-assign]
    eng._model = type("M", (), {"decoding": _Decoding()})()
    assert eng.transcribe_batch(paths) == ["ok", "ok"]
    assert calls == [(2, True), (1, False)]

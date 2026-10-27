import dataclasses

import pytest

from lifemate.backends.asr import GoldTranscriber, resolve_asr_dtype
from lifemate.core.types import Sample, Turn


def _turn(role: str, audio_text: str | None) -> Turn:
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
    t = GoldTranscriber()
    t.warmup()
    tr = t.transcribe(SAMPLE)
    assert tr.final == "cuối" and tr.history == ("đầu", None) and tr.seconds >= 0


def test_gold_missing_final() -> None:
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, audio_text=None))


def test_gold_missing_history() -> None:
    with pytest.raises(ValueError):
        GoldTranscriber().transcribe(dataclasses.replace(SAMPLE, history=(_turn("user", None),)))


@pytest.mark.parametrize(
    ("cfg", "on_gpu", "cc", "want"),
    [
        ("auto", True, (8, 0), "bfloat16"),  # A30
        ("auto", True, (7, 5), "float16"),  # T4: no fast bfloat16
        ("auto", False, None, "float32"),  # CPU
        ("float32", True, (8, 0), "float32"),  # explicit wins
        ("float16", True, (8, 6), "float16"),
    ],
)
def test_resolve_asr_dtype(cfg: str, on_gpu: bool, cc: tuple[int, int] | None, want: str) -> None:
    """Protects: ASR is never left in float32 on a GPU by default (it OOMed next to vLLM on a T4)."""
    assert resolve_asr_dtype(cfg, on_gpu, cc) == want


@pytest.mark.parametrize(("cfg", "on_gpu", "cc"), [("bfloat16", True, (7, 5)), ("float16", False, None)])
def test_resolve_asr_dtype_rejects(cfg: str, on_gpu: bool, cc: tuple[int, int] | None) -> None:
    """Catches: bfloat16 on a pre-sm80 GPU, or a half-precision dtype on CPU."""
    with pytest.raises(ValueError, match=r"asr\.dtype"):
        resolve_asr_dtype(cfg, on_gpu, cc)

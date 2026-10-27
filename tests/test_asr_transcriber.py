"""ASRTranscriber with a scripted fake backend (no model)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

import lifemate.backends.asr as asr_mod
from lifemate.backends.asr import ASRTranscriber, make_backend
from lifemate.core.config import ASRConfig
from lifemate.core.types import Sample, Turn


class FakeASRBackend:
    """Returns `text:<file name>` per path; records calls."""

    def __init__(self, drop: int = 0) -> None:
        self.calls: list[list[Path]] = []
        self.warmups = 0
        self.drop = drop

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        self.calls.append(list(wav_paths))
        out = [f"text:{p.name}" for p in wav_paths]
        return out[: len(out) - self.drop]

    def warmup(self) -> None:
        self.warmups += 1


def _turn(role: str, audio: str | None) -> Turn:
    return Turn(role=role, audio=audio, audio_text=None, text=None, tool_calls=(), tool_results=())  # type: ignore[arg-type]


SAMPLE = Sample(
    id="t1",
    split="train",
    audio="audio/t1/t1.wav",
    audio_text=None,
    context={},
    history=(
        _turn("user", "audio/t1/t1_u1.wav"),
        _turn("assistant", None),
        _turn("user", None),  # path derived from the id
        _turn("assistant", None),
    ),
    label=None,
)


def _make(batch_size: int = 8, drop: int = 0) -> tuple[ASRTranscriber, FakeASRBackend]:
    fake = FakeASRBackend(drop)
    cfg = ASRConfig(backend="asr", model="m", revision="r", batch_size=batch_size)
    return ASRTranscriber(cfg, data_root=Path("/data"), split="train", backend=fake), fake


def test_order_and_transcript() -> None:
    tr, fake = _make()
    t = tr.transcribe(SAMPLE)
    assert [p.name for p in fake.calls[0]] == ["t1_u1.wav", "t1_u2.wav", "t1.wav"]
    assert fake.calls[0][0] == Path("/data/train/audio/t1/t1_u1.wav")
    assert len(fake.calls) == 1
    assert t.final == "text:t1.wav"
    assert t.history == ("text:t1_u1.wav", None, "text:t1_u2.wav", None)
    assert t.seconds >= 0 and tr.source == "asr"


def test_batching_keeps_order() -> None:
    tr, fake = _make(batch_size=2)
    t = tr.transcribe(SAMPLE)
    assert [len(c) for c in fake.calls] == [2, 1]
    assert t.history[2] == "text:t1_u2.wav" and t.final == "text:t1.wav"


def test_count_mismatch_raises() -> None:
    tr, _ = _make(drop=1)
    with pytest.raises(ValueError, match="texts"):
        tr.transcribe(SAMPLE)


def test_warmup_once() -> None:
    tr, fake = _make()
    tr.warmup()
    tr.warmup()
    assert fake.warmups == 1 and fake.calls == []


def test_engine_registry_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects: `engine` picks the backend; ASRTranscriber builds it when none is injected."""
    built: list[ASRConfig] = []

    def factory(cfg: ASRConfig) -> FakeASRBackend:
        built.append(cfg)
        return FakeASRBackend()

    monkeypatch.setitem(asr_mod.ASR_ENGINES, "scripted", factory)
    cfg = ASRConfig(backend="asr", model="m", revision="r", engine="scripted")
    tr = ASRTranscriber(cfg, data_root=Path("/data"), split="train")
    assert built == [cfg] and tr.transcribe(SAMPLE).final == "text:t1.wav"
    with pytest.raises(ValueError, match="engine"):
        make_backend(ASRConfig(backend="asr", model="m", revision="r", engine="nope"))


def test_faster_whisper_engine_and_sanitize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """faster_whisper engine passes generate_kwargs through; sanitize=true cleans history hallucinations."""
    import sys
    import types

    calls: list[dict] = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _Seg:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Model:
        def __init__(self, path: str, device: str, compute_type: str) -> None:
            calls.append({"path": path, "device": device, "compute_type": compute_type})

        def transcribe(self, audio, **kw):  # type: ignore[no-untyped-def]
            calls.append(kw)
            text = "Hãy subscribe cho kênh La La School" if len(calls) % 2 == 0 else "bật đèn"
            return iter([_Seg(text)]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_Model))
    from lifemate.backends import asr as asr_mod

    cfg = ASRConfig(
        backend="asr",
        model=str(tmp_path),
        revision="",
        engine="faster_whisper",
        device="cpu",
        generate_kwargs={"language": "vi", "beam_size": 5},
        sanitize=True,
    )
    backend = asr_mod.make_backend(cfg)
    assert isinstance(backend, asr_mod.FasterWhisperASR) and calls[0]["compute_type"] == "float32"
    monkeypatch.setattr(asr_mod, "read_wav", lambda p, sr: p)
    out = backend.transcribe_batch([Path("a.wav"), Path("b.wav")])
    assert out == ["Hãy subscribe cho kênh La La School", "bật đèn"] and calls[1] == {"language": "vi", "beam_size": 5}
    assert asr_mod.sanitize(out[0], is_history=True) == ""

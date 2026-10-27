"""Transcribers (SPEC 4.4, 6.1)."""

from __future__ import annotations

import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from lifemate.config import ASRConfig
from lifemate.types import Sample, Transcript


class GoldTranscriber:
    """Returns the gold `audio_text` (train only); raises ValueError when any user turn lacks it."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Gold text of the final turn and of every user turn in history (None for assistant turns)."""
        start = time.perf_counter()
        if sample.audio_text is None:
            raise ValueError(f"{sample.id}: final turn has no audio_text (gold transcripts exist only in train)")
        history: list[str | None] = []
        for i, turn in enumerate(sample.history):
            if turn.role == "user":
                if turn.audio_text is None:
                    raise ValueError(f"{sample.id}: history turn {i} has no audio_text")
                history.append(turn.audio_text)
            else:
                history.append(None)
        return Transcript(final=sample.audio_text, history=tuple(history), seconds=time.perf_counter() - start)

    def warmup(self) -> None:
        """Nothing to warm up."""


class ASRBackend(Protocol):
    """Speech-to-text model: one text per input, same order."""

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Transcribe every WAV file; returns texts in input order."""
        ...

    def warmup(self) -> None:
        """Run the model once on 1 s of silence."""
        ...


def read_wav(path: Path, expected_rate: int) -> Any:
    """Read a 16-bit PCM WAV (mono or averaged to mono) as float32 numpy in [-1, 1]."""
    import numpy as np  # numpy comes with transformers (extra `gpu`)

    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    if rate != expected_rate:
        raise ValueError(f"{path}: expected {expected_rate} Hz, got {rate} Hz")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


class HFSeq2SeqASR:
    """transformers ASR pipeline, greedy decoding, no chunking (needs extras `tokenizer` and `gpu`)."""

    def __init__(self, cfg: ASRConfig) -> None:
        from transformers import pipeline

        if not cfg.model:
            raise ValueError("asr.model: required when asr.backend = 'asr'")
        self._cfg = cfg
        device: int | str
        if cfg.device == "auto":
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        else:
            device = 0 if cfg.device == "cuda" else "cpu"
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg.model,
            revision=cfg.revision or None,
            device=device,
        )
        self._generate_kwargs = {**cfg.generate_kwargs, "num_beams": 1, "do_sample": False}

    def _run(self, items: list[dict[str, Any]]) -> list[str]:
        pipe: Any = self._pipe
        out = pipe(items, batch_size=self._cfg.batch_size, generate_kwargs=self._generate_kwargs)
        if isinstance(out, dict):
            out = [out]
        return [str(o["text"]).strip() for o in out]

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Read every WAV and run the pipeline on the list."""
        sr = self._cfg.sample_rate
        return self._run([{"raw": read_wav(p, sr), "sampling_rate": sr} for p in wav_paths])

    def warmup(self) -> None:
        """One pass over 1 s of silence."""
        import numpy as np

        sr = self._cfg.sample_rate
        self._run([{"raw": np.zeros(sr, dtype=np.float32), "sampling_rate": sr}])


class ASRTranscriber:
    """Real ASR over the sample WAV files (SPEC 6.1); one backend batch per sample."""

    source = "asr"

    def __init__(self, cfg: ASRConfig, *, data_root: Path, split: str, backend: ASRBackend | None = None) -> None:
        self._cfg = cfg
        self._base = Path(data_root) / split
        self._backend: ASRBackend = backend if backend is not None else HFSeq2SeqASR(cfg)
        self._warmed = False

    def wav_paths(self, sample: Sample) -> list[Path]:
        """User-turn WAVs in history order (_u1, _u2, ...) then the final request."""
        paths: list[Path] = []
        k = 0
        for turn in sample.history:
            if turn.role == "user":
                k += 1
                rel = turn.audio or f"audio/{sample.id}/{sample.id}_u{k}.wav"
                paths.append(self._base / rel)
        paths.append(self._base / (sample.audio or f"audio/{sample.id}/{sample.id}.wav"))
        return paths

    def transcribe(self, sample: Sample) -> Transcript:
        """Transcribe history user turns and the final turn; None for assistant turns."""
        start = time.perf_counter()
        paths = self.wav_paths(sample)
        size = max(1, self._cfg.batch_size)
        texts: list[str] = []
        for i in range(0, len(paths), size):
            chunk = paths[i : i + size]
            got = self._backend.transcribe_batch(chunk)
            if len(got) != len(chunk):
                raise ValueError(f"{sample.id}: ASR returned {len(got)} texts for {len(chunk)} files")
            texts.extend(got)
        it = iter(texts[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        return Transcript(final=texts[-1], history=history, seconds=time.perf_counter() - start)

    def warmup(self) -> None:
        """Warm the backend once."""
        if not self._warmed:
            self._backend.warmup()
            self._warmed = True

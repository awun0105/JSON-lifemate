"""Transcribers (SPEC 4.4, 6.1)."""

from __future__ import annotations

import time
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

from lifemate.backends.asr_sanitizer import sanitize
from lifemate.core.config import ASRConfig
from lifemate.core.types import Sample, Transcript


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


def resolve_asr_dtype(cfg_dtype: str, on_gpu: bool, compute_capability: tuple[int, int] | None) -> str:
    """Torch dtype name for the ASR model: auto -> bfloat16 on sm80+, float16 on older GPUs, float32 on CPU.

    Without this the transformers pipeline loads float32: about 3 GB for PhoWhisper-medium, which does not
    fit next to vLLM on a 15 GB T4 and wastes memory and time on the A30.
    """
    if cfg_dtype != "auto":
        if cfg_dtype != "float32" and not on_gpu:
            raise ValueError(f"asr.dtype = {cfg_dtype!r} needs a GPU; use 'auto' or 'float32' on CPU")
        if cfg_dtype == "bfloat16" and compute_capability is not None and compute_capability < (8, 0):
            cc = compute_capability
            raise ValueError(f"asr.dtype = 'bfloat16' needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
        return cfg_dtype
    if not on_gpu:
        return "float32"
    return "bfloat16" if compute_capability is not None and compute_capability >= (8, 0) else "float16"


class HFSeq2SeqASR:
    """transformers ASR pipeline, greedy decoding, no chunking (needs extras `tokenizer` and `gpu`)."""

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from transformers import pipeline

        self._cfg = cfg
        device: int | str
        if cfg.device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        else:
            device = 0 if cfg.device == "cuda" else "cpu"
        on_gpu = device == 0
        cc = tuple(torch.cuda.get_device_capability(0)) if on_gpu else None
        self.dtype = resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=cfg.model,
            revision=cfg.revision or None,
            device=device,
            dtype=getattr(torch, self.dtype),
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


class ChunkFormerASR:
    """ChunkFormer CTC (khanhld/chunkformer-ctc-large-vie) via the `chunkformer` package; greedy CTC.

    `cfg.model` is a HF id or a local snapshot folder; `cfg.revision` is ignored by the package, so pin
    the snapshot folder in Docker. `generate_kwargs` may override chunk_size / left_context_size /
    right_context_size / total_batch_duration.
    """

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "chunk_size": 64,
        "left_context_size": 128,
        "right_context_size": 128,
        "total_batch_duration": 1800,
    }

    def __init__(self, cfg: ASRConfig) -> None:
        from chunkformer import ChunkFormerModel

        self._cfg = cfg
        self._model: Any = ChunkFormerModel.from_pretrained(cfg.model)
        self._kwargs = {**self._DEFAULTS, **cfg.generate_kwargs}

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """The package reads the files itself; one text per path, input order."""
        out = self._model.batch_decode(audio_paths=[str(p) for p in wav_paths], **self._kwargs)
        return [str(t).strip() for t in out]

    def warmup(self) -> None:
        """One pass over 1 s of silence written to a temp WAV."""
        import tempfile

        import numpy as np

        sr = self._cfg.sample_rate
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "silence.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(np.zeros(sr, dtype="<i2").tobytes())
            self.transcribe_batch([p])


class VieGemASR:
    """VieGem-ASR (5CD-AI/VieGem-ASR): streaming CTC on the EmbeddingGemma 2 audio encoder; greedy, offline mode.

    `cfg.model` is a local snapshot folder holding `viegem_asr.pt` and the `asr/` code folder (the repo ships
    its own model code, not a transformers architecture). Files are decoded one by one: the published code
    takes a single utterance, and ≤ 30 s clips keep that cheap.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import sys

        import torch
        from transformers import AutoTokenizer

        root = Path(cfg.model)
        if not (root / "viegem_asr.pt").is_file():
            raise ValueError(f"asr.model: {root} has no viegem_asr.pt (engine = 'viegem' needs the snapshot folder)")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from asr.model import CTCModel, ModelConfig, greedy_decode  # type: ignore[import-not-found]
        from asr.text import Units  # type: ignore[import-not-found]

        self._cfg = cfg
        use_cuda = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        self._device = torch.device("cuda" if use_cuda else "cpu")
        ck = torch.load(root / "viegem_asr.pt", map_location="cpu")
        model_cfg = ModelConfig.from_dict(ck["model_config"])
        self._units = Units(ck["gemma_ids"])
        self._tok = AutoTokenizer.from_pretrained(model_cfg.encoder_id)
        model = CTCModel(model_cfg, self._units.gemma_ids)
        model.load_state_dict(ck["model"], strict=False)
        self._model = model.to(self._device).eval()
        self._greedy_decode = greedy_decode
        self._torch = torch

    def _one(self, audio: Any) -> str:
        torch = self._torch
        x = torch.from_numpy(audio)[None].to(self._device)
        lens = torch.tensor([x.shape[1]], device=self._device)
        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, out_lens = self._model(x, lens, streaming=False)
            else:
                logits, out_lens = self._model(x, lens, streaming=False)
        ids = self._greedy_decode(logits, out_lens)[0]
        return str(self._units.decode(self._tok, ids)).strip()

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Decode each WAV in order."""
        sr = self._cfg.sample_rate
        return [self._one(read_wav(p, sr)) for p in wav_paths]

    def warmup(self) -> None:
        """One pass over 1 s of silence."""
        import numpy as np

        self._one(np.zeros(self._cfg.sample_rate, dtype=np.float32))


class FasterWhisperASR:
    """Whisper on CTranslate2 (`faster-whisper`, extra `asr-faster-whisper`), one file at a time.

    Team lead's choice (ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/inference.py@7033c9f): whisper-large-v3,
    beam 5, VAD, a domain `initial_prompt`, no conditioning on previous text. Decoding options come from
    `[asr].generate_kwargs` unchanged; `cfg.model` is a CTranslate2 model repo (e.g. Systran/faster-whisper-large-v3)
    pinned at `cfg.revision`, or a local folder with the converted model (Docker: /app/models/asr).
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from faster_whisper import WhisperModel  # type: ignore[import-not-found,unused-ignore]  # extra asr-faster-whisper

        self._cfg = cfg
        on_gpu = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        cc = tuple(torch.cuda.get_device_capability(0)) if on_gpu else None
        self.dtype = resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        path = cfg.model
        if cfg.revision and not Path(cfg.model).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(cfg.model, revision=cfg.revision)
        self._model = WhisperModel(path, device="cuda" if on_gpu else "cpu", compute_type=self.dtype)
        self._kwargs = dict(cfg.generate_kwargs)

    def _one(self, audio: Any) -> str:
        segments, _info = self._model.transcribe(audio, **self._kwargs)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Each WAV in order."""
        sr = self._cfg.sample_rate
        return [self._one(read_wav(p, sr)) for p in wav_paths]

    def warmup(self) -> None:
        """One pass over 1 s of silence."""
        import numpy as np

        self._one(np.zeros(self._cfg.sample_rate, dtype=np.float32))


ASR_ENGINES: dict[str, Callable[[ASRConfig], ASRBackend]] = {
    "faster_whisper": FasterWhisperASR,
    "hf_seq2seq": HFSeq2SeqASR,
    "chunkformer": ChunkFormerASR,
    "viegem": VieGemASR,
}


def make_backend(cfg: ASRConfig) -> ASRBackend:
    """Build the ASR model backend named by `cfg.engine`."""
    try:
        factory = ASR_ENGINES[cfg.engine]
    except KeyError:
        raise ValueError(f"asr.engine: unknown engine {cfg.engine!r}") from None
    return factory(cfg)


class ASRTranscriber:
    """Real ASR over the sample WAV files (SPEC 6.1); one backend batch per sample."""

    source = "asr"

    def __init__(self, cfg: ASRConfig, *, data_root: Path, split: str, backend: ASRBackend | None = None) -> None:
        self._cfg = cfg
        self._base = Path(data_root) / split
        self._backend: ASRBackend = backend if backend is not None else make_backend(cfg)
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
        if self._cfg.sanitize:
            texts = [sanitize(t, is_history=i < len(texts) - 1) for i, t in enumerate(texts)]
        it = iter(texts[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        return Transcript(final=texts[-1], history=history, seconds=time.perf_counter() - start)

    def warmup(self) -> None:
        """Warm the backend once."""
        if not self._warmed:
            self._backend.warmup()
            self._warmed = True

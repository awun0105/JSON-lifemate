"""Các bộ chuyển giọng nói thành văn bản (transcriber) cho agent (SPEC 4.4, 6.1).

Nội dung:
- `GoldTranscriber`: trả lại `audio_text` gốc của tập train, dùng khi muốn tách lỗi ASR khỏi lỗi LLM.
- `ASRTranscriber`: chạy ASR thật trên các file WAV của mẫu, mỗi mẫu một lô (batch) gửi xuống backend.
- Các backend mô hình theo giao thức `ASRBackend`: `FasterWhisperASR`, `HFSeq2SeqASR`, `ChunkFormerASR`, `NemoCTCASR`.

Cách xây dựng:
- Thư viện nặng (torch, transformers, faster_whisper, chunkformer, nemo, numpy) chỉ được import bên trong hàm khởi tạo
  hoặc hàm dùng đến, nên import module này không cần extra `gpu` và test chạy được trên CPU thuần.
- Backend được chọn theo tên qua registry `ASR_ENGINES` (`[asr].engine`); `make_backend` báo lỗi rõ ràng khi tên
  không có trong registry.
- Luồng dữ liệu: mẫu -> danh sách WAV (các lượt user trong lịch sử, rồi lượt cuối) -> backend trả văn bản theo
  đúng thứ tự -> `sanitize` (nếu bật) -> `Transcript` gồm văn bản lượt cuối và lịch sử (None cho lượt assistant).
- `resolve_asr_dtype` chọn dtype theo GPU để mô hình ASR vừa bộ nhớ khi chạy chung với vLLM.
"""

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
    """Trả `audio_text` gốc (chỉ có ở tập train); báo ValueError khi một lượt user bất kỳ thiếu trường này."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Văn bản gốc của lượt cuối và của mọi lượt user trong lịch sử (None cho lượt assistant)."""
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
        """Không có gì cần khởi động."""


class ASRBackend(Protocol):
    """Mô hình chuyển giọng nói thành văn bản: mỗi đầu vào một văn bản, giữ nguyên thứ tự."""

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Nhận dạng mọi file WAV; trả danh sách văn bản theo thứ tự đầu vào."""
        ...

    def warmup(self) -> None:
        """Chạy mô hình một lần trên 1 giây im lặng."""
        ...


def read_wav(path: Path, expected_rate: int) -> Any:
    """Đọc WAV PCM 16-bit (mono, hoặc lấy trung bình các kênh thành mono) thành mảng numpy float32 trong [-1, 1].

    Báo ValueError khi độ sâu bit khác 16 hoặc tần số lấy mẫu khác `expected_rate`; hàm không tự resample.
    """
    import numpy as np  # numpy đi kèm transformers (extra `gpu`)

    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    if rate != expected_rate:
        raise ValueError(f"{path}: expected {expected_rate} Hz, got {rate} Hz")
    # Mẫu int16 little-endian, chia 32768 để đưa về khoảng [-1, 1]
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def resolve_asr_dtype(cfg_dtype: str, on_gpu: bool, compute_capability: tuple[int, int] | None) -> str:
    """Tên dtype torch cho mô hình ASR: auto -> bfloat16 trên sm80+, float16 trên GPU cũ hơn, float32 trên CPU.

    Thiếu bước này, pipeline transformers nạp float32: khoảng 3 GB với PhoWhisper-medium, không vừa cạnh vLLM
    trên T4 15 GB và tốn bộ nhớ lẫn thời gian trên A30. Giá trị chọn tay được kiểm tra: dtype khác float32 cần GPU,
    bfloat16 cần compute capability >= 8.0.
    """
    if cfg_dtype != "auto":
        if cfg_dtype != "float32" and not on_gpu:
            raise ValueError(f"asr.dtype = {cfg_dtype!r} needs a GPU; use 'auto' or 'float32' on CPU")
        if cfg_dtype == "bfloat16" and compute_capability is not None and compute_capability < (8, 0):
            cc = compute_capability
            raise ValueError(f"asr.dtype = 'bfloat16' needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
        return cfg_dtype
    # Chế độ auto: CPU luôn float32; GPU Ampere trở lên dùng bfloat16, GPU cũ (T4, V100) dùng float16
    if not on_gpu:
        return "float32"
    return "bfloat16" if compute_capability is not None and compute_capability >= (8, 0) else "float16"


# Khóa chỉ faster-whisper có; transformers không có tương đương nên báo lỗi thay vì lặng lẽ bỏ qua.
_HF_UNSUPPORTED = frozenset({"vad_filter", "vad_parameters", "hotwords"})


def hf_generate_kwargs(gk: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Dịch `generate_kwargs` kiểu faster-whisper sang tham số `generate` của transformers; trả (kwargs, câu mồi).

    - `beam_size` -> `num_beams` (mặc định 1, tức tham lam); `do_sample=False` luôn bật.
    - `initial_prompt` tách riêng (nơi gọi đổi thành `prompt_ids`).
    - `condition_on_previous_text` -> `condition_on_prev_tokens` (chỉ có tác dụng với audio dài hơn 30 giây).
    - `best_of` bị bỏ: faster-whisper chỉ dùng nó khi lấy mẫu (temperature > 0), giải mã tất định thì vô tác dụng.
    - `vad_filter`, `vad_parameters`, `hotwords`: ValueError, vì transformers không có.
    """
    bad = sorted(k for k in gk if k in _HF_UNSUPPORTED and gk[k] not in (False, None, ""))
    if bad:
        raise ValueError(f"hf_seq2seq does not support generate_kwargs {bad} (faster_whisper only)")
    out = {k: v for k, v in gk.items() if k not in _HF_UNSUPPORTED | {"beam_size", "best_of", "initial_prompt"}}
    if "condition_on_previous_text" in out:
        out["condition_on_prev_tokens"] = out.pop("condition_on_previous_text")
    out["num_beams"] = int(gk.get("beam_size", out.get("num_beams", 1)))
    out["do_sample"] = False
    return out, str(gk.get("initial_prompt") or "")


class HFSeq2SeqASR:
    """Pipeline ASR của transformers (PhoWhisper), không cắt đoạn (cần extra `tokenizer` và `gpu`).

    `generate_kwargs` dùng cùng tên khóa với faster-whisper để hai engine chạy chung một bộ cấu hình
    (`hf_generate_kwargs` dịch sang tên của transformers). Mặc định giải mã tham lam; `do_sample=False` luôn bật
    để kết quả tất định. torch và transformers được import trong `__init__`.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from transformers import pipeline

        self._cfg = cfg
        # pipeline nhận chỉ số GPU (0) hoặc chuỗi "cpu"
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
        self._generate_kwargs, self._prompt = hf_generate_kwargs(cfg.generate_kwargs)
        if self._prompt:
            # Câu mồi: transformers nhận dạng token (`prompt_ids`) chứ không nhận chuỗi như faster-whisper.
            pipe: Any = self._pipe
            ids = pipe.tokenizer.get_prompt_ids(self._prompt, return_tensors="pt")
            self._generate_kwargs["prompt_ids"] = ids.to(pipe.device)

    def _run(self, items: list[dict[str, Any]]) -> list[str]:
        """Chạy pipeline trên danh sách đầu vào âm thanh; trả văn bản đã bỏ khoảng trắng hai đầu."""
        pipe: Any = self._pipe
        out = pipe(items, batch_size=self._cfg.batch_size, generate_kwargs=self._generate_kwargs)
        # Với một đầu vào duy nhất, pipeline trả dict thay vì list
        if isinstance(out, dict):
            out = [out]
        texts = [str(o["text"]).strip() for o in out]
        # Phòng khi pipeline trả kèm câu mồi ở đầu văn bản: bỏ đi để chỉ còn lời nghe được.
        if self._prompt:
            texts = [t[len(self._prompt) :].strip() if t.startswith(self._prompt) else t for t in texts]
        return texts

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Đọc mọi file WAV rồi chạy pipeline trên cả danh sách."""
        sr = self._cfg.sample_rate
        return self._run([{"raw": read_wav(p, sr), "sampling_rate": sr} for p in wav_paths])

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng."""
        import numpy as np

        sr = self._cfg.sample_rate
        self._run([{"raw": np.zeros(sr, dtype=np.float32), "sampling_rate": sr}])


class ChunkFormerASR:
    """ChunkFormer (khanhld/chunkformer-ctc-large-vie, khanhld/chunkformer-rnnt-large-vie) qua gói `chunkformer`.

    Giải mã tham lam (CTC hoặc RNNT, gói tự nhận loại mô hình từ config).

    Gói tự tải mô hình nhưng bỏ qua revision, nên engine tải trước snapshot của `cfg.model` đúng `cfg.revision` rồi
    nạp từ thư mục đó (`cfg.model` cũng có thể là thư mục cục bộ). `generate_kwargs` nhận chunk_size /
    left_context_size / right_context_size / total_batch_duration; các khóa chung của cấu hình kiểu Whisper
    (language, task, temperature = 0, beam_size = 1, condition_on_previous_text) bị bỏ qua vì không áp dụng cho CTC.
    Khóa khác gây ValueError. Bản CTC có giấy phép CC-BY-NC-4.0 (phi thương mại, cần BTC xác nhận trước khi nộp);
    bản RNNT có giấy phép CC-BY-4.0.
    """

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "chunk_size": 64,
        "left_context_size": 128,
        "right_context_size": 128,
        "total_batch_duration": 1800,
    }
    _IGNORED: ClassVar[frozenset[str]] = frozenset(
        {"language", "task", "temperature", "beam_size", "condition_on_previous_text"}
    )

    def __init__(self, cfg: ASRConfig) -> None:
        from chunkformer import ChunkFormerModel

        gk = dict(cfg.generate_kwargs)
        bad = sorted(k for k in gk if k not in self._IGNORED | self._DEFAULTS.keys())
        if bad or int(gk.get("beam_size", 1)) != 1 or gk.get("temperature", 0.0) not in (0, 0.0):
            raise ValueError(f"chunkformer supports only greedy CTC options; got generate_kwargs {sorted(gk)}")
        self._cfg = cfg
        path = cfg.model
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(cfg.model, revision=cfg.revision or None)
        self._model: Any = ChunkFormerModel.from_pretrained(path)
        self._kwargs = {**self._DEFAULTS, **{k: v for k, v in gk.items() if k in self._DEFAULTS}}

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Gói tự đọc file; mỗi đường dẫn một văn bản, theo thứ tự đầu vào."""
        out = self._model.batch_decode(audio_paths=[str(p) for p in wav_paths], **self._kwargs)
        return [str(t).strip() for t in out]

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng ghi ra file WAV tạm (gói chỉ nhận đường dẫn file)."""
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


class NemoCTCASR:
    """Parakeet CTC của NVIDIA (nvidia/parakeet-ctc-0.6b-Vietnamese) qua NeMo (`pip install nemo_toolkit[asr]`).

    `cfg.model` là repo HF chứa đúng một file `.nemo`, tải theo `cfg.revision`. Giải mã CTC tham lam.
    Không dùng `ASRModel.transcribe`: bộ nạp dữ liệu của nó (lhotse) lỗi với torch 2.13 (TypeError ở
    `Sampler.__init__`, kernel chạy thử 28/10/2026), nên engine tự đọc WAV, gọi `forward` rồi bộ giải mã CTC
    của mô hình, tắt dither như `transcribe` làm. CTC không có câu mồi, beam thường hay chống lặp; các khóa chung kiểu
    Whisper (language, task, temperature = 0, beam_size = 1, condition_on_previous_text) bị bỏ qua, khóa khác gây
    ValueError. Beam + KenLM đi kèm repo không dùng: gói flashlight-text không build được trên Python 3.13 của Kaggle.
    """

    _IGNORED: ClassVar[frozenset[str]] = frozenset(
        {"language", "task", "temperature", "beam_size", "condition_on_previous_text"}
    )

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from huggingface_hub import hf_hub_download, list_repo_files
        from nemo.collections.asr.models import ASRModel

        gk = dict(cfg.generate_kwargs)
        bad = sorted(k for k in gk if k not in self._IGNORED)
        if bad or int(gk.get("beam_size", 1)) != 1 or gk.get("temperature", 0.0) not in (0, 0.0):
            raise ValueError(f"nemo_ctc supports only greedy CTC; got generate_kwargs {sorted(gk)}")
        self._cfg = cfg
        rev = cfg.revision or None
        nemo_files = [f for f in list_repo_files(cfg.model, revision=rev) if f.endswith(".nemo")]
        if len(nemo_files) != 1:
            raise ValueError(f"asr.model: {cfg.model} must hold exactly one .nemo file, found {nemo_files}")
        on_gpu = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        self._device = torch.device("cuda" if on_gpu else "cpu")
        model: Any = ASRModel.restore_from(
            hf_hub_download(cfg.model, nemo_files[0], revision=rev), map_location=self._device
        )
        cc = tuple(torch.cuda.get_device_capability(0)) if on_gpu else None
        self.dtype = resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        self._autocast = on_gpu and self.dtype != "float32"
        # Như ASRModel.transcribe: tắt dither và đệm khung khi suy luận để kết quả tất định.
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        self._model = model.eval()
        self._torch = torch

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Đọc WAV, đệm thành một lô, chạy encoder rồi giải mã CTC tham lam; một văn bản mỗi file, đúng thứ tự."""
        torch = self._torch
        audios = [read_wav(p, self._cfg.sample_rate) for p in wav_paths]
        lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
        x = torch.zeros(len(audios), int(lens.max()), dtype=torch.float32)
        for i, a in enumerate(audios):
            x[i, : len(a)] = torch.from_numpy(a)
        dev = self._device
        with torch.no_grad(), torch.autocast("cuda", dtype=getattr(torch, self.dtype), enabled=self._autocast):
            log_probs, enc_len, _ = self._model.forward(input_signal=x.to(dev), input_signal_length=lens.to(dev))
        out = self._model.decoding.ctc_decoder_predictions_tensor(log_probs.float(), decoder_lengths=enc_len)
        if isinstance(out, tuple):  # một số phiên bản trả (best, all)
            out = out[0]
        return [str(getattr(o, "text", o)).strip() for o in out]

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng."""
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


class FasterWhisperASR:
    """Whisper trên CTranslate2 (`faster-whisper`, extra `asr-faster-whisper`), xử lý từng file một.

    Team lead's choice (ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/inference.py@7033c9f): whisper-large-v3,
    beam 5, VAD, a domain `initial_prompt`, no conditioning on previous text.
    Lựa chọn của trưởng nhóm: whisper-large-v3, beam 5, VAD, `initial_prompt` theo miền bài toán, không điều kiện
    hóa theo văn bản trước đó. Tùy chọn giải mã lấy nguyên từ `[asr].generate_kwargs`; `cfg.model` là repo mô hình
    CTranslate2 (ví dụ Systran/faster-whisper-large-v3) ghim ở `cfg.revision`, hoặc thư mục cục bộ chứa mô hình đã
    chuyển đổi (Docker: /app/models/asr).
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from faster_whisper import WhisperModel  # type: ignore[import-not-found,unused-ignore]  # extra asr-faster-whisper

        self._cfg = cfg
        on_gpu = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        cc = tuple(torch.cuda.get_device_capability(0)) if on_gpu else None
        self.dtype = resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        # WhisperModel không nhận revision: khi có revision và model không phải thư mục cục bộ,
        # tải snapshot đúng revision rồi nạp từ đường dẫn đó
        path = cfg.model
        if cfg.revision and not Path(cfg.model).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(cfg.model, revision=cfg.revision)
        self._model = WhisperModel(path, device="cuda" if on_gpu else "cpu", compute_type=self.dtype)
        self._kwargs = dict(cfg.generate_kwargs)

    def _one(self, audio: Any) -> str:
        """Nhận dạng một câu nói; nối văn bản các đoạn (segment) bằng dấu cách."""
        # `segments` là generator: việc giải mã thực sự diễn ra khi duyệt qua
        segments, _info = self._model.transcribe(audio, **self._kwargs)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Nhận dạng từng file WAV theo thứ tự."""
        sr = self._cfg.sample_rate
        return [self._one(read_wav(p, sr)) for p in wav_paths]

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng."""
        import numpy as np

        self._one(np.zeros(self._cfg.sample_rate, dtype=np.float32))


# Registry theo tên: giá trị `[asr].engine` -> lớp backend (lớp chỉ được khởi tạo khi được chọn)
ASR_ENGINES: dict[str, Callable[[ASRConfig], ASRBackend]] = {
    "faster_whisper": FasterWhisperASR,
    "hf_seq2seq": HFSeq2SeqASR,
    "chunkformer": ChunkFormerASR,
    "nemo_ctc": NemoCTCASR,
}


def make_backend(cfg: ASRConfig) -> ASRBackend:
    """Tạo backend mô hình ASR theo tên `cfg.engine`; tên lạ gây ValueError."""
    try:
        factory = ASR_ENGINES[cfg.engine]
    except KeyError:
        raise ValueError(f"asr.engine: unknown engine {cfg.engine!r}") from None
    return factory(cfg)


class ASRTranscriber:
    """ASR thật trên các file WAV của mẫu (SPEC 6.1); mỗi mẫu gửi backend theo lô.

    Có thể truyền sẵn `backend` (dùng trong test); nếu không, backend được tạo bằng `make_backend`.
    """

    source = "asr"

    def __init__(self, cfg: ASRConfig, *, data_root: Path, split: str, backend: ASRBackend | None = None) -> None:
        self._cfg = cfg
        self._base = Path(data_root) / split
        self._backend: ASRBackend = backend if backend is not None else make_backend(cfg)
        self._warmed = False

    def wav_paths(self, sample: Sample) -> list[Path]:
        """Các WAV của lượt user theo thứ tự lịch sử (_u1, _u2, ...) rồi đến yêu cầu cuối.

        Dùng đường dẫn `audio` của mẫu khi có, nếu không thì suy ra theo quy ước tên file của bộ dữ liệu.
        """
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
        """Nhận dạng các lượt user trong lịch sử và lượt cuối; None cho lượt assistant.

        File được chia thành từng lô `batch_size`; số văn bản trả về phải khớp số file. Khi bật `sanitize`,
        mọi lượt trừ lượt cuối được làm sạch theo kiểu lịch sử (`is_history=True`).
        """
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
        raw = tuple(texts)
        if self._cfg.sanitize:
            texts = [sanitize(t, is_history=i < len(texts) - 1) for i, t in enumerate(texts)]
        # Phân phát lần lượt văn bản lịch sử cho các lượt user; lượt assistant nhận None
        it = iter(texts[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        return Transcript(final=texts[-1], history=history, seconds=time.perf_counter() - start, raw=raw)

    def warmup(self) -> None:
        """Khởi động backend một lần duy nhất."""
        if not self._warmed:
            self._backend.warmup()
            self._warmed = True

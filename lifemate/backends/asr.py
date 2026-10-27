"""Các bộ chuyển giọng nói thành văn bản (transcriber) cho agent (SPEC 4.4, 6.1).

Nội dung:
- `GoldTranscriber`: trả lại `audio_text` gốc của tập train, dùng khi muốn tách lỗi ASR khỏi lỗi LLM.
- `ASRTranscriber`: chạy ASR thật trên các file WAV của mẫu, mỗi mẫu một lô (batch) gửi xuống backend.
- Các backend mô hình theo giao thức `ASRBackend`: `FasterWhisperASR`, `HFSeq2SeqASR`, `ChunkFormerASR`,
  `VieGemASR`.

Cách xây dựng:
- Thư viện nặng (torch, transformers, faster_whisper, chunkformer, numpy) chỉ được import bên trong hàm khởi tạo
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


class HFSeq2SeqASR:
    """Pipeline ASR của transformers, giải mã tham lam (greedy), không cắt đoạn (cần extra `tokenizer` và `gpu`).

    torch và transformers được import trong `__init__`; `num_beams=1`, `do_sample=False` luôn ghi đè
    `generate_kwargs` của cấu hình để kết quả tất định.
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
        self._generate_kwargs = {**cfg.generate_kwargs, "num_beams": 1, "do_sample": False}

    def _run(self, items: list[dict[str, Any]]) -> list[str]:
        """Chạy pipeline trên danh sách đầu vào âm thanh; trả văn bản đã bỏ khoảng trắng hai đầu."""
        pipe: Any = self._pipe
        out = pipe(items, batch_size=self._cfg.batch_size, generate_kwargs=self._generate_kwargs)
        # Với một đầu vào duy nhất, pipeline trả dict thay vì list
        if isinstance(out, dict):
            out = [out]
        return [str(o["text"]).strip() for o in out]

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
    """ChunkFormer CTC (khanhld/chunkformer-ctc-large-vie) qua gói `chunkformer`; giải mã CTC tham lam.

    `cfg.model` là id trên HF hoặc thư mục snapshot cục bộ; gói bỏ qua `cfg.revision`, nên cần ghim thư mục
    snapshot trong Docker. `generate_kwargs` có thể ghi đè chunk_size / left_context_size /
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
        """Gói tự đọc file; mỗi đường dẫn một văn bản, theo thứ tự đầu vào."""
        out = self._model.batch_decode(audio_paths=[str(p) for p in wav_paths], **self._kwargs)
        return [str(t).strip() for t in out]

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng ghi ra file WAV tạm.

        Gói chỉ nhận đường dẫn file, không nhận mảng âm thanh, nên phải ghi WAV ra thư mục tạm.
        """
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
    """VieGem-ASR (5CD-AI/VieGem-ASR): CTC streaming trên bộ mã hóa âm thanh EmbeddingGemma 2; tham lam, chế độ offline.

    `cfg.model` là thư mục snapshot cục bộ chứa `viegem_asr.pt` và thư mục mã `asr/` (repo kèm mã mô hình riêng,
    không phải kiến trúc transformers). Các file được giải mã lần lượt từng file: mã công bố chỉ nhận một câu nói,
    và clip ≤ 30 giây giữ chi phí này ở mức thấp.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import sys

        import torch
        from transformers import AutoTokenizer

        root = Path(cfg.model)
        if not (root / "viegem_asr.pt").is_file():
            raise ValueError(f"asr.model: {root} has no viegem_asr.pt (engine = 'viegem' needs the snapshot folder)")
        # Mã mô hình nằm trong snapshot (`asr/`), nên thêm thư mục snapshot vào sys.path trước khi import
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
        # strict=False: checkpoint không nhất thiết chứa đủ mọi tham số của CTCModel
        model.load_state_dict(ck["model"], strict=False)
        self._model = model.to(self._device).eval()
        self._greedy_decode = greedy_decode
        self._torch = torch

    def _one(self, audio: Any) -> str:
        """Giải mã một câu nói (mảng float32) thành văn bản; trên CUDA chạy autocast bfloat16."""
        torch = self._torch
        # Thêm chiều batch = 1; `lens` là độ dài thực của câu nói
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
        """Giải mã từng file WAV theo thứ tự."""
        sr = self._cfg.sample_rate
        return [self._one(read_wav(p, sr)) for p in wav_paths]

    def warmup(self) -> None:
        """Chạy một lượt trên 1 giây im lặng."""
        import numpy as np

        self._one(np.zeros(self._cfg.sample_rate, dtype=np.float32))


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
    "viegem": VieGemASR,
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
        if self._cfg.sanitize:
            texts = [sanitize(t, is_history=i < len(texts) - 1) for i, t in enumerate(texts)]
        # Phân phát lần lượt văn bản lịch sử cho các lượt user; lượt assistant nhận None
        it = iter(texts[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        return Transcript(final=texts[-1], history=history, seconds=time.perf_counter() - start)

    def warmup(self) -> None:
        """Khởi động backend một lần duy nhất."""
        if not self._warmed:
            self._backend.warmup()
            self._warmed = True

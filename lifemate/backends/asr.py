"""Các transcriber chuyển giọng nói của mẫu thành văn bản cho agent, cùng các backend model ASR (SPEC 4.4, 6.1).

Nội dung chính:
- `GoldTranscriber`: trả lại `audio_text` gốc của tập train, dùng khi muốn tách lỗi ASR khỏi lỗi LLM.
- `ASRTranscriber`: chạy ASR thật trên các file WAV của mẫu, mỗi mẫu gửi xuống backend theo batch.
- `ASRBackend`: Protocol chung của các backend model.
- `PromptedASRBackend`: backend có câu mồi và chép lại được không câu mồi (cho lớp bảo vệ câu mồi).
- `FasterWhisperASR`, `HFSeq2SeqASR`, `ChunkFormerASR`, `NemoCTCASR`: bốn backend model ASR.
- `ASR_ENGINES`, `make_backend`: chọn backend theo tên trong `[asr].engine`.
- `read_wav`, `resolve_asr_dtype`, `hf_generate_kwargs`: hàm phụ đọc audio, chọn dtype, dịch tham số giải mã.

Cách hoạt động:
- Từ mẫu, lập danh sách file WAV: các lượt user trong lịch sử trước, lượt cuối sau.
- Backend trả về văn bản theo đúng thứ tự file.
- Nếu `[asr].prompt_guard_ngram` bật, câu nào hỏng do câu mồi (`asr_guard.needs_retry`) được chép lại không câu mồi.
- Nếu `[asr].sanitize` bật, văn bản được lọc ảo giác bằng `sanitize`.
- Kết quả là `Transcript` gồm văn bản lượt cuối và lịch sử (`None` ở lượt assistant).

Lưu ý:
- Thư viện nặng (torch, transformers, faster_whisper, chunkformer, nemo, numpy) chỉ được import bên trong hàm
  khởi tạo hoặc hàm cần dùng. Nhờ vậy import module này không cần extra `gpu`, và test chạy được trên CPU.
- `make_backend` báo lỗi rõ ràng khi tên engine không có trong `ASR_ENGINES`.
- `resolve_asr_dtype` chọn dtype theo GPU để model ASR vừa bộ nhớ khi chạy chung GPU với vLLM.
"""

from __future__ import annotations

import time
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

import structlog

from lifemate.backends.asr_guard import needs_retry
from lifemate.backends.asr_sanitizer import sanitize
from lifemate.core.config import ASRConfig
from lifemate.core.types import Sample, Transcript

log = structlog.get_logger(__name__)


class GoldTranscriber:
    """Trả về `audio_text` gốc (chỉ tập train có) thay cho kết quả ASR.

    Dùng để đo riêng chất lượng LLM, khi giả định ASR không sai chữ nào.
    """

    def transcribe(self, sample: Sample) -> Transcript:
        """Lấy văn bản gốc của lượt cuối và của mọi lượt user trong lịch sử.

        Tham số:
            sample: mẫu cần lấy văn bản.

        Trả về:
            `Transcript` với `history` là `None` ở các lượt assistant.

        Lỗi:
            ValueError: khi lượt cuối hoặc một lượt user trong lịch sử không có `audio_text`.
        """
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
        """Không cần làm gì."""


class ASRBackend(Protocol):
    """Model chuyển giọng nói thành văn bản: mỗi file đầu vào cho một văn bản, giữ nguyên thứ tự."""

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Nhận dạng mọi file WAV và trả về danh sách văn bản theo thứ tự đầu vào."""
        ...

    def warmup(self) -> None:
        """Chạy model một lần trên 1 giây im lặng để khởi động."""
        ...


class PromptedASRBackend(ASRBackend, Protocol):
    """Backend có câu mồi (Whisper) và chép lại được không câu mồi, cho lớp bảo vệ câu mồi."""

    def transcribe_batch_unprompted(self, wav_paths: Sequence[Path]) -> list[str]:
        """Như `transcribe_batch` nhưng bỏ câu mồi (`initial_prompt`, `hotwords`)."""
        ...


def read_wav(path: Path, expected_rate: int) -> Any:
    """Đọc file WAV PCM 16-bit thành mảng numpy float32 trong khoảng [-1, 1].

    File nhiều kênh được lấy trung bình các kênh thành mono. Hàm không tự resample.

    Tham số:
        path: đường dẫn file WAV.
        expected_rate: tần số lấy mẫu bắt buộc, đơn vị Hz.

    Trả về:
        Mảng numpy float32 một chiều.

    Lỗi:
        ValueError: khi độ sâu bit khác 16, hoặc tần số lấy mẫu khác `expected_rate`.
    """
    import numpy as np  # numpy đi kèm transformers (extra `gpu`)

    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    if rate != expected_rate:
        raise ValueError(f"{path}: expected {expected_rate} Hz, got {rate} Hz")
    # Mẫu int16 little-endian; chia cho 32768 để đưa về khoảng [-1, 1].
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        # Dữ liệu các kênh nằm xen kẽ nhau, nên reshape thành (số khung, số kênh) rồi lấy trung bình theo hàng.
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def resolve_asr_dtype(cfg_dtype: str, on_gpu: bool, compute_capability: tuple[int, int] | None) -> str:
    """Chọn dtype torch cho model ASR: `auto` thành bfloat16 trên sm80 trở lên, float16 trên GPU cũ hơn, float32 trên
    CPU.

    Thiếu bước này, pipeline transformers nạp model ở float32: khoảng 3 GB với PhoWhisper-medium, không vừa khi
    chạy cạnh vLLM trên T4 15 GB, và tốn cả bộ nhớ lẫn thời gian trên A30.

    Tham số:
        cfg_dtype: giá trị `asr.dtype` trong config.
        on_gpu: `True` khi model chạy trên GPU.
        compute_capability: compute capability của GPU; `None` khi không biết hoặc chạy CPU.

    Trả về:
        Tên dtype: `"float32"`, `"float16"` hoặc `"bfloat16"`.

    Lỗi:
        ValueError: khi config chọn dtype khác float32 mà không có GPU, hoặc chọn bfloat16 trên GPU dưới sm80.
    """
    if cfg_dtype != "auto":
        if cfg_dtype != "float32" and not on_gpu:
            raise ValueError(f"asr.dtype = {cfg_dtype!r} needs a GPU; use 'auto' or 'float32' on CPU")
        if cfg_dtype == "bfloat16" and compute_capability is not None and compute_capability < (8, 0):
            cc = compute_capability
            raise ValueError(f"asr.dtype = 'bfloat16' needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
        return cfg_dtype
    # Chế độ `auto`: CPU luôn dùng float32; GPU Ampere trở lên dùng bfloat16; GPU cũ (T4, V100) dùng float16.
    if not on_gpu:
        return "float32"
    return "bfloat16" if compute_capability is not None and compute_capability >= (8, 0) else "float16"


# Các khoá chỉ faster-whisper có. Transformers không có tham số tương đương, nên báo lỗi thay vì lặng lẽ bỏ qua.
_HF_UNSUPPORTED = frozenset({"vad_filter", "vad_parameters", "hotwords"})


def hf_generate_kwargs(gk: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Dịch `generate_kwargs` kiểu faster-whisper sang tham số `generate` của transformers.

    Quy tắc dịch:
    - `beam_size` thành `num_beams` (mặc định 1, tức giải mã tham lam); `do_sample=False` luôn bật.
    - `initial_prompt` được tách riêng; nơi gọi đổi câu mồi thành `prompt_ids`.
    - `condition_on_previous_text` thành `condition_on_prev_tokens` (chỉ có tác dụng với audio dài hơn 30 giây).
    - `best_of` bị bỏ: faster-whisper chỉ dùng khoá này khi lấy mẫu (temperature > 0), còn giải mã tất định thì
      khoá này vô tác dụng.
    - `vad_filter`, `vad_parameters`, `hotwords` gây `ValueError`, vì transformers không có.

    Tham số:
        gk: `[asr].generate_kwargs` theo tên khoá của faster-whisper.

    Trả về:
        Cặp (kwargs cho transformers, câu mồi `initial_prompt` hoặc chuỗi rỗng).

    Lỗi:
        ValueError: khi `gk` bật một khoá chỉ faster-whisper hỗ trợ.
    """
    # Khoá có mặt nhưng đang tắt (`False`, `None`, chuỗi rỗng) thì bỏ qua được, không cần báo lỗi.
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
    """Backend dùng pipeline ASR của transformers (PhoWhisper), không cắt đoạn; cần extra `tokenizer` và `gpu`.

    `generate_kwargs` dùng cùng tên khoá với faster-whisper để hai engine dùng chung một bộ config;
    `hf_generate_kwargs` dịch sang tên của transformers. Mặc định giải mã tham lam, và `do_sample=False` luôn bật
    để kết quả tất định. torch và transformers chỉ được import trong `__init__`.

    Tham số:
        cfg: mục `[asr]` của config.

    Thuộc tính:
        dtype: tên dtype đã chọn cho model.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from transformers import pipeline

        self._cfg = cfg
        # `pipeline` nhận chỉ số GPU (0) hoặc chuỗi "cpu".
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
            # Câu mồi: transformers nhận dạng token (`prompt_ids`), không nhận chuỗi như faster-whisper.
            pipe: Any = self._pipe
            ids = pipe.tokenizer.get_prompt_ids(self._prompt, return_tensors="pt")
            self._generate_kwargs["prompt_ids"] = ids.to(pipe.device)

    def _run(self, items: list[dict[str, Any]], prompted: bool = True) -> list[str]:
        """Chạy pipeline trên danh sách audio đầu vào và trả về văn bản đã bỏ khoảng trắng hai đầu.

        `prompted=False` bỏ câu mồi (`prompt_ids`), dùng cho lớp bảo vệ câu mồi.
        """
        pipe: Any = self._pipe
        gk = (
            self._generate_kwargs if prompted else {k: v for k, v in self._generate_kwargs.items() if k != "prompt_ids"}
        )
        out = pipe(items, batch_size=self._cfg.batch_size, generate_kwargs=gk)
        # Khi chỉ có một đầu vào, pipeline trả về dict thay vì list.
        if isinstance(out, dict):
            out = [out]
        texts = [str(o["text"]).strip() for o in out]
        # Pipeline có thể trả kèm câu mồi ở đầu văn bản; cắt bỏ để chỉ còn lời thật sự nghe được.
        if self._prompt and prompted:
            texts = [t[len(self._prompt) :].strip() if t.startswith(self._prompt) else t for t in texts]
        return texts

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Đọc mọi file WAV rồi chạy pipeline trên cả danh sách."""
        sr = self._cfg.sample_rate
        return self._run([{"raw": read_wav(p, sr), "sampling_rate": sr} for p in wav_paths])

    def transcribe_batch_unprompted(self, wav_paths: Sequence[Path]) -> list[str]:
        """Như `transcribe_batch` nhưng không câu mồi."""
        sr = self._cfg.sample_rate
        return self._run([{"raw": read_wav(p, sr), "sampling_rate": sr} for p in wav_paths], prompted=False)

    def warmup(self) -> None:
        """Chạy một lần trên 1 giây im lặng."""
        import numpy as np

        sr = self._cfg.sample_rate
        self._run([{"raw": np.zeros(sr, dtype=np.float32), "sampling_rate": sr}])


class ChunkFormerASR:
    """Backend ChunkFormer (`khanhld/chunkformer-ctc-large-vie`, `khanhld/chunkformer-rnnt-large-vie`) qua gói
    `chunkformer`.

    Giải mã tham lam, CTC hoặc RNNT; gói tự nhận loại model từ config.

    Gói tự tải model nhưng bỏ qua revision. Vì vậy engine tải trước snapshot của `cfg.model` đúng `cfg.revision`
    rồi nạp từ thư mục đó. `cfg.model` cũng có thể là thư mục trên máy.

    `generate_kwargs` nhận `chunk_size`, `left_context_size`, `right_context_size`, `total_batch_duration`. Các khoá
    chung của config kiểu Whisper (`language`, `task`, `temperature = 0`, `beam_size = 1`,
    `condition_on_previous_text`) bị bỏ qua vì không áp dụng cho CTC. Khoá khác gây `ValueError`.

    Giấy phép: bản CTC là CC-BY-NC-4.0 (phi thương mại, cần ban tổ chức xác nhận trước khi nộp); bản RNNT là
    CC-BY-4.0.

    Tham số:
        cfg: mục `[asr]` của config.

    Lỗi:
        ValueError: khi `generate_kwargs` có khoá lạ, hoặc yêu cầu beam search hay lấy mẫu (temperature khác 0).
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
        # `cfg.model` là thư mục trên máy thì nạp thẳng; không thì tải snapshot đúng revision từ Hugging Face.
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(cfg.model, revision=cfg.revision or None)
        self._model: Any = ChunkFormerModel.from_pretrained(path)
        self._kwargs = {**self._DEFAULTS, **{k: v for k, v in gk.items() if k in self._DEFAULTS}}

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Nhận dạng các file WAV (gói tự đọc file) và trả về một văn bản cho mỗi file, theo thứ tự đầu vào."""
        out = self._model.batch_decode(audio_paths=[str(p) for p in wav_paths], **self._kwargs)
        return [str(t).strip() for t in out]

    def warmup(self) -> None:
        """Chạy một lần trên 1 giây im lặng, ghi ra file WAV tạm vì gói chỉ nhận đường dẫn file."""
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
    """Backend Parakeet CTC của NVIDIA (`nvidia/parakeet-ctc-0.6b-Vietnamese`) qua NeMo (`pip install
    nemo_toolkit[asr]`).

    `cfg.model` là repo HF chứa đúng một file `.nemo`, tải theo `cfg.revision`. Giải mã CTC tham lam.

    Engine không dùng `ASRModel.transcribe`, vì bộ nạp dữ liệu của hàm đó (lhotse) lỗi với torch 2.13 (`TypeError`
    ở `Sampler.__init__`, phát hiện khi chạy thử kernel ngày 28/10/2026). Thay vào đó, engine tự đọc WAV, gọi
    `forward`, rồi gọi bộ giải mã CTC của model, đồng thời tắt dither giống như `transcribe`.

    CTC không có câu mồi, beam search hay chống lặp như Whisper. Các khoá chung kiểu Whisper (`language`, `task`,
    `temperature = 0`, `beam_size = 1`, `condition_on_previous_text`) bị bỏ qua; khoá khác gây `ValueError`.

    Beam search + KenLM đi kèm repo không được dùng: gói flashlight-text không build được trên Python 3.13 của Kaggle.

    Tham số:
        cfg: mục `[asr]` của config.

    `dtype = "auto"` chạy float32 (float16 trên T4 cho đầu ra NaN, xem `__init__`). Với float16/bfloat16 tường minh,
    câu nào ra NaN/inf được chạy lại ở float32.

    Thuộc tính:
        dtype: tên dtype đã chọn cho model.

    Lỗi:
        ValueError: khi `generate_kwargs` có khoá lạ hoặc yêu cầu giải mã khác tham lam, hoặc repo không chứa đúng
            một file `.nemo`.
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
        # `auto` chọn float32: chạy float16 trên T4 cho đầu ra NaN ở nhiều câu (chẩn đoán 29/10/2026 trên 216 câu hỏng
        # của lần chạy trước: 70 câu ra `⁇` ở float16, 0 câu ở float32, và float32 còn nhanh hơn). Chọn float16 hoặc
        # bfloat16 tường minh thì vẫn chạy autocast, có đường dự phòng float32 ở `transcribe_batch`.
        self.dtype = "float32" if cfg.dtype == "auto" else resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        self._autocast = on_gpu and self.dtype != "float32"
        # Giống `ASRModel.transcribe`: tắt dither (nhiễu ngẫu nhiên thêm vào đặc trưng) và đệm khung khi suy luận, để
        # kết quả tất định.
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        self._model = model.eval()
        self._torch = torch

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Nhận dạng một batch file WAV bằng encoder và bộ giải mã CTC tham lam.

        Các file được đọc, đệm số 0 cho bằng độ dài file dài nhất, chạy encoder một lần cho cả batch, rồi giải mã.

        Tham số:
            wav_paths: các file WAV cần nhận dạng.

        Trả về:
            Một văn bản cho mỗi file, đúng thứ tự đầu vào.
        """
        torch = self._torch
        audios = [read_wav(p, self._cfg.sample_rate) for p in wav_paths]
        # Độ dài thật của từng file, để model bỏ qua phần đệm số 0 ở cuối.
        lens = torch.tensor([len(a) for a in audios], dtype=torch.long)
        x = torch.zeros(len(audios), int(lens.max()), dtype=torch.float32)
        for i, a in enumerate(audios):
            x[i, : len(a)] = torch.from_numpy(a)
        log_probs, enc_len = self._forward(x, lens, self._autocast)
        log_probs = log_probs.float()
        # Câu có đầu ra NaN/inf (tràn số khi chạy autocast) được chạy lại ở float32 từng câu một, kèm cảnh báo.
        bad = (~torch.isfinite(log_probs)).flatten(1).any(dim=1)
        if bool(bad.any()):
            rows = int(bad.sum())
            log.warning("asr_nonfinite_output", engine="nemo_ctc", dtype=self.dtype, rows=rows, batch=len(audios))
        if self._autocast:
            for i in bad.nonzero().flatten().tolist():
                lp, el = self._forward(x[i : i + 1, : int(lens[i])], lens[i : i + 1], False)
                n = int(el[0])
                log_probs[i].fill_(0.0)
                log_probs[i, :n] = lp[0, :n].float()
                enc_len[i] = n
        out = self._model.decoding.ctc_decoder_predictions_tensor(log_probs.float(), decoder_lengths=enc_len)
        if isinstance(out, tuple):  # một số phiên bản NeMo trả về cặp (best, all)
            out = out[0]
        return [str(getattr(o, "text", o)).strip() for o in out]

    def _forward(self, x: Any, lens: Any, autocast: bool) -> tuple[Any, Any]:
        """Chạy model trên batch `x` (độ dài thật `lens`), có hoặc không autocast; trả (log_probs, độ dài đầu ra)."""
        torch, dev = self._torch, self._device
        dtype = getattr(torch, self.dtype) if autocast else torch.float32
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=autocast):
            log_probs, enc_len, _ = self._model.forward(input_signal=x.to(dev), input_signal_length=lens.to(dev))
        return log_probs, enc_len

    def warmup(self) -> None:
        """Chạy một lần trên 1 giây im lặng (ghi ra file WAV tạm)."""
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
    """Backend Whisper chạy trên CTranslate2 (`faster-whisper`, extra `asr-faster-whisper`), xử lý từng file một.

    Theo lựa chọn của trưởng nhóm (ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/inference.py@7033c9f):
    whisper-large-v3, beam 5, VAD, `initial_prompt` theo miền bài toán, không dựa vào văn bản của đoạn trước.

    Tuỳ chọn giải mã lấy nguyên từ `[asr].generate_kwargs`. `cfg.model` là repo model CTranslate2 (ví dụ
    `Systran/faster-whisper-large-v3`) ghim ở `cfg.revision`, hoặc thư mục trên máy chứa model đã chuyển đổi
    (trong Docker: `/app/models/asr`).

    Tham số:
        cfg: mục `[asr]` của config.

    Thuộc tính:
        dtype: tên dtype đã chọn, truyền cho CTranslate2 làm `compute_type`.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        import torch
        from faster_whisper import WhisperModel  # type: ignore[import-not-found,unused-ignore]  # extra asr-faster-whisper

        self._cfg = cfg
        on_gpu = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
        cc = tuple(torch.cuda.get_device_capability(0)) if on_gpu else None
        self.dtype = resolve_asr_dtype(cfg.dtype, on_gpu, cc)
        # `WhisperModel` không nhận revision. Vì vậy khi có revision và model không phải thư mục trên máy, tải snapshot
        # đúng revision rồi nạp từ đường dẫn đó.
        path = cfg.model
        if cfg.revision and not Path(cfg.model).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(cfg.model, revision=cfg.revision)
        self._model = WhisperModel(path, device="cuda" if on_gpu else "cpu", compute_type=self.dtype)
        self._kwargs = dict(cfg.generate_kwargs)
        # Cùng tham số nhưng bỏ câu mồi, cho lớp bảo vệ câu mồi.
        self._unprompted = {k: v for k, v in self._kwargs.items() if k not in ("initial_prompt", "hotwords")}

    def _one(self, audio: Any, prompted: bool = True) -> str:
        """Nhận dạng một câu nói và nối văn bản các đoạn (segment) bằng dấu cách."""
        return self.transcribe_one(audio, None if prompted else "")[0]

    def transcribe_one(self, audio: Any, prompt: str | None = None) -> tuple[str, dict[str, float]]:
        """Nhận dạng một câu nói với câu mồi `prompt`, trả về văn bản và điểm tin cậy của large-v3.

        Tham số:
            audio: mảng float32 của câu nói.
            prompt: `None` dùng câu mồi của config; chuỗi rỗng là không câu mồi; chuỗi khác thay câu mồi của config
                (câu mồi theo mẫu, benchmark ASR lần 3).

        Trả về:
            Cặp (văn bản, điểm): `avg_logprob` (trung bình theo số token của các đoạn), `no_speech_prob` (lớn nhất
            trong các đoạn), `n_segments`. Không có đoạn nào thì điểm là 0.
        """
        if prompt is None:
            kwargs = self._kwargs
        elif prompt == "":
            kwargs = self._unprompted
        else:
            kwargs = {**self._unprompted, "initial_prompt": prompt}
        # `segments` là generator: model chỉ thật sự giải mã khi generator được duyệt qua.
        segments, _info = self._model.transcribe(audio, **kwargs)
        segs = list(segments)
        # Số token của từng đoạn làm trọng số; đoạn thiếu trường (model giả trong test) tính 1 token, điểm 0.
        weights = [max(1, len(getattr(s, "tokens", ()) or ())) for s in segs]
        logprobs = [float(getattr(s, "avg_logprob", 0.0)) for s in segs]
        scores = {
            "avg_logprob": sum(w * lp for w, lp in zip(weights, logprobs, strict=True)) / sum(weights) if segs else 0.0,
            "no_speech_prob": max((float(getattr(s, "no_speech_prob", 0.0)) for s in segs), default=0.0),
            "n_segments": float(len(segs)),
        }
        return " ".join(s.text.strip() for s in segs).strip(), scores

    def score_texts(self, audio: Any, texts: Sequence[str]) -> list[tuple[float, int]]:
        """Điểm âm thanh của từng câu cho sẵn: tổng log xác suất mà large-v3, không câu mồi, gán cho các token của câu.

        Không sinh chữ: chạy encoder một lần trên 30 giây đầu của câu nói, rồi ép chuỗi từng câu qua decoder bằng
        `align` của CTranslate2 (hàm faster-whisper dùng cho `word.probability`), với các token mở đầu
        `<|startoftranscript|><|vi|><|transcribe|>` và không có câu mồi.

        Tham số:
            audio: mảng float32 của câu nói.
            texts: các câu cần chấm.

        Trả về:
            Mỗi câu một cặp (tổng log xác suất các token mà `align` trả về, số token đó).
        """
        import math

        from faster_whisper.audio import pad_or_trim  # type: ignore[import-not-found,unused-ignore]
        from faster_whisper.tokenizer import Tokenizer  # type: ignore[import-not-found,unused-ignore]

        model: Any = self._model
        fe = model.feature_extractor
        features = fe(audio)
        num_frames = min(fe.nb_max_frames, features.shape[-1] - 1)
        encoder_output = model.encode(pad_or_trim(features[:, :num_frames]))
        language = str(self._kwargs.get("language") or "vi")
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language)
        out: list[tuple[float, int]] = []
        for text in texts:
            tokens = tokenizer.encode(" " + text.strip())
            if not tokens:
                out.append((0.0, 0))
                continue
            result = model.model.align(encoder_output, tokenizer.sot_sequence, [tokens], num_frames)[0]
            probs = list(result.text_token_probs)
            out.append((sum(math.log(max(p, 1e-12)) for p in probs), len(probs)))
        return out

    def transcribe_batch(self, wav_paths: Sequence[Path], prompts: Sequence[str] | None = None) -> list[str]:
        """Nhận dạng lần lượt từng file WAV theo thứ tự; `prompts` (nếu có) là câu mồi riêng cho từng file."""
        sr = self._cfg.sample_rate
        if prompts is None:
            return [self._one(read_wav(p, sr)) for p in wav_paths]
        return [self.transcribe_one(read_wav(p, sr), q or "")[0] for p, q in zip(wav_paths, prompts, strict=True)]

    def transcribe_batch_unprompted(self, wav_paths: Sequence[Path]) -> list[str]:
        """Như `transcribe_batch` nhưng không câu mồi."""
        sr = self._cfg.sample_rate
        return [self._one(read_wav(p, sr), prompted=False) for p in wav_paths]

    def warmup(self) -> None:
        """Chạy một lần trên 1 giây im lặng."""
        import numpy as np

        self._one(np.zeros(self._cfg.sample_rate, dtype=np.float32))


# Registry theo tên: giá trị `[asr].engine` -> lớp backend. Lớp chỉ được khởi tạo khi được chọn.
ASR_ENGINES: dict[str, Callable[[ASRConfig], ASRBackend]] = {
    "faster_whisper": FasterWhisperASR,
    "hf_seq2seq": HFSeq2SeqASR,
    "chunkformer": ChunkFormerASR,
    "nemo_ctc": NemoCTCASR,
}


def make_backend(cfg: ASRConfig) -> ASRBackend:
    """Tạo backend model ASR theo tên `cfg.engine`.

    Lỗi:
        ValueError: khi `cfg.engine` không có trong `ASR_ENGINES`.
    """
    try:
        factory = ASR_ENGINES[cfg.engine]
    except KeyError:
        raise ValueError(f"asr.engine: unknown engine {cfg.engine!r}") from None
    return factory(cfg)


class ASRTranscriber:
    """Chạy ASR thật trên các file WAV của mẫu (SPEC 6.1), mỗi mẫu gửi xuống backend theo batch.

    Test có thể truyền sẵn `backend`; nếu không, backend được tạo bằng `make_backend`.

    Tham số:
        cfg: mục `[asr]` của config.
        data_root: thư mục dữ liệu cuộc thi.
        split: tên split; file audio nằm trong `data_root/split`.
        backend: backend có sẵn; `None` thì tạo mới từ `cfg`.

    Thuộc tính:
        source: nguồn transcript ghi vào trace, luôn là `"asr"`.
        n_prompt_retries: số câu đã chép lại không câu mồi do lớp bảo vệ câu mồi.

    Lỗi:
        ValueError: khi bật `prompt_guard_ngram` mà backend không chép lại được không câu mồi.
    """

    source = "asr"

    def __init__(self, cfg: ASRConfig, *, data_root: Path, split: str, backend: ASRBackend | None = None) -> None:
        self._cfg = cfg
        self._base = Path(data_root) / split
        self._backend: ASRBackend = backend if backend is not None else make_backend(cfg)
        self._warmed = False
        self.n_prompt_retries = 0
        if cfg.prompt_guard_ngram and not hasattr(self._backend, "transcribe_batch_unprompted"):
            raise ValueError(f"asr.prompt_guard_ngram: engine {cfg.engine!r} cannot transcribe without the prompt")

    def wav_paths(self, sample: Sample) -> list[Path]:
        """Liệt kê file WAV của các lượt user theo thứ tự lịch sử (`_u1`, `_u2`, ...) rồi tới yêu cầu cuối.

        Dùng đường dẫn `audio` của mẫu khi có; không có thì suy ra theo quy ước đặt tên file của bộ dữ liệu.
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
        """Nhận dạng các lượt user trong lịch sử và lượt cuối của mẫu.

        File được chia thành từng batch `batch_size`, và số văn bản trả về phải khớp số file. Khi `sanitize` bật,
        mọi lượt trừ lượt cuối được lọc theo kiểu lịch sử (`is_history=True`).

        Tham số:
            sample: mẫu cần nhận dạng.

        Trả về:
            `Transcript` với `history` là `None` ở lượt assistant, và `raw` là văn bản trước khi lọc.

        Lỗi:
            ValueError: khi backend trả về số văn bản khác số file.
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
        if self._cfg.prompt_guard_ngram:
            texts = self._guard(sample.id, paths, texts)
        raw = tuple(texts)
        if self._cfg.sanitize:
            texts = [sanitize(t, is_history=i < len(texts) - 1) for i, t in enumerate(texts)]
        # Chia lần lượt văn bản lịch sử cho các lượt user; lượt assistant nhận `None`.
        it = iter(texts[:-1])
        history = tuple(next(it) if t.role == "user" else None for t in sample.history)
        return Transcript(final=texts[-1], history=history, seconds=time.perf_counter() - start, raw=raw)

    def _guard(self, sample_id: str, paths: list[Path], texts: list[str]) -> list[str]:
        """Chép lại không câu mồi các câu hỏng do câu mồi (`asr_guard.needs_retry`); trả văn bản đã thay."""
        cfg = self._cfg
        prompt = str(cfg.generate_kwargs.get("initial_prompt") or "")
        bad = [
            i
            for i, t in enumerate(texts)
            if needs_retry(
                t,
                sanitize(t, is_history=i < len(texts) - 1),
                prompt,
                cfg.prompt_guard_ngram,
                cfg.prompt_guard_max_compression,
            )
        ]
        if not bad:
            return texts
        backend: Any = self._backend  # đã kiểm tra `transcribe_batch_unprompted` ở `__init__`
        out = list(texts)
        size = max(1, cfg.batch_size)
        for j in range(0, len(bad), size):
            idx = bad[j : j + size]
            got = backend.transcribe_batch_unprompted([paths[i] for i in idx])
            if len(got) != len(idx):
                raise ValueError(f"{sample_id}: ASR returned {len(got)} texts for {len(idx)} files")
            for i, t in zip(idx, got, strict=True):
                out[i] = t
        self.n_prompt_retries += len(bad)
        log.info("asr_prompt_retry", sample_id=sample_id, utterances=len(bad), total=len(texts))
        return out

    def warmup(self) -> None:
        """Khởi động backend, chỉ một lần duy nhất."""
        if not self._warmed:
            self._backend.warmup()
            self._warmed = True

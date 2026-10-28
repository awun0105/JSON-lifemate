"""Test lớp bảo vệ câu mồi: `lifemate.backends.asr_guard` và cách `ASRTranscriber` chép lại không câu mồi.

Nội dung chính:
- `PromptedBackend`: backend giả có câu mồi, câu nào bị "hỏng" thì chép lại không câu mồi ra chữ khác.

Lưu ý:
- Ví dụ câu hỏng lấy theo các lỗi thật của whisper-large-v3 ở benchmark ASR lần 1 (chép lại câu mồi, đếm tiếp danh
  sách, câu phụ đề bịa).
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from lifemate.backends.asr import ASRTranscriber
from lifemate.backends.asr_guard import compression_ratio, echoes_prompt, needs_retry
from lifemate.backends.asr_sanitizer import sanitize
from lifemate.core.config import ASRConfig
from lifemate.core.types import Sample

PROMPT = "Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."


def _retry(raw: str, ngram: int = 3) -> bool:
    """Gọi `needs_retry` với chữ sau lọc tính từ `raw`, như `ASRTranscriber` làm cho lượt cuối."""
    return needs_retry(raw, sanitize(raw, is_history=False), PROMPT, ngram, 2.4)


def test_needs_retry_signals() -> None:
    """Bốn dấu hiệu hỏng đều bị bắt; câu bình thường, kể cả có một từ của câu mồi, thì không."""
    assert not _retry("Trời ơi to quá, giảm còn phân nửa lẹ giùm em.")
    assert not _retry("Nhắn Zalo giùm mẹ nha.")  # chỉ trùng 2 từ liên tiếp với câu mồi
    # Người dùng nói thật đúng 3 từ của câu mồi: cụm 3 từ báo nhầm, cụm 4 từ thì không.
    assert _retry("Nhắn Zalo cho mẹ nha.") and not _retry("Nhắn Zalo cho mẹ nha.", ngram=4)
    assert _retry("Chuyển sang kênh VTV3.")  # chép lại câu mồi
    assert not _retry("Chuyển sang kênh HTV7 nha.", ngram=4)  # cụm 4 từ thì không trùng
    assert _retry("Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn")  # bộ lọc cắt
    assert _retry("")  # rỗng
    assert _retry("Mate 2." + ".0" * 80)  # lặp: tỉ lệ nén cao
    assert compression_ratio("") == 0.0 and compression_ratio("abc" * 100) > 2.4
    assert echoes_prompt("ừ chuyển sang kênh khác", PROMPT, 3) and not echoes_prompt("chuyển kênh", PROMPT, 3)


class PromptedBackend:
    """Backend giả: có câu mồi thì lượt cuối chép lại câu mồi; không câu mồi thì ra chữ đúng."""

    def __init__(self) -> None:
        self.unprompted: list[list[str]] = []

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Lượt cuối (`t1.wav`) bị chép lại câu mồi, các lượt khác ra chữ bình thường."""
        return [PROMPT if p.name == "t1.wav" else f"text {p.stem}" for p in wav_paths]

    def transcribe_batch_unprompted(self, wav_paths: Sequence[Path]) -> list[str]:
        """Ghi lại các file được chép lại và trả chữ đúng."""
        self.unprompted.append([p.name for p in wav_paths])
        return [f"fixed {p.stem}" for p in wav_paths]

    def warmup(self) -> None:
        """Không cần làm gì."""


class PlainBackend:
    """Backend giả không chép lại được không câu mồi (như ChunkFormer)."""

    def transcribe_batch(self, wav_paths: Sequence[Path]) -> list[str]:
        """Trả chữ giả cho mỗi file."""
        return ["x" for _ in wav_paths]

    def warmup(self) -> None:
        """Không cần làm gì."""


SAMPLE = Sample(id="t1", split="train", audio="audio/t1/t1.wav", audio_text=None, context={}, history=(), label=None)


def _cfg(ngram: int) -> ASRConfig:
    """Config ASR có câu mồi `PROMPT` và lớp bảo vệ cụm `ngram` từ (0 là tắt)."""
    return ASRConfig(
        backend="asr",
        model="m",
        revision="r",
        engine="faster_whisper",
        generate_kwargs={"initial_prompt": PROMPT},
        sanitize=True,
        prompt_guard_ngram=ngram,
    )


def test_transcriber_retries_only_bad_utterances() -> None:
    """Câu hỏng được chép lại không câu mồi và thay vào kết quả (cả `raw`); bộ đếm tăng; tắt bảo vệ thì giữ nguyên."""
    fake = PromptedBackend()
    tr = ASRTranscriber(_cfg(3), data_root=Path("/data"), split="train", backend=fake)
    t = tr.transcribe(SAMPLE)
    assert t.final == "fixed t1" and t.raw == ("fixed t1",)
    assert fake.unprompted == [["t1.wav"]] and tr.n_prompt_retries == 1
    off = ASRTranscriber(_cfg(0), data_root=Path("/data"), split="train", backend=PromptedBackend())
    assert off.transcribe(SAMPLE).final == PROMPT and off.n_prompt_retries == 0


def test_transcriber_rejects_backend_without_unprompted() -> None:
    """Bật bảo vệ câu mồi với backend không chép lại được không câu mồi thì báo lỗi ngay khi dựng."""
    with pytest.raises(ValueError, match="prompt_guard_ngram"):
        ASRTranscriber(_cfg(3), data_root=Path("/data"), split="train", backend=PlainBackend())

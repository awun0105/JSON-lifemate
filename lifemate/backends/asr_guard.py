"""Lớp bảo vệ câu mồi: phát hiện câu ASR hỏng do câu mồi (`initial_prompt`) để chép lại không câu mồi.

Nội dung chính:
- `compression_ratio`: tỉ lệ nén zlib của văn bản, cao khi văn bản lặp.
- `echoes_prompt`: văn bản có chép lại một cụm `n` từ liên tiếp của câu mồi không.
- `needs_retry`: gộp các dấu hiệu hỏng thành một quyết định.

Cách hoạt động: một câu chép với câu mồi bị coi là hỏng khi có một trong bốn dấu hiệu, không cần đáp án:
1. Rỗng sau khi lọc.
2. Bộ lọc ảo giác có cắt (`cleaned` khác `raw` đã chuẩn hoá): model đang bịa câu kiểu phụ đề.
3. Có cụm `ngram` từ liên tiếp trùng câu mồi: model chép lại câu mồi thay vì nghe audio.
4. Tỉ lệ nén lớn hơn `max_compression`: model lặp vô hạn ("VTV3 VTV4 VTV5…"). Ngưỡng mặc định 2,4 giống ngưỡng
   faster-whisper dùng để phát hiện câu lặp.

Lưu ý:
- Lý do và số đo offline trên kết quả lần 1: `docs/reports/asr_benchmark_en_mixed.md` mục 4.2.
- Người dùng nói thật đúng `ngram` từ có trong câu mồi thì bị báo nhầm; câu đó được chép lại không câu mồi, mất phần
  lợi của câu mồi nhưng không sai thêm.
"""

from __future__ import annotations

import re
import unicodedata
import zlib

from lifemate.backends.asr_sanitizer import normalize

# Một từ là một dãy chữ cái hoặc chữ số liền nhau; dấu câu là ranh giới.
_WORD = re.compile(r"[^\W_]+")


def _word_tokens(text: str) -> list[str]:
    """Tách văn bản thành từ: chữ thường, NFC, bỏ dấu câu."""
    return _WORD.findall(unicodedata.normalize("NFC", text.lower()))


def compression_ratio(text: str) -> float:
    """Tỉ lệ độ dài UTF-8 trên độ dài sau khi nén zlib; 0.0 với chuỗi rỗng (cách tính giống faster-whisper)."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def echoes_prompt(text: str, prompt: str, ngram: int) -> bool:
    """`True` khi `text` chứa một cụm `ngram` từ liên tiếp có trong `prompt` (so sau khi bỏ dấu câu, chữ thường)."""
    prompt_words, text_words = _word_tokens(prompt), _word_tokens(text)
    prompt_ngrams = {tuple(prompt_words[i : i + ngram]) for i in range(len(prompt_words) - ngram + 1)}
    return any(tuple(text_words[i : i + ngram]) in prompt_ngrams for i in range(len(text_words) - ngram + 1))


def needs_retry(raw: str, cleaned: str, prompt: str, ngram: int, max_compression: float) -> bool:
    """Quyết định có chép lại câu này không câu mồi.

    Tham số:
        raw: chữ ASR trước khi lọc.
        cleaned: chữ ASR sau `asr_sanitizer.sanitize`.
        prompt: câu mồi đã dùng.
        ngram: độ dài cụm từ dùng để phát hiện chép lại câu mồi.
        max_compression: ngưỡng tỉ lệ nén; lớn hơn là lặp.

    Trả về:
        `True` khi câu có một trong bốn dấu hiệu hỏng ở docstring module.
    """
    return (
        not cleaned.strip()
        or cleaned != normalize(raw)
        or echoes_prompt(raw, prompt, ngram)
        or compression_ratio(raw) > max_compression
    )

"""Câu mồi theo mẫu cho whisper-large-v3 (benchmark ASR lần 3, thử nghiệm E).

Nội dung chính:
- `PROMPT_VARIANTS`: các biến thể câu mồi.
- `build_sample_prompts`: một câu mồi cho mỗi câu người dùng của mẫu, đúng thứ tự file âm thanh.

Cách hoạt động:
- E0: câu mồi cố định (`en_full_v1`) cho mọi câu.
- E1: câu mồi cố định, rồi câu trả lời (`text`) của lượt trợ lý **liền trước** câu đó.
- E2: câu mồi cố định, rồi một câu nêu tên tiếng Việt (`name_vi`) của các app đã cài, rồi lời trợ lý liền trước.
- Câu không có lượt trợ lý liền trước (lượt đầu của hội thoại) chỉ nhận phần cố định, cộng tên app ở E2.
- Phần riêng theo mẫu đặt **cuối**, vì Whisper chỉ giữ 223 token cuối của câu mồi và phần cuối gần âm thanh nhất.

Lưu ý:
- Hàm chỉ đọc `sample.context` (`installed_apps`), lượt trợ lý trong `sample.history` và catalog; không đọc lượt người
  dùng (lúc thi lượt đó chỉ có âm thanh) và không đọc file môi trường của BTC.
- Thứ tự câu mồi khớp `ASRTranscriber.wav_paths`: các lượt người dùng trong lịch sử trước, lượt cuối sau.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lifemate.core.types import Sample

PROMPT_VARIANTS = ("E0", "E1", "E2")


def _app_sentence(sample: Sample, catalog_apps: Mapping[str, Mapping[str, Any]]) -> str:
    """Câu nêu tên tiếng Việt của các app đã cài, theo thứ tự trong `installed_apps`; rỗng khi không có app nào."""
    installed = sample.context.get("installed_apps") or {}
    names = [str(catalog_apps.get(str(app), {}).get("name_vi") or str(app).replace("_", " ")) for app in installed]
    return f"Ứng dụng đã cài: {', '.join(names)}." if names else ""


def build_sample_prompts(
    sample: Sample, catalog_apps: Mapping[str, Mapping[str, Any]], fixed_prompt: str, variant: str
) -> list[str]:
    """Một câu mồi cho mỗi câu người dùng của `sample`, theo thứ tự file âm thanh (lịch sử trước, lượt cuối sau).

    Tham số:
        sample: mẫu cần chép.
        catalog_apps: mã app -> mục catalog của app đó.
        fixed_prompt: câu mồi cố định (`en_full_v1`).
        variant: một tên trong `PROMPT_VARIANTS`.

    Trả về:
        Danh sách câu mồi, độ dài bằng số câu người dùng của mẫu cộng 1 (lượt cuối).

    Lỗi:
        ValueError: khi `variant` không có trong `PROMPT_VARIANTS`.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    fixed = fixed_prompt.strip()
    apps = _app_sentence(sample, catalog_apps) if variant == "E2" else ""

    def prompt_after(previous: Any) -> str:
        """Câu mồi của câu người dùng đứng ngay sau lượt `previous` (hoặc `None` khi là lượt đầu)."""
        parts = [fixed]
        if apps:
            parts.append(apps)
        if variant != "E0" and previous is not None and previous.role == "assistant" and (previous.text or "").strip():
            parts.append(str(previous.text).strip())
        return " ".join(parts)

    prompts: list[str] = []
    previous = None
    for turn in sample.history:
        if turn.role == "user":
            prompts.append(prompt_after(previous))
        previous = turn
    prompts.append(prompt_after(previous))
    return prompts

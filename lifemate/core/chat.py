"""Tiện ích chung cho danh sách tin nhắn chat, dùng ở agent, backend LLM và bộ đếm token.

Nội dung chính:
- `split_tools`: tách trường `tools` khỏi tin `system` đầu tiên.

Lưu ý:
- Ở định dạng ChatML (`lifemate.agent.chatml`), danh sách tool được giữ trong trường `tools` của tin `system`, đúng như
  dữ liệu huấn luyện của repo lead. Chat template của Qwen (và vLLM `LLM.chat`) lại nhận danh sách này qua tham số
  `tools` riêng, nên cần tách ra trước khi dựng prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def split_tools(messages: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Trả (tin nhắn không còn trường `tools`, danh sách tool); danh sách là `None` khi tin đầu không có `tools`."""
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system" and "tools" in out[0]:
        return out, out[0].pop("tools")
    return out, None

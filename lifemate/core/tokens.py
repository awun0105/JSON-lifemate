"""Đếm token của danh sách tin nhắn theo chat template của model (SPEC 4.8).

Nội dung chính:
- `HFTokenCounter`: đếm chính xác bằng tokenizer Hugging Face, tính cả token mà chat template chèn thêm.
- `SimpleTokenCounter`: ước lượng thô theo khoảng trắng, dùng với `FakeLLM` hoặc khi không có `transformers`.

Cách hoạt động:
- Cả hai lớp thoả Protocol `TokenCounter` trong `lifemate.core.types`, nên agent đổi bộ đếm mà không phải sửa mã.
- Tokenizer chạy trên CPU, không chiếm GPU của LLM.

Lưu ý:
- `transformers` chỉ được import trong `HFTokenCounter.__init__`. Nhờ vậy máy không cài extra `tokenizer` vẫn
  import được module này.
"""

from __future__ import annotations

import json
from typing import Any

from lifemate.core.chat import split_tools

# Nội dung mẫu ngắn, dùng để đo số token mà template chèn thêm quanh mỗi tin nhắn.
_PROBE = "xin chào"


class HFTokenCounter:
    """Đếm token bằng `transformers.AutoTokenizer.apply_chat_template`, chạy trên CPU.

    Dùng khi cần số token khớp với prompt thật mà LLM nhận, ví dụ để kiểm tra giới hạn context.

    Tham số:
        model: tên model trên Hugging Face hoặc đường dẫn tới thư mục model.
        revision: commit hoặc branch của model; `None` là bản mặc định.
        local_files_only: `True` thì chỉ đọc cache trên máy, không tải qua mạng (dùng khi chạy offline).
    """

    def __init__(self, model: str, revision: str | None = None, *, local_files_only: bool = False) -> None:
        # `transformers` thuộc extra `tokenizer`, nên chỉ import ở đây.
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        # Số token template chèn thêm cho từng role; mỗi role chỉ tính một lần rồi lưu lại.
        self._overhead: dict[str, int] = {}

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Đếm số token prompt của `messages` sau khi dựng bằng chat template.

        Kết quả tính cả phần mở đầu lượt trả lời của assistant (`add_generation_prompt=True`), đúng như prompt gửi
        cho LLM.

        Tham số:
            messages: danh sách tin nhắn dạng `{"role": ..., "content": ...}`.

        Trả về:
            Số token của prompt.
        """
        # Định dạng ChatML (`lifemate.agent.chatml`) để danh sách tool trong trường `tools` của tin `system`; chat
        # template nhận danh sách này qua tham số `tools`.
        msgs, tools = split_tools(messages)
        kwargs: dict[str, Any] = {"tools": tools} if tools is not None else {}
        ids: Any = self._tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kwargs)
        # Tuỳ phiên bản `transformers`, kết quả là list id hoặc `BatchEncoding`/dict có khoá `input_ids`.
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return len(ids)

    def count_text(self, text: str) -> int:
        """Đếm số token của văn bản thô, không tính token đặc biệt."""
        return len(self._tok.encode(text, add_special_tokens=False))

    def template_overhead(self, role: str) -> int:
        """Trả về số token mà template chèn thêm khi nối một tin nhắn có role `role`.

        Cách tính: dựng một hội thoại cơ sở, nối thêm một tin nhắn `role` có nội dung `_PROBE`, lấy số token chênh
        lệch rồi trừ đi số token của chính `_PROBE`. Kết quả được lưu lại, mỗi role chỉ tính một lần.

        Tham số:
            role: role của tin nhắn, ví dụ `"user"` hoặc `"tool"`.

        Trả về:
            Số token template thêm vào cho một tin nhắn của role đó.
        """
        if role not in self._overhead:
            base = [{"role": "system", "content": _PROBE}, {"role": "user", "content": _PROBE}]
            # Nhiều chat template bắt buộc user và assistant xen kẽ nhau. Vì vậy, trước khi nối một tin nhắn user mới,
            # cần thêm một lượt assistant.
            if role == "user":
                base.append({"role": "assistant", "content": _PROBE})
            grown = [*base, {"role": role, "content": _PROBE}]
            self._overhead[role] = self.count(grown) - self.count(base) - self.count_text(_PROBE)
        return self._overhead[role]


class SimpleTokenCounter:
    """Ước lượng số token bằng cách đếm từ theo khoảng trắng, cộng 4 token cho mỗi tin nhắn.

    Chỉ là ước lượng, dùng với `FakeLLM` hoặc khi không có `transformers`. Số 4 xấp xỉ phần token template
    (role, ký hiệu mở/đóng) quanh mỗi tin nhắn.
    """

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Trả về số token ước lượng của `messages`."""
        tools = messages[0].get("tools") if messages else None
        extra = len(json.dumps(tools, ensure_ascii=False).split()) if tools else 0
        calls = sum(
            len(json.dumps(m["tool_calls"], ensure_ascii=False).split()) for m in messages if m.get("tool_calls")
        )
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages) + extra + calls

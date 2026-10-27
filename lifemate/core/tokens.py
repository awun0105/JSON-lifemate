"""Đếm token theo chat template của mô hình (SPEC 4.8, HFTokenCounter).

Nội dung file:
- `HFTokenCounter`: đếm chính xác bằng tokenizer của Hugging Face, kể cả token do chat template thêm vào.
- `SimpleTokenCounter`: ước lượng thô theo khoảng trắng, dùng với FakeLLM hoặc khi không có transformers.

Cách xây dựng: cả hai lớp thỏa Protocol `TokenCounter` (lifemate.core.types), nên agent đổi bộ đếm mà không
phải sửa mã. `transformers` chỉ được import trong `HFTokenCounter.__init__`, để môi trường không cài extra
`tokenizer` vẫn import được module này. Tokenizer chạy trên CPU, không chiếm GPU của LLM.
"""

from __future__ import annotations

from typing import Any

# Nội dung mẫu ngắn dùng để đo phần token template thêm vào mỗi tin nhắn.
_PROBE = "xin chào"


class HFTokenCounter:
    """TokenCounter dựa trên `transformers.AutoTokenizer.apply_chat_template`; chạy trên CPU.

    Tham số khởi tạo:
        model: tên hoặc đường dẫn mô hình trên Hugging Face.
        revision: commit hoặc nhánh của mô hình; None là mặc định.
        local_files_only: True thì chỉ đọc cache cục bộ, không tải qua mạng (dùng khi chạy offline).
    """

    def __init__(self, model: str, revision: str | None = None, *, local_files_only: bool = False) -> None:
        # transformers đến từ extra `tokenizer`.
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        # Cache phần token template thêm vào theo từng role, tính một lần.
        self._overhead: dict[str, int] = {}

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Số token prompt của `messages` khi dựng bằng chat template, cộng cả phần mở đầu lượt sinh."""
        ids: Any = self._tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        # Tùy phiên bản transformers, kết quả có thể là list id hoặc BatchEncoding/dict chứa `input_ids`.
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return len(ids)

    def count_text(self, text: str) -> int:
        """Số token của văn bản thô, không tính token đặc biệt."""
        return len(self._tok.encode(text, add_special_tokens=False))

    def template_overhead(self, role: str) -> int:
        """Số token template thêm vào khi nối một tin nhắn `role` (tính một lần bằng phép trừ).

        Cách tính: dựng một hội thoại cơ sở, nối thêm một tin nhắn `role` có nội dung `_PROBE`, rồi lấy hiệu số
        token trừ đi số token của chính `_PROBE`.
        """
        if role not in self._overhead:
            base = [{"role": "system", "content": _PROBE}, {"role": "user", "content": _PROBE}]
            # Thêm lượt assistant để tin nhắn user mới nối vào giữ đúng thứ tự xen kẽ user/assistant
            # mà nhiều chat template bắt buộc.
            if role == "user":
                base.append({"role": "assistant", "content": _PROBE})
            grown = [*base, {"role": role, "content": _PROBE}]
            self._overhead[role] = self.count(grown) - self.count(base) - self.count_text(_PROBE)
        return self._overhead[role]


class SimpleTokenCounter:
    """Đếm token theo khoảng trắng trong nội dung mỗi tin nhắn, cộng 4 cho mỗi tin nhắn.

    Dùng với FakeLLM hoặc khi không có transformers; chỉ là ước lượng.
    """

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Số token prompt ước lượng của `messages`."""
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages)

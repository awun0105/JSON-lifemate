"""T15: số token đếm tăng dần trong `Conversation` phải bằng số đếm lại toàn bộ (SPEC 4.6, 4.8).

Cách hoạt động:
- Dùng `HFTokenCounter` với tokenizer Qwen2.5-0.5B có sẵn trong cache HF trên máy.

Lưu ý:
- Test mang marker `tokenizer`: thiếu transformers hoặc tokenizer chưa có trong cache thì skip.
- Không cần dữ liệu cuộc thi.
"""

from __future__ import annotations

import pytest

from lifemate.agent.prompt import Conversation
from lifemate.agent.protocol import call_text

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.tokenizer
def test_token_count_incremental() -> None:
    """Phép đếm tăng dần của `Conversation` khớp với đếm lại toàn bộ qua chat template sau mỗi lượt."""
    pytest.importorskip("transformers", reason="transformers missing (install the `tokenizer` extra)")
    from lifemate.core.tokens import HFTokenCounter

    try:
        counter = HFTokenCounter(MODEL, None, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"tokenizer {MODEL} not in the local HF cache: {exc}")
    conv = Conversation(
        [{"role": "system", "content": "Quy tắc: chỉ trả JSON."}, {"role": "user", "content": "Tắt đèn phòng khách"}],
        counter,
    )
    # 20 vòng gọi tool và nhận kết quả; sau mỗi vòng, hai cách đếm phải bằng nhau.
    for i in range(20):
        text = call_text("smart_home.light_set", {"device_id": f"dev_{i}", "power": "off"})
        conv.append_assistant(text, completion_tokens=counter.count_text(text))
        conv.append_user(f'KẾT QUẢ: {{"ok":true,"step":{i},"ghi_chú":"đã tắt đèn số {i}"}}')
        assert conv.prompt_tokens() == counter.count(list(conv.messages)), i

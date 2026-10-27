"""T15: incremental token count equals a full recount (SPEC 4.6, 4.8)."""

from __future__ import annotations

import pytest

from lifemate.prompt import Conversation
from lifemate.protocol import call_text

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.tokenizer
def test_token_count_incremental() -> None:
    """Protects: Conversation's incremental count. Catches: drift vs a full chat-template recount."""
    pytest.importorskip("transformers", reason="transformers missing (install the `tokenizer` extra)")
    from lifemate.tokens import HFTokenCounter

    try:
        counter = HFTokenCounter(MODEL, None, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"tokenizer {MODEL} not in the local HF cache: {exc}")
    conv = Conversation(
        [{"role": "system", "content": "Quy tắc: chỉ trả JSON."}, {"role": "user", "content": "Tắt đèn phòng khách"}],
        counter,
    )
    for i in range(20):
        text = call_text("smart_home.light_set", {"device_id": f"dev_{i}", "power": "off"})
        conv.append_assistant(text, completion_tokens=counter.count_text(text))
        conv.append_user(f'KẾT QUẢ: {{"ok":true,"step":{i},"ghi_chú":"đã tắt đèn số {i}"}}')
        assert conv.prompt_tokens() == counter.count(list(conv.messages)), i

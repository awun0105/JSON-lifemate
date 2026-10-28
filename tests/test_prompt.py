"""T6, T7, T14, T24: test ghép prompt, giấu app held-out, hội thoại chỉ nối thêm và các mức rút gọn (SPEC 4.6).

Lưu ý:
- Dùng catalog tổng hợp `CATALOG`, template ngắn `TEMPLATE` và `WhitespaceTokenCounter` từ `tests/fakes.py` thay
  cho tokenizer thật.
- Không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

from typing import Any

import pytest

from lifemate.agent.prompt import (
    Conversation,
    HeldOutLeak,
    PromptBuilder,
    render_calendar,
    render_context,
    render_history,
    truncate_result,
)
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.core.types import Sample, Transcript, Turn
from tests.fakes import WhitespaceTokenCounter

TEMPLATE = 'Quy tắc: trả {"action":"done"} khi xong.\n\n# Các app và tool đã cài\n\n{tool_docs}\n'
SENTINEL = "ZZ_LABEL_SENTINEL_ZZ"


def _tool(name: str, kind: str, props: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tạo mô tả tool trong catalog với các thuộc tính `props`."""
    return {
        "name": name,
        "kind": kind,
        "description": f"mô tả {name}",
        "returns": "",
        "parameters": {"type": "object", "properties": props or {}, "required": []},
        "app": name.split(".", maxsplit=1)[0],
    }


LONG_DESC = " ".join(["chi tiết"] * 40)
CATALOG: dict[str, Any] = {
    "apps": [
        {
            "id": "alarm",
            "name_vi": "Báo thức",
            "description": "đặt báo thức",
            "tools": [
                _tool("alarm.list", "read"),
                _tool("alarm.delete", "write", {"alarm_id": {"type": "string", "description": LONG_DESC}}),
            ],
        },
        {
            "id": "medication",
            "name_vi": "Thuốc",
            "description": "nhắc uống thuốc",
            "tools": [
                _tool("medication.list", "read"),
            ],
        },
        {"id": "sys", "name_vi": "Hệ thống", "description": "hệ thống", "tools": [_tool(ASK_TOOL, "meta")]},
    ]
}
INSTALLED = {"alarm": {"tools": ["alarm.list", "alarm.delete"]}, "medication": {"tools": ["medication.list"]}}
CONTEXT = {
    "current_time": "2026-11-13T08:00:00+07:00",
    "timezone": "Asia/Ho_Chi_Minh",
    "location": {"lat": 10.5, "lng": 106.7},
    "user_profile": {"name": "Linh", "city": "Huế"},
    "installed_apps": INSTALLED,
}


def _user(text: str) -> Turn:
    """Tạo lượt user có văn bản gốc `text`."""
    return Turn("user", "a.wav", text, None, (), ())


def _asst(text: str, name: str, result: dict[str, Any]) -> Turn:
    """Tạo lượt assistant có một lời gọi tool `name` và kết quả `result`."""
    return Turn("assistant", None, None, text, ({"name": name, "arguments": {"k": 1}},), (result,))


def _sample(sid: str, history: tuple[Turn, ...] = (), label: dict[str, Any] | None = None) -> Sample:
    """Tạo `Sample` có câu cuối cố định "yêu cầu cuối"."""
    return Sample(sid, "train", "a.wav", "yêu cầu cuối", CONTEXT, history, label)


def _transcript(sample: Sample) -> Transcript:
    """Tạo transcript gold: dùng nguyên văn bản gốc của câu cuối và các lượt user."""
    return Transcript(
        final=sample.audio_text or "",
        history=tuple(t.audio_text if t.role == "user" else None for t in sample.history),
        seconds=0.0,
    )


def _builder(max_model_len: int = 100_000, held_out: frozenset[str] = frozenset()) -> PromptBuilder:
    """Tạo `PromptBuilder` với giới hạn độ dài và tập app held-out cho trước."""
    return PromptBuilder(
        TEMPLATE,
        CatalogIndex(CATALOG),
        WhitespaceTokenCounter(),
        max_model_len=max_model_len,
        max_tokens=50,
        max_result_chars=400,
        held_out_apps=held_out,
    )


def _all_text(conv: Conversation) -> str:
    """Nối nội dung mọi tin nhắn để tìm chuỗi."""
    return "\n".join(m["content"] for m in conv.messages)


def test_render_blocks() -> None:
    """Khối ngữ cảnh không lộ `installed_apps`; khối lịch sử dùng transcript ASR và cắt kết quả tool."""
    ctx = render_context(CONTEXT)
    assert ctx.startswith("# Ngữ cảnh\n")
    assert "location: 10.5, 106.7" in ctx and "user_profile.city: Huế" in ctx
    assert "installed_apps" not in ctx and "alarm" not in ctx
    hist = (_user("có báo thức nào"), _asst("Có một.", "alarm.list", {"ok": True, "n": "x" * 50}))
    text = render_history(hist, ("có báo thức nào (asr)", None), 20)
    assert "Người dùng: có báo thức nào (asr)" in text
    assert "Trợ lý: Có một." in text
    assert '  [đã gọi] alarm.list({"k":1}) → ' + truncate_result({"ok": True, "n": "x" * 50}, 20) in text
    assert render_history((), (), 20) == ""


def test_prompt_no_label() -> None:
    """T6: nhãn không bao giờ lọt vào các tin nhắn."""
    label = {"tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": SENTINEL}}], "note": SENTINEL}
    sample = _sample("s1", (_user("hỏi"), _asst("đáp", "alarm.list", {"ok": True})), label)
    conv, level = _builder().start(sample, _transcript(sample))
    assert level == 0
    assert SENTINEL not in _all_text(conv)
    assert conv.messages[1]["content"].endswith("# Yêu cầu hiện tại\nyêu cầu cuối")


def test_held_out_stripped() -> None:
    """T7: train mode bỏ mô tả của app held-out; lịch sử có gọi app held-out thì báo `HeldOutLeak`."""
    builder = _builder(held_out=frozenset({"medication"}))
    sample = _sample("s2")
    conv, _ = builder.start(sample, _transcript(sample), train_mode=True)
    assert "medication" not in _all_text(conv)
    assert "medication.list" not in builder.allowed_tools(sample, train_mode=True)
    conv_eval, _ = builder.start(sample, _transcript(sample))
    assert "medication.list" in _all_text(conv_eval)

    leaky = _sample("s3", (_user("thuốc?"), _asst("đây", "medication.list", {"ok": True})))
    with pytest.raises(HeldOutLeak) as err:
        builder.start(leaky, _transcript(leaky), train_mode=True)
    assert err.value.sample_id == "s3" and err.value.app == "medication"
    builder.start(leaky, _transcript(leaky))  # chế độ eval: không báo lỗi


def test_conversation_append_only() -> None:
    """T14: tin nhắn cũ không bao giờ đổi, và phần quy tắc đầu prompt giống hệt từng byte giữa các mẫu."""
    builder = _builder()
    a = _sample("a")
    b = _sample("b", (_user("x"), _asst("y", "alarm.list", {"ok": True})))
    conv_a, _ = builder.start(a, _transcript(a))
    conv_b, _ = builder.start(b, _transcript(b))
    for conv in (conv_a, conv_b):
        assert conv.messages[0]["content"].startswith(builder.rules_prefix)
    assert builder.rules_prefix == TEMPLATE.split("{tool_docs}", maxsplit=1)[0]

    counter = WhitespaceTokenCounter()
    snapshots: list[tuple[dict[str, Any], ...]] = []
    for i in range(5):
        snapshots.append(tuple(dict(m) for m in conv_a.messages))
        before = conv_a.messages
        conv_a.append_assistant(f'{{"action":"call","name":"alarm.list","arguments":{{}}}} {i}', completion_tokens=2)
        conv_a.append_user(f'KẾT QUẢ: {{"ok":true,"i":{i}}}')
        assert isinstance(conv_a.messages, tuple)
        assert conv_a.messages[: len(before)] == before
        # Tin nhắn cũ phải là cùng đối tượng, không phải bản sao.
        assert all(x is y for x, y in zip(before, conv_a.messages, strict=False))
        assert conv_a.prompt_tokens() == counter.count(list(conv_a.messages))
    for snap in snapshots:
        assert conv_a.messages[: len(snap)] == snap


def test_reconcile() -> None:
    """`reconcile` thay số token ước lượng bằng số thật từ backend và trả `True` khi có chênh lệch."""
    conv = Conversation([{"role": "user", "content": "a b c"}], WhitespaceTokenCounter())
    assert conv.prompt_tokens() == 7
    assert conv.reconcile(7) is False
    assert conv.reconcile(9) is True
    conv.append_user("d")
    assert conv.prompt_tokens() == 14


def _long_sample() -> Sample:
    """Tạo mẫu có 4 cặp lượt với kết quả tool dài, để buộc phải rút gọn prompt."""
    turns: list[Turn] = []
    for i in range(4):
        turns.append(_user(f"lượt {i} " + "nói " * 5))
        turns.append(_asst(f"trả lời {i}", "alarm.list", {"ok": True, "data": " ".join(["kq"] * 150)}))
    return _sample("long", tuple(turns))


def test_prompt_shrink_levels() -> None:
    """T24: áp lần lượt mức 1, 2, 3, dừng ở mức đầu tiên vừa giới hạn; luôn giữ phần quy tắc và câu cuối."""
    sample = _long_sample()
    tr = _transcript(sample)
    conv0, got = _builder().start(sample, tr)
    assert got == 0
    full = conv0.prompt_tokens()

    def run(max_len: int) -> tuple[Conversation, int]:
        """Dựng prompt với giới hạn `max_len` và kiểm tra phần quy tắc, câu cuối vẫn còn."""
        conv, level = _builder(max_model_len=max_len).start(sample, tr)
        assert conv.messages[0]["content"].startswith(_builder().rules_prefix)
        assert conv.messages[1]["content"].endswith("# Yêu cầu hiện tại\nyêu cầu cuối")
        return conv, level

    # 50 là `max_tokens` dành cho đầu ra: vừa đủ chỗ thì không rút gọn, thiếu 1 token thì sang mức 1.
    assert run(full + 50)[1] == 0
    conv1, lvl = run(full + 49)
    assert lvl == 1 and LONG_DESC in conv1.messages[0]["content"]
    t1 = conv1.prompt_tokens()
    conv2, lvl = run(t1 + 49)
    assert lvl == 2 and LONG_DESC not in conv2.messages[0]["content"] and "lượt 0" in conv2.messages[1]["content"]
    t2 = conv2.prompt_tokens()
    conv3, lvl = run(t2 + 49)
    assert lvl == 3
    user = conv3.messages[1]["content"]
    assert "lượt 0" not in user and "lượt 3" in user and "trả lời 3" in user
    assert conv3.prompt_tokens() + 50 <= t2 + 49

    # Giới hạn quá nhỏ: dừng ở mức 3 dù vẫn vượt, chỉ giữ các lượt gần nhất.
    conv_x, lvl = run(10)
    assert lvl == 3 and conv_x.prompt_tokens() + 50 > 10
    user = conv_x.messages[1]["content"]
    assert "lượt 3" in user and "trả lời 3" in user and "lượt 2" not in user


def test_render_calendar_weekdays_and_weeks() -> None:
    """Lịch ghi đúng tên thứ, "mai" và ranh giới tuần từ thứ Hai (các lỗi ngày tháng từng gặp khi chạy smoke)."""
    lines = render_calendar("2026-11-28T07:10:00+07:00")  # thứ Bảy
    assert lines[0] == "Lịch (bây giờ là thứ Bảy 2026-11-28 07:10):"
    assert "  2026-11-27 thứ Sáu (tuần này, hôm qua)" in lines
    assert "  2026-11-29 Chủ nhật (tuần này, mai)" in lines
    assert "  2026-11-30 thứ Hai (tuần sau, mốt / ngày kia)" in lines
    assert "  2026-12-11 thứ Sáu (2 tuần nữa)" in lines
    assert len(lines) == 1 + 15
    assert render_calendar("not a time") == []

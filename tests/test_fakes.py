"""Kiểm tra các đối tượng giả dùng chung trong test: `FakeEnv`, `FakeLLM`, `FakeClock`, `WhitespaceTokenCounter`.

Các fake lấy từ `tests/fakes.py` và `lifemate.backends.llm`. Không cần dữ liệu cuộc thi hay tokenizer.
"""

import json

from lifemate.backends.llm import FakeLLM
from lifemate.core.types import Sample
from tests.fakes import FakeClock, FakeEnv, WhitespaceTokenCounter

CALLS = [{"name": "a.x", "arguments": {"k": 1}}, {"name": "a.y", "arguments": {}}]


def _sample(sid: str = "s") -> Sample:
    """Tạo `Sample` có nhãn là hai tool call trong `CALLS`."""
    return Sample(sid, "train", "a.wav", "t", {}, (), {"tool_calls": CALLS})


def test_fake_env_rejects_and_logs() -> None:
    """`FakeEnv` từ chối tool lạ và thiếu tham số, ghi log mọi lần gọi, đổi hash trạng thái sau khi ghi."""
    env = FakeEnv({"a.x"}, {"a.x": ["k"]})
    assert env.validate("a.x", {"k": 1}) == {"ok": True, "error": None}
    assert not env.validate("b.z", {})["ok"]
    r = env.call("a.x", {})
    assert r["ok"] is False and "k" in r["error"]
    assert env.call("a.x", {"k": 1}) == {"ok": True, "done": True}
    calls = env.get_tool_calls()
    assert len(calls) == 2
    # Xoá bản sao không ảnh hưởng log bên trong.
    calls.clear()
    assert len(env.get_tool_calls()) == 2
    h = env.get_state_hash()
    env.call("a.x", {"k": 2})
    assert env.get_state_hash() != h


def test_fake_llm_replays_then_done() -> None:
    """`FakeLLM` phát lại lần lượt các tool call trong nhãn, sau đó luôn trả done; sample lạ trả done ngay."""
    llm = FakeLLM([_sample()])
    msgs = [{"role": "user", "content": "xin chào bạn"}]
    outs = [llm.generate(msgs, {}, sample_id="s") for _ in range(4)]
    assert [json.loads(o.text) for o in outs] == [
        {"action": "call", **CALLS[0]},
        {"action": "call", **CALLS[1]},
        {"action": "done"},
        {"action": "done"},
    ]
    assert outs[0].prompt_tokens == 3
    assert llm.generate(msgs, {}, sample_id="other").text == '{"action":"done"}'


def test_broken_outputs_first() -> None:
    """Đầu ra hỏng được trả trước, sau đó mới phát lại nhãn."""
    llm = FakeLLM([_sample()], broken_outputs={"s": ["not json"]})
    assert llm.generate([], {}, sample_id="s").text == "not json"
    assert json.loads(llm.generate([], {}, sample_id="s").text)["name"] == "a.x"


def test_clock_advances() -> None:
    """Mỗi lần generate đẩy đồng hồ giả thêm `delay_s` giây."""
    clock = FakeClock()
    llm = FakeLLM([_sample()], delay_s=2.5, clock=clock)
    g = llm.generate([], {}, sample_id="s")
    llm.generate([], {}, sample_id="s")
    assert clock() == 5.0 and g.seconds == 2.5


def test_whitespace_counter() -> None:
    """Bộ đếm token theo khoảng trắng cộng thêm phần phụ cố định cho mỗi message."""
    assert WhitespaceTokenCounter().count([{"content": "a b"}, {"content": ""}]) == 10

import json

from lifemate.llm import FakeLLM
from lifemate.types import Sample
from tests.fakes import FakeClock, FakeEnv, WhitespaceTokenCounter

CALLS = [{"name": "a.x", "arguments": {"k": 1}}, {"name": "a.y", "arguments": {}}]


def _sample(sid: str = "s") -> Sample:
    return Sample(sid, "train", "a.wav", "t", {}, (), {"tool_calls": CALLS})


def test_fake_env_rejects_and_logs() -> None:
    env = FakeEnv({"a.x"}, {"a.x": ["k"]})
    assert env.validate("a.x", {"k": 1}) == {"ok": True, "error": None}
    assert not env.validate("b.z", {})["ok"]
    r = env.call("a.x", {})
    assert r["ok"] is False and "k" in r["error"]
    assert env.call("a.x", {"k": 1}) == {"ok": True, "done": True}
    calls = env.get_tool_calls()
    assert len(calls) == 2
    calls.clear()
    assert len(env.get_tool_calls()) == 2
    h = env.get_state_hash()
    env.call("a.x", {"k": 2})
    assert env.get_state_hash() != h


def test_fake_llm_replays_then_done() -> None:
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
    llm = FakeLLM([_sample()], broken_outputs={"s": ["not json"]})
    assert llm.generate([], {}, sample_id="s").text == "not json"
    assert json.loads(llm.generate([], {}, sample_id="s").text)["name"] == "a.x"


def test_clock_advances() -> None:
    clock = FakeClock()
    llm = FakeLLM([_sample()], delay_s=2.5, clock=clock)
    g = llm.generate([], {}, sample_id="s")
    llm.generate([], {}, sample_id="s")
    assert clock() == 5.0 and g.seconds == 2.5


def test_whitespace_counter() -> None:
    assert WhitespaceTokenCounter().count([{"content": "a b"}, {"content": ""}]) == 10

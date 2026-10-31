"""Test định dạng ChatML (`lifemate.agent.chatml`, `lifemate.agent.chatml_agent`).

Nội dung chính:
- Tin nhắn dựng được giống từng phần tử với `build_messages` của repo lead (`tests/lead_reference.py`), trên mọi mẫu
  train (marker `data`), cả chữ đáp án lẫn chữ ASR, cả bản đầy đủ lẫn bản rút gọn schema.
- `parse_reply` đọc đúng khối `<tool_call>`, câu kết, và báo lỗi khối hỏng.
- `ChatMLAgent` với LLM giả và môi trường giả: lời gọi đúng tới env, lời gọi bị guard chặn thành tin `tool`
  `{"ok": false, ...}` mà không tới env, hỏi lại thì dừng, câu kết thì dừng.
- Chat template thật của Qwen (marker `tokenizer`) in lời gọi thành khối mà `parse_reply` đọc lại được.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from lifemate.agent.chatml import ChatMLPrompt, build_messages, parse_reply, tool_call_constraint
from lifemate.agent.chatml_agent import ChatMLAgent
from lifemate.agent.guards import build_guard_chain
from lifemate.agent.tooldocs import CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.backends.llm import FakeLLM, is_structural_tag
from lifemate.core.types import Action, ErrorCode, ParseError, Sample, StopReason, Transcript, Turn
from lifemate.io.data import load_split
from tests import lead_reference as lead
from tests.fakes import FakeClock, WhitespaceTokenCounter
from tests.test_budget import AGENT_CFG, LLM_CFG
from tests.test_guards import ASK_TOOL, INDEX, INSTALLED, make_env

LLM = dataclasses.replace(LLM_CFG, max_model_len=100_000)
CFG = dataclasses.replace(AGENT_CFG, format="chatml")


def _lead_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Catalog dạng `catalog_indexed.json` của lead (app theo id, tool theo tên) dựng từ catalog của split."""
    return {
        "apps": {a["id"]: a for a in catalog["apps"]},
        "tools": {t["name"]: t for a in catalog["apps"] for t in a["tools"]},
    }


@pytest.mark.data
def test_messages_match_lead_on_every_train_sample(data_root: Path) -> None:
    """Mọi mẫu train: tin nhắn giống `build_messages` của lead, với chữ đáp án và chữ ASR, đầy đủ và rút gọn."""
    raw = [json.loads(x) for x in (data_root / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines() if x]
    catalog = json.loads((data_root / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    index, lead_cat = CatalogIndex(catalog), _lead_catalog(catalog)
    samples = {s.id: s for s in load_split(data_root, "train")}
    for row in raw:
        sample = samples[row["id"]]
        gold = GoldTranscriber().transcribe(sample)
        # Chữ "ASR" giả: thêm khoảng trắng thừa để kiểm tra luôn bước chuẩn hoá.
        asr_hist = tuple(
            f"  asr {i}  {t.audio_text}" if t.role == "user" else None for i, t in enumerate(sample.history)
        )
        asr = Transcript(final=f" asr  {row['audio_text']} ", history=asr_hist, seconds=0.0)
        by_audio = {t["audio"]: asr_hist[i] for i, t in enumerate(row["history"]) if t["role"] == "user"}
        for compact in (False, True):
            ours = build_messages(sample, index, gold, compact=compact)
            assert ours == lead.build_messages(row, lead_cat, compact_schema=compact), row["id"]
            ours = build_messages(sample, index, asr, compact=compact)
            want = lead.build_messages(row, lead_cat, asr.final, by_audio, compact_schema=compact)
            assert ours == want, row["id"]


def _sample(history: tuple[Any, ...] = ()) -> Sample:
    """Mẫu tối thiểu với các app của catalog giả."""
    return Sample(
        id="s1",
        split="train",
        audio="a.wav",
        audio_text="xoá báo thức a1",
        context={"current_time": "2026-11-10T08:00:00+07:00", "installed_apps": INSTALLED},
        history=history,
        label={"tool_calls": []},
    )


def test_parse_reply() -> None:
    """Một hoặc nhiều khối `<tool_call>`, tham số dạng chuỗi JSON, câu kết, và các khối hỏng."""
    one = parse_reply('<tool_call>\n{"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}\n</tool_call>')
    assert one == [Action(kind="call", name="alarm.delete", arguments={"alarm_id": "a1"})]
    two = parse_reply(
        '<tool_call>\n{"name": "alarm.list", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "notes.search", "arguments": "{\\"q\\": \\"x\\"}"}\n</tool_call>'
    )
    assert isinstance(two, list) and [a.name for a in two] == ["alarm.list", "notes.search"]
    assert two[1].arguments == {"q": "x"}
    assert parse_reply("Tôi đã thực hiện xong toàn bộ các thao tác bạn yêu cầu.") is None
    assert isinstance(parse_reply("<tool_call>\n{not json}\n</tool_call>"), ParseError)
    assert isinstance(parse_reply('<tool_call>\n{"name": "alarm.list", "arguments": {'), ParseError)
    assert isinstance(parse_reply('<tool_call>\n{"arguments": {}}\n</tool_call>'), ParseError)


def _agent(outputs: list[str], cfg: Any = CFG) -> tuple[ChatMLAgent, FakeLLM]:
    """`ChatMLAgent` với LLM giả trả lần lượt `outputs` cho mẫu `s1`."""
    counter = WhitespaceTokenCounter()
    llm = FakeLLM([], broken_outputs={"s1": outputs}, dialect="chatml")
    prompts = ChatMLPrompt(INDEX, counter, max_model_len=LLM.max_model_len, max_tokens=LLM.max_tokens)
    agent = ChatMLAgent(llm, GoldTranscriber(), prompts, build_guard_chain(cfg.guards), INDEX, counter, cfg, LLM)
    return agent, llm


def _call(name: str, **arguments: Any) -> str:
    """Một khối `<tool_call>` như Qwen sinh ra."""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def test_agent_call_then_final_text() -> None:
    """Lời gọi hợp lệ tới env, kết quả thành tin `tool`, câu kết bằng lời thì dừng `DONE`."""
    agent, _ = _agent([_call("alarm.delete", alarm_id="a1"), "Tôi đã thực hiện xong toàn bộ các thao tác bạn yêu cầu."])
    env = make_env()
    res = agent.run(_sample(), env, clock=FakeClock())
    assert res.stop_reason is StopReason.DONE
    assert [c["name"] for c in res.tool_calls] == ["alarm.delete"]
    assert res.errors == ()


def test_agent_blocked_call_becomes_tool_error_and_never_reaches_env() -> None:
    """Thiếu tham số bắt buộc: guard chặn, env không ghi lời gọi; mô hình sửa ở lượt sau."""
    agent, _ = _agent([_call("alarm.delete"), _call("alarm.delete", alarm_id="a1"), "Xong."])
    env = make_env()
    res = agent.run(_sample(), env, clock=FakeClock())
    assert res.stop_reason is StopReason.DONE
    assert [c["arguments"] for c in res.tool_calls] == [{"alarm_id": "a1"}]
    assert ErrorCode.INVALID_ARGS in res.errors
    blocked = res.steps[0]
    assert blocked.error is ErrorCode.INVALID_ARGS and blocked.result is None


def test_agent_messages_follow_lead_format() -> None:
    """Hội thoại sau một lượt: tin `assistant` có `tool_calls` id `call_curr_0`, rồi tin `tool` cùng id."""
    agent, _ = _agent([_call("alarm.list"), "Xong."])
    seen: list[list[dict[str, Any]]] = []
    inner = agent.llm.generate

    def spy(messages: list[dict[str, Any]], schema: Any, **kw: Any) -> Any:
        seen.append([dict(m) for m in messages])
        assert schema is not None and schema["type"] == "structural_tag"
        return inner(messages, schema, **kw)

    agent.llm.generate = spy  # type: ignore[method-assign]
    agent.run(_sample(), make_env(), clock=FakeClock())
    second = seen[1]
    assert second[0]["role"] == "system" and "tools" in second[0]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_curr_0"
    assert second[-2]["tool_calls"][0]["function"] == {"name": "alarm.list", "arguments": "{}"}
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_curr_0"


def test_agent_ask_stops() -> None:
    """Gọi `sys.ask_missing_info` thì dừng `ASKED`, không gọi thêm lượt nào."""
    agent, _ = _agent([_call(ASK_TOOL, question="báo thức nào?"), _call("alarm.list")])
    res = agent.run(_sample(), make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.ASKED
    assert [c["name"] for c in res.tool_calls] == [ASK_TOOL]


def test_agent_parse_error_retries() -> None:
    """Khối `<tool_call>` hỏng tốn một lượt thử lại; hết lượt thì dừng `PARSE_ERROR`."""
    bad = "<tool_call>\n{oops}\n</tool_call>"
    agent, _ = _agent([bad, bad, bad, bad])
    res = agent.run(_sample(), make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.PARSE_ERROR
    assert res.tool_calls == ()


def test_fake_llm_replays_gold_in_chatml() -> None:
    """`FakeLLM` ở định dạng ChatML phát lại nhãn thành khối `<tool_call>` rồi câu kết."""
    sample = dataclasses.replace(
        _sample(), label={"tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "a1"}}]}
    )
    counter = WhitespaceTokenCounter()
    llm = FakeLLM([sample], dialect="chatml")
    prompts = ChatMLPrompt(INDEX, counter, max_model_len=LLM.max_model_len, max_tokens=LLM.max_tokens)
    agent = ChatMLAgent(llm, GoldTranscriber(), prompts, build_guard_chain(CFG.guards), INDEX, counter, CFG, LLM)
    res = agent.run(sample, make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.DONE
    assert [c["arguments"] for c in res.tool_calls] == [{"alarm_id": "a1"}]


@pytest.mark.tokenizer
def test_qwen_template_renders_calls_that_parse_back() -> None:
    """Chat template thật của Qwen in tin `assistant` có `tool_calls` thành khối mà `parse_reply` đọc lại đúng."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not in the local cache")
    msgs = build_messages(_sample(), INDEX, GoldTranscriber().transcribe(_sample()))
    tools = msgs[0].pop("tools")
    msgs.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_curr_0",
                    "type": "function",
                    "function": {
                        "name": "alarm.delete",
                        "arguments": json.dumps({"alarm_id": "a1"}, ensure_ascii=False),
                    },
                }
            ],
        }
    )
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    reply = text.rsplit("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    assert parse_reply(reply) == [Action(kind="call", name="alarm.delete", arguments={"alarm_id": "a1"})]
    assert "<tools>" in text and '"name": "alarm.delete"' in text


def test_tool_call_constraint_shape() -> None:
    """Luật ép: mỗi tool một cấu trúc mở đúng chuỗi template của Qwen; bỏ `required`, mô tả, khoá `x-*`; cấm khoá lạ."""
    msgs = build_messages(_sample(), INDEX, GoldTranscriber().transcribe(_sample()))
    tag = tool_call_constraint(msgs[0]["tools"])
    assert is_structural_tag(tag) and tag["triggers"] == ["<tool_call>"]
    names = [t["function"]["name"] for t in msgs[0]["tools"]]
    assert [s["begin"] for s in tag["structures"]] == [f'<tool_call>\n{{"name": "{n}", "arguments": ' for n in names]
    assert all(s["end"] == "}\n</tool_call>" for s in tag["structures"])
    delete = tag["structures"][names.index("alarm.delete")]["schema"]
    assert delete["additionalProperties"] is False and "required" not in delete
    assert all(
        set(p) <= {"type", "enum", "items", "properties", "additionalProperties"} for p in delete["properties"].values()
    )


def test_constrain_tools_off_sends_no_constraint() -> None:
    """`constrain_tools = false`: agent gọi LLM không ràng buộc."""
    agent, _ = _agent([_call("alarm.list"), "Xong."], cfg=dataclasses.replace(CFG, constrain_tools=False))
    schemas: list[Any] = []
    inner = agent.llm.generate

    def spy(messages: list[dict[str, Any]], schema: Any, **kw: Any) -> Any:
        schemas.append(schema)
        return inner(messages, schema, **kw)

    agent.llm.generate = spy  # type: ignore[method-assign]
    agent.run(_sample(), make_env(), clock=FakeClock())
    assert schemas == [None, None]


@pytest.mark.tokenizer
def test_constraint_compiles_and_matches_with_xgrammar() -> None:
    """Biên dịch luật bằng xgrammar như vLLM: lời gọi đúng và câu kết qua; tool lạ, tham số lạ, sai thứ tự khoá bị chặn.

    Chạy khi máy có `xgrammar` và tokenizer Qwen trong cache (máy GPU có vLLM). Thứ tự khoá: xgrammar buộc tham số theo
    thứ tự khai báo trong schema; bộ chấm không xét thứ tự khoá nên không ảnh hưởng điểm.
    """
    xgr = pytest.importorskip("xgrammar")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not in the local cache")
    msgs = build_messages(_sample(), INDEX, GoldTranscriber().transcribe(_sample()))
    tag = tool_call_constraint(msgs[0]["tools"])
    items = [
        xgr.StructuralTagItem(begin=s["begin"], schema=json.dumps(s["schema"]), end=s["end"]) for s in tag["structures"]
    ]
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok)).compile_structural_tag(
        items, tag["triggers"]
    )

    def ok(text: str) -> bool:
        return bool(xgr.GrammarMatcher(compiled).accept_string(text))

    assert ok(_call("alarm.delete", alarm_id="a1"))
    assert ok("Tôi đã thực hiện xong toàn bộ các thao tác bạn yêu cầu.")
    assert ok(_call("alarm.delete"))  # thiếu tham số bắt buộc vẫn qua: để guard chặn rồi mô hình hỏi lại
    assert not ok(_call("alarm.set", alarm_id="a1"))
    assert not ok(_call("alarm.delete", alarm_id="a1", bia=1))


def test_long_tool_result_is_cut_instead_of_overflowing() -> None:
    """Kết quả tool quá dài: tin `tool` bị cắt còn `max_result_chars` ký tự, mẫu chạy tiếp thay vì dừng vì tràn."""
    huge = {"items": ["mục " + str(i) for i in range(3000)]}

    class BigEnv:
        """Môi trường giả trả một kết quả rất dài cho `alarm.list`."""

        def __init__(self) -> None:
            self.inner = make_env()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.inner.call(name, arguments)
            return huge

    llm_cfg = dataclasses.replace(LLM, max_model_len=3000, max_tokens=100)
    cfg = dataclasses.replace(CFG, max_result_chars=400)
    counter = WhitespaceTokenCounter()
    llm = FakeLLM([], broken_outputs={"s1": [_call("alarm.list"), "Xong."]}, dialect="chatml")
    prompts = ChatMLPrompt(INDEX, counter, max_model_len=llm_cfg.max_model_len, max_tokens=llm_cfg.max_tokens)
    agent = ChatMLAgent(llm, GoldTranscriber(), prompts, build_guard_chain(cfg.guards), INDEX, counter, cfg, llm_cfg)
    res = agent.run(_sample(), BigEnv(), clock=FakeClock())  # type: ignore[arg-type]
    assert res.stop_reason is StopReason.DONE


def test_prefill_reads_system_message_once_before_first_generate() -> None:
    """`prefill_prefix`: mỗi mẫu đọc trước đúng một lần, nội dung là tin `system` của prompt đầu, trước lần sinh đầu."""
    agent, llm = _agent([_call("alarm.list"), "Xong."])
    order: list[str] = []
    inner_prefill, inner_gen = llm.prefill, llm.generate

    def prefill(messages: list[dict[str, Any]]) -> float:
        order.append("prefill")
        return inner_prefill(messages)

    def generate(messages: list[dict[str, Any]], schema: Any, **kw: Any) -> Any:
        order.append("generate")
        assert messages[0] == llm.prefills[0][0]  # phần đầu đọc trước trùng tin đầu của prompt thật
        return inner_gen(messages, schema, **kw)

    llm.prefill, llm.generate = prefill, generate  # type: ignore[method-assign]
    res = agent.run(_sample(), make_env(), clock=FakeClock())
    assert order == ["prefill", "generate", "generate"]
    assert len(llm.prefills) == 1 and [m["role"] for m in llm.prefills[0]] == ["system"]
    assert "prefill_s" in res.timing


def test_prefill_off_and_prefill_errors_do_not_change_the_run() -> None:
    """Tắt `prefill_prefix` thì không đọc trước; đọc trước lỗi thì mẫu vẫn chạy như thường."""
    agent, llm = _agent([_call("alarm.list"), "Xong."], cfg=dataclasses.replace(CFG, prefill_prefix=False))
    agent.run(_sample(), make_env(), clock=FakeClock())
    assert llm.prefills == []

    agent, llm = _agent([_call("alarm.list"), "Xong."])

    def broken(messages: list[dict[str, Any]]) -> float:
        raise RuntimeError("engine busy")

    llm.prefill = broken  # type: ignore[method-assign]
    res = agent.run(_sample(), make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.DONE and res.timing["prefill_s"] == 0.0


@pytest.mark.tokenizer
def test_prefill_render_is_a_token_prefix_of_the_real_prompt() -> None:
    """Chat template thật của Qwen: tin `system` render không có lượt `assistant` là phần đầu (cả chuỗi lẫn dãy token)
    của prompt đầy đủ, nên bộ nhớ đệm phần đầu của vLLM khớp được."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not in the local cache")
    history = (
        Turn(role="user", audio="u1.wav", audio_text="có báo thức nào", text=None, tool_calls=(), tool_results=()),
        Turn(role="assistant", audio=None, audio_text=None, text="Có một báo thức.", tool_calls=(), tool_results=()),
    )
    sample = _sample(history)
    prompts = ChatMLPrompt(INDEX, WhitespaceTokenCounter(), max_model_len=LLM.max_model_len, max_tokens=LLM.max_tokens)
    full, _ = prompts.start(sample, GoldTranscriber().transcribe(sample))
    prefix = prompts.prefix(sample)
    tools = full[0]["tools"]
    strip = lambda ms: [{k: v for k, v in m.items() if k != "tools"} for m in ms]  # noqa: E731
    full_text = tok.apply_chat_template(strip(full), tools=tools, tokenize=False, add_generation_prompt=True)
    pre_text = tok.apply_chat_template(strip(prefix), tools=tools, tokenize=False, add_generation_prompt=False)
    assert full_text.startswith(pre_text) and len(pre_text) > 0.5 * len(full_text)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    pre_ids = tok(pre_text, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(pre_ids)] == pre_ids


def test_seen_text_excludes_the_call_being_checked() -> None:
    """`seen_text` gồm system, tool và kết quả trước đó, nhưng không gồm lượt `assistant` đang xét."""
    from lifemate.agent.chatml_agent import seen_text

    messages = [
        {"role": "system", "content": "luật", "tools": [{"function": {"name": "calendar.list_events"}}]},
        {"role": "user", "content": "dời lịch"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "calendar.list_events", "arguments": "{}"}}]},
        {"role": "tool", "content": '{"events": [{"id": "evt_01"}]}'},
        {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"event_id": "evt_99"}'}}]},
    ]
    text = seen_text(messages)
    assert "evt_01" in text and "calendar.list_events" in text and "evt_99" not in text


def test_extra_rules_are_inserted_after_lead_rules_and_default_is_unchanged() -> None:
    """`extra_rules` chèn mỗi quy tắc một dòng sau bốn quy tắc của lead, trước dòng thời gian; mặc định không đổi gì."""
    from lifemate.agent.chatml import build_system_prompt

    ctx = {"current_time": "2026-11-10T08:00:00+07:00"}
    base = build_system_prompt(ctx)
    ruled = build_system_prompt(ctx, ("Chỉ điền tham số người dùng nói.",))
    assert ruled == base.replace("- Thời gian hiện tại", "- Chỉ điền tham số người dùng nói.\n- Thời gian hiện tại")
    rules_file = Path(__file__).resolve().parent.parent / "configs" / "prompts" / "chatml_rules_v1.txt"
    assert len([x for x in rules_file.read_text(encoding="utf-8").splitlines() if x.strip()]) == 4


def test_free_blocks_do_not_spend_retries() -> None:
    """Chặn vì đọc lặp không tốn lượt thử lại: lặp ba lần (quá `max_retries` = 2) vẫn chạy tiếp tới câu kết."""
    cfg = dataclasses.replace(CFG, guards=["allowed_tool", "schema", "duplicate_read"])
    outputs = [_call("alarm.list")] * 4 + ["Xong."]
    agent, _ = _agent(outputs, cfg=cfg)
    res = agent.run(_sample(), make_env(), clock=FakeClock())
    assert res.stop_reason is StopReason.DONE
    assert [c["name"] for c in res.tool_calls] == ["alarm.list"]
    assert ErrorCode.DUPLICATE_READ in res.errors

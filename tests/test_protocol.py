"""T4, T13: output parser and per-sample schema (SPEC 4.7)."""

from __future__ import annotations

import pytest

from lifemate.protocol import DONE_TEXT, call_text, output_schema, parse, schema_json
from lifemate.tooldocs import ASK_TOOL
from lifemate.types import Action, ParseError

GOOD = [
    ('{"action":"done"}', Action("done", None, None)),
    ('  {"action": "done"}\n', Action("done", None, None)),
    ('{"action":"call","name":"sms.send","arguments":{"to":"a"}}', Action("call", "sms.send", {"to": "a"})),
    ('Đây là kết quả: {"action":"call","name":"a.b","arguments":{}} xong', Action("call", "a.b", {})),
    ('```json\n{"action":"done"}\n```', Action("done", None, None)),
    ('{"action":"call","name":"a.b","arguments":{"x":{"y":1}}}', Action("call", "a.b", {"x": {"y": 1}})),
]

BAD = [
    "",
    "xin chào",
    '{"action":"call","name":"a.b","arguments":{}',
    '{"action":"stop"}',
    '{"name":"a.b","arguments":{}}',
    '{"action":"call","name":"a.b","arguments":[]}',
    '{"action":"call","name":"a.b","arguments":"{}"}',
    '{"action":"call","arguments":{}}',
    '{"action":"call","name":3,"arguments":{}}',
    '[{"action":"done"}]',
    "} {",
    '"done"',
    "{" * 5000 + "}" * 5000,
]


@pytest.mark.parametrize(("text", "want"), GOOD)
def test_parse_good(text: str, want: Action) -> None:
    assert parse(text) == want


@pytest.mark.parametrize("text", BAD)
def test_parse_bad_never_raises(text: str) -> None:
    """T4: broken output gives ParseError, never an exception or an Action."""
    got = parse(text)
    assert isinstance(got, ParseError)
    assert got.raw == text


def test_round_trip_texts() -> None:
    assert parse(DONE_TEXT) == Action("done", None, None)
    assert parse(call_text("a.b", {"z": 1, "a": "ư"})) == Action("call", "a.b", {"z": 1, "a": "ư"})


def test_output_schema() -> None:
    """T13: names sorted, sys tool once, same tool set -> same schema string."""
    schema = output_schema(["sms.send", "alarm.list", ASK_TOOL])
    names = schema["anyOf"][0]["properties"]["name"]["enum"]
    assert names == sorted(names) == ["alarm.list", "sms.send", ASK_TOOL]
    assert output_schema(["b.x"])["anyOf"][0]["properties"]["name"]["enum"] == ["b.x", ASK_TOOL]
    assert schema_json(("sms.send", "alarm.list")) == schema_json(["alarm.list", "sms.send", ASK_TOOL])
    assert schema_json(["sms.send"]) != schema_json(["alarm.list"])
    assert schema["anyOf"][1] == {
        "type": "object",
        "properties": {"action": {"const": "done"}},
        "required": ["action"],
        "additionalProperties": False,
    }

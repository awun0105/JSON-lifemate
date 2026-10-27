"""T4, T13: kiểm tra bộ phân tích đầu ra của mô hình và schema đầu ra theo từng mẫu (SPEC 4.7).

Cách kiểm tra: gọi trực tiếp `parse`, `call_text`, `output_schema`, `schema_json` trên các chuỗi viết tay
(GOOD là đầu ra hợp lệ, BAD là đầu ra hỏng). Không dùng fake, không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import pytest

from lifemate.agent.protocol import DONE_TEXT, call_text, output_schema, parse, schema_json
from lifemate.agent.tooldocs import ASK_TOOL
from lifemate.core.types import Action, ParseError

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
    """Đầu ra hợp lệ (kể cả có khoảng trắng, chữ xung quanh hoặc khối ```json) được phân tích đúng thành Action."""
    assert parse(text) == want


@pytest.mark.parametrize("text", BAD)
def test_parse_bad_never_raises(text: str) -> None:
    """T4: đầu ra hỏng trả về ParseError giữ nguyên chuỗi gốc, không bao giờ ném ngoại lệ hay trả Action."""
    got = parse(text)
    assert isinstance(got, ParseError)
    assert got.raw == text


def test_round_trip_texts() -> None:
    """Chuỗi do DONE_TEXT và call_text sinh ra phân tích ngược lại đúng Action ban đầu, kể cả ký tự tiếng Việt."""
    assert parse(DONE_TEXT) == Action("done", None, None)
    assert parse(call_text("a.b", {"z": 1, "a": "ư"})) == Action("call", "a.b", {"z": 1, "a": "ư"})


def test_output_schema() -> None:
    """T13: tên tool được sắp xếp, tool sys chỉ xuất hiện một lần, cùng tập tool thì cùng chuỗi schema."""
    schema = output_schema(["sms.send", "alarm.list", ASK_TOOL])
    names = schema["anyOf"][0]["properties"]["name"]["enum"]
    assert names == sorted(names) == ["alarm.list", "sms.send", ASK_TOOL]
    # ASK_TOOL luôn được thêm vào dù danh sách đầu vào không có.
    assert output_schema(["b.x"])["anyOf"][0]["properties"]["name"]["enum"] == ["b.x", ASK_TOOL]
    assert schema_json(("sms.send", "alarm.list")) == schema_json(["alarm.list", "sms.send", ASK_TOOL])
    assert schema_json(["sms.send"]) != schema_json(["alarm.list"])
    # Nhánh thứ hai của anyOf là hành động "done", không cho thêm trường lạ.
    assert schema["anyOf"][1] == {
        "type": "object",
        "properties": {"action": {"const": "done"}},
        "required": ["action"],
        "additionalProperties": False,
    }

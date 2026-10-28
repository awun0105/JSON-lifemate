"""Quy định định dạng đầu ra của LLM: JSON schema cho từng mẫu và bộ parse chịu lỗi (SPEC 4.7).

Nội dung chính:
- `ACTION_CALL`, `ACTION_DONE`, `DONE_TEXT`: tên hai loại hành động và văn bản chuẩn của hành động `done`.
- `call_text`: văn bản chuẩn của một lời gọi tool.
- `output_schema`, `schema_json`: JSON schema dùng cho constrained decoding (ép LLM sinh đúng cấu trúc).
- `parse`: biến đầu ra của LLM thành `Action` hoặc `ParseError`.

Cách hoạt động:
- Mỗi lượt, LLM trả về đúng một JSON: `{"action": "call", "name", "arguments"}` hoặc `{"action": "done"}`.
- Schema chỉ cho `name` nhận các tool được phép của mẫu, luôn kèm `sys.ask_missing_info`. Tên tool được sắp xếp,
  nên hai mẫu cùng tập tool có schema giống hệt nhau từng byte. Schema được cache theo tập tool.
- `call_text` và `DONE_TEXT` dùng JSON chuẩn tắc, nên văn bản assistant lúc train khớp từng byte với văn bản
  model sinh ra lúc chạy.

Lưu ý:
- `parse` không bao giờ ném lỗi: đầu ra hỏng thành `ParseError`, để vòng lặp agent xử lý bằng cách thử lại.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from typing import Any

from lifemate.agent.tooldocs import ASK_TOOL
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Action, ParseError

ACTION_CALL = "call"
ACTION_DONE = "done"

DONE_TEXT = canonical_json({"action": ACTION_DONE})


def call_text(name: str, arguments: dict[str, Any]) -> str:
    """Tạo văn bản assistant chuẩn (JSON chuẩn tắc) cho một lời gọi tool."""
    return canonical_json({"action": ACTION_CALL, "name": name, "arguments": arguments})


@functools.lru_cache(maxsize=1024)
def _schema_cached(allowed: frozenset[str]) -> dict[str, Any]:
    """Dựng schema cho một tập tool; có cache vì nhiều mẫu dùng chung tập tool."""
    # Luôn thêm tool hỏi lại. Tên được sắp xếp để schema không phụ thuộc thứ tự duyệt của set.
    names = sorted(allowed | {ASK_TOOL})
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"const": ACTION_CALL},
                    "name": {"enum": names},
                    "arguments": {"type": "object"},
                },
                "required": ["action", "name", "arguments"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"action": {"const": ACTION_DONE}},
                "required": ["action"],
                "additionalProperties": False,
            },
        ]
    }


def output_schema(allowed_tools: Iterable[str]) -> dict[str, Any]:
    """Trả về JSON schema ràng buộc một lượt của LLM.

    Tên tool trong schema đã sắp xếp, và `sys.ask_missing_info` xuất hiện đúng một lần.

    Tham số:
        allowed_tools: tên các tool được phép của mẫu.

    Trả về:
        Dict schema. Dict này được cache và dùng chung, nên không được sửa trực tiếp.
    """
    return _schema_cached(frozenset(allowed_tools))


def schema_json(allowed_tools: Iterable[str]) -> str:
    """Trả về chuỗi JSON chuẩn tắc của `output_schema`; giống hệt nhau với các mẫu có cùng tập tool."""
    return canonical_json(output_schema(allowed_tools))


def _loads_object(text: str) -> Any:
    """Đọc JSON từ `text`; nếu lỗi thì thử lại với đoạn từ dấu `{` đầu tiên tới dấu `}` cuối cùng.

    Lỗi:
        json.JSONDecodeError: khi cả hai cách đều không đọc được.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model đôi khi bọc JSON trong văn bản thừa (lời giải thích, code fence), nên cắt lấy phần trong ngoặc nhọn.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse(text: str) -> Action | ParseError:
    """Parse một đầu ra của LLM thành `Action`; không bao giờ ném lỗi.

    Tham số:
        text: văn bản thô do model sinh ra.

    Trả về:
        `Action` loại `done` hoặc `call`, hoặc `ParseError` (kèm văn bản thô) khi đầu ra không hợp lệ.
    """
    stripped = text.strip()
    try:
        obj = _loads_object(stripped)
    except (json.JSONDecodeError, RecursionError) as exc:
        # `RecursionError` xảy ra khi JSON lồng quá sâu. Đó cũng chỉ là đầu ra hỏng, không được làm sập vòng lặp.
        return ParseError(message=f"invalid json: {exc}", raw=text)
    if not isinstance(obj, dict):
        return ParseError(message="output is not a JSON object", raw=text)
    action = obj.get("action")
    if action == ACTION_DONE:
        return Action(kind="done", name=None, arguments=None)
    if action != ACTION_CALL:
        return ParseError(message=f"unknown action: {action!r}", raw=text)
    name, arguments = obj.get("name"), obj.get("arguments")
    if not isinstance(name, str) or not name:
        return ParseError(message="call needs a non-empty string name", raw=text)
    if not isinstance(arguments, dict):
        return ParseError(message="call arguments must be an object", raw=text)
    return Action(kind="call", name=name, arguments=arguments)

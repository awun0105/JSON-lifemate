"""Định dạng đầu ra của LLM: JSON schema theo từng mẫu và bộ phân tích dễ dãi (SPEC 4.7).

Nội dung: hằng số hành động (`call`, `done`), hàm tạo văn bản chuẩn của một lời gọi, hàm sinh JSON schema
cho constrained decoding và hàm `parse` biến đầu ra của mô hình thành `Action`.

Cách xây dựng:
- Mỗi lượt mô hình trả về đúng một JSON: `{"action": "call", "name", "arguments"}` hoặc `{"action": "done"}`.
- Schema giới hạn `name` vào tập tool được phép của mẫu (luôn có `sys.ask_missing_info`), tên được sắp xếp
  để hai mẫu cùng tập tool cho ra schema giống hệt từng byte; schema được cache theo tập tool.
- `call_text` và `DONE_TEXT` dùng JSON chuẩn hóa, nên văn bản assistant khi huấn luyện trùng khớp từng byte
  với văn bản mô hình sinh ra lúc chạy.
- `parse` không bao giờ ném lỗi: đầu ra hỏng trở thành `ParseError` để vòng lặp agent xử lý bằng thử lại.
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
    """Văn bản assistant chuẩn của một lời gọi tool (JSON chuẩn hóa)."""
    return canonical_json({"action": ACTION_CALL, "name": name, "arguments": arguments})


@functools.lru_cache(maxsize=1024)
def _schema_cached(allowed: frozenset[str]) -> dict[str, Any]:
    """Dựng schema cho một tập tool; được cache vì nhiều mẫu dùng chung tập tool."""
    # Luôn thêm tool hỏi lại; sắp xếp tên để schema ổn định, không phụ thuộc thứ tự của set.
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
    """JSON schema ràng buộc một lượt của LLM; tên tool được sắp xếp, `sys.ask_missing_info` có đúng một lần.

    Dict trả về được cache và dùng chung: không được sửa trực tiếp.
    """
    return _schema_cached(frozenset(allowed_tools))


def schema_json(allowed_tools: Iterable[str]) -> str:
    """Chuỗi chuẩn hóa của `output_schema`; giống hệt nhau với các mẫu có cùng tập tool."""
    return canonical_json(output_schema(allowed_tools))


def _loads_object(text: str) -> Any:
    """Đọc JSON; nếu thất bại thì thử lại với đoạn từ `{` đầu tiên tới `}` cuối cùng.

    Raises:
        json.JSONDecodeError: khi cả hai cách đều không đọc được.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Mô hình đôi khi bọc JSON trong văn bản thừa (giải thích, code fence); cắt lấy phần ngoặc nhọn.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse(text: str) -> Action | ParseError:
    """Phân tích một đầu ra của LLM thành `Action`; không bao giờ ném lỗi.

    Args:
        text: văn bản thô mô hình sinh ra.

    Returns:
        `Action` loại `done` hoặc `call`, hoặc `ParseError` (kèm văn bản thô) khi đầu ra không hợp lệ.
    """
    stripped = text.strip()
    try:
        obj = _loads_object(stripped)
    except (json.JSONDecodeError, RecursionError) as exc:
        # RecursionError: JSON lồng quá sâu cũng chỉ là đầu ra hỏng, không được làm sập vòng lặp.
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

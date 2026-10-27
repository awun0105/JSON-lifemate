"""LLM output format: per-sample JSON schema and tolerant parser (SPEC 4.7)."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from typing import Any

from lifemate.jsonutil import canonical_json
from lifemate.tooldocs import ASK_TOOL
from lifemate.types import Action, ParseError

ACTION_CALL = "call"
ACTION_DONE = "done"

DONE_TEXT = canonical_json({"action": ACTION_DONE})


def call_text(name: str, arguments: dict[str, Any]) -> str:
    """Canonical assistant text of one tool call."""
    return canonical_json({"action": ACTION_CALL, "name": name, "arguments": arguments})


@functools.lru_cache(maxsize=1024)
def _schema_cached(allowed: frozenset[str]) -> dict[str, Any]:
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
    """JSON schema constraining one LLM turn; tool names sorted, `sys.ask_missing_info` included once.

    The returned dict is cached and shared: do not mutate it.
    """
    return _schema_cached(frozenset(allowed_tools))


def schema_json(allowed_tools: Iterable[str]) -> str:
    """Canonical string of `output_schema`; identical for samples with the same tool set."""
    return canonical_json(output_schema(allowed_tools))


def _loads_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def parse(text: str) -> Action | ParseError:
    """Parse one LLM output into an Action; never raises, returns ParseError on bad output."""
    stripped = text.strip()
    try:
        obj = _loads_object(stripped)
    except (json.JSONDecodeError, RecursionError) as exc:
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

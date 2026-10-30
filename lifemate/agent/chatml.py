"""Dựng hội thoại ChatML theo đúng định dạng tool calling của Qwen, giống hệt dữ liệu huấn luyện của repo lead.

Nội dung chính:
- `normalize_input_text`, `build_system_prompt`, `compact_tool_schema`, `tool_schemas`, `build_messages`: dựng danh sách
  tin nhắn của một mẫu, từng tin giống `src/utils/prompt_utils.py` › `build_messages` của repo lead.
- `tool_content`, `arguments_text`: cách viết kết quả tool và tham số lời gọi, như lead.
- `parse_reply`: đọc lượt trả lời của mô hình: các khối `<tool_call>…</tool_call>`, hoặc câu kết bằng lời.
- `tool_call_constraint`: luật ép đầu ra (structural tag của vLLM/xgrammar): sau `<tool_call>` mô hình chỉ viết được
  `{"name": <tool đã cài>, "arguments": <đúng tên, kiểu, enum của tham số>}`; câu kết bằng lời vẫn tự do.
- `ChatMLPrompt`: dựng hội thoại đầu tiên của mẫu và rút gọn khi quá dài.

Cách hoạt động (định dạng):
- Tin `system`: lời dặn ngắn của lead kèm thời gian, hồ sơ, vị trí; trường `tools` là danh sách schema
  `{"type": "function", "function": {name, description, parameters}}` của các tool đã cài cộng `sys.ask_missing_info`,
  xếp theo tên. Chat template của Qwen chèn danh sách này vào khối `<tools>`.
- Lịch sử: lượt người dùng là chữ (ASR hoặc đáp án) đã chuẩn hoá; lượt trợ lý có lời gọi thành tin `assistant` với
  `tool_calls`, mỗi kết quả một tin vai `tool`, rồi câu trả lời bằng lời (nếu có) thành một tin `assistant` riêng.
- Lượt hiện tại: tin `user` là chữ câu cuối. Sau đó mỗi lời gọi của mô hình là một tin `assistant` có `tool_calls`
  (id `call_curr_<k>`) và kết quả là một tin `tool`; mô hình kết thúc bằng một câu trả lời không có `<tool_call>`.

Lưu ý:
- Nguồn: Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/utils/prompt_utils.py@c703bd9
  (`build_system_prompt`, `compact_tool_schema`, `build_tools_schema_for_sample`, `build_messages`) và
  src/preprocessing/normalize.py (`normalize_input_text`). Chuỗi chữ, thứ tự khoá và cách `json.dumps` giữ nguyên, vì
  mô hình fine-tune trên dữ liệu của lead phải thấy đúng prompt đó lúc chạy (`tests/test_chatml.py` so với bản gốc).
- Khác bản gốc: lượt người dùng trong lịch sử lấy chữ ASR theo chỉ số lượt (`Transcript.history`) thay vì theo đường dẫn
  audio; kết quả như nhau. Ở train mode, app held-out bị bỏ khỏi `installed_apps` trước khi dựng danh sách tool, giống
  `train_seen.jsonl` của lead.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from lifemate.agent.prompt import HeldOutLeak
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.core.types import Action, ParseError, Sample, TokenCounter, Transcript

SHRINK_COMPACT = 1
SHRINK_HISTORY = 2
_MIN_HISTORY_TURNS = 2

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def normalize_input_text(text: str | None) -> str:
    """Chuẩn hoá chữ đầu vào như lead: Unicode NFC và gộp khoảng trắng; chuỗi rỗng khi `text` rỗng."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def build_system_prompt(context: Mapping[str, Any]) -> str:
    """Lời dặn hệ thống của lead, kèm thời gian, múi giờ, hồ sơ và vị trí của mẫu."""
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    user_prof = context.get("user_profile", {})
    location = context.get("location", {})
    prompt = (
        "Bạn là LifeMate, trợ lý AI thông minh điều khiển thiết bị và dịch vụ qua công cụ (tools).\n"
        "Quy tắc quan trọng:\n"
        "- Chỉ gọi các công cụ có trong danh sách được cung cấp.\n"
        "- Tận dụng thông tin và kết quả đã có trong lịch sử trò chuyện; "
        "không gọi lại các công cụ tra cứu nếu dữ liệu đã có sẵn.\n"
        "- Nếu thiếu thông tin bắt buộc không thể suy đoán từ ngữ cảnh, "
        "hãy gọi 'sys.ask_missing_info' để hỏi lại người dùng.\n"
        "- Khi hoàn tất mọi hành động hoặc nếu yêu cầu không cần gọi công cụ nào, "
        "hãy trả lời bằng văn bản kết thúc lượt.\n"
        f"- Thời gian hiện tại: {current_time} (Múi giờ: {timezone})\n"
    )
    if user_prof:
        prompt += f"- Hồ sơ người dùng: {json.dumps(user_prof, ensure_ascii=False)}\n"
    if location:
        prompt += f"- Vị trí hiện tại: {json.dumps(location, ensure_ascii=False)}\n"
    return prompt.strip()


def compact_tool_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Bản rút gọn của một schema tool: giữ tên, mô tả, kiểu, enum, ràng buộc; bỏ mô tả của từng tham số con."""
    fn = tool.get("function", {})
    params = fn.get("parameters", {})
    compact_props: dict[str, Any] = {}
    for p_name, p_info in params.get("properties", {}).items():
        if isinstance(p_info, dict):
            compact_props[p_name] = {
                k: p_info[k] for k in ("type", "enum", "items", "required", "format", "x-ref") if k in p_info
            }
        else:
            compact_props[p_name] = p_info
    compact_params: dict[str, Any] = {"type": params.get("type", "object"), "properties": compact_props}
    if "required" in params:
        compact_params["required"] = params["required"]
    return {
        "type": "function",
        "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": compact_params,
        },
    }


def _function(spec: Mapping[str, Any], default_params: Any) -> dict[str, Any]:
    """Một mục `{"type": "function", ...}` từ mục catalog của tool."""
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", default_params),
        },
    }


def tool_schemas(
    index: CatalogIndex, installed_apps: Mapping[str, Any], *, compact: bool = False
) -> list[dict[str, Any]]:
    """Schema các tool của app đã cài cộng `sys.ask_missing_info`, xếp theo tên (như lead lúc không xáo trộn).

    Tham số:
        index: chỉ mục catalog của split.
        installed_apps: `context.installed_apps` của mẫu (đã bỏ app held-out nếu ở train mode).
        compact: `True` thì rút gọn bằng `compact_tool_schema`.
    """
    out: list[dict[str, Any]] = []
    for app_id, app_info in installed_apps.items():
        allowed = None
        if isinstance(app_info, dict) and isinstance(app_info.get("tools"), list):
            allowed = set(app_info["tools"])
        if app_id not in index.apps:
            continue
        for tool in index.app_spec(app_id).get("tools", []):
            if allowed is None or tool["name"] in allowed:
                out.append(_function(tool, {"type": "object", "properties": {}}))
    if index.has_tool(ASK_TOOL):
        out.append(_function(index.tool_spec(ASK_TOOL), {}))
    if compact:
        out = [compact_tool_schema(t) for t in out]
    out.sort(key=lambda t: t["function"]["name"])
    return out


def arguments_text(arguments: Any) -> str:
    """Tham số của một lời gọi dạng chuỗi JSON, như lead (`json.dumps(..., ensure_ascii=False)`)."""
    return json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)


def tool_content(result: Any) -> str:
    """Nội dung tin `tool` cho một kết quả, như lead."""
    return json.dumps(result, ensure_ascii=False) if isinstance(result, dict | list) else str(result)


def assistant_call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    """Tin `assistant` chứa đúng một lời gọi tool."""
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments_text(arguments)}}
        ],
    }


def tool_message(call_id: str, name: str, result: Any) -> dict[str, Any]:
    """Tin vai `tool` mang kết quả của lời gọi `call_id`."""
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": tool_content(result)}


def build_messages(
    sample: Sample,
    index: CatalogIndex,
    transcript: Transcript,
    *,
    installed_apps: Mapping[str, Any] | None = None,
    compact: bool = False,
    skip_turns: int = 0,
) -> list[dict[str, Any]]:
    """Danh sách tin nhắn của một mẫu trước lượt trả lời đầu tiên của mô hình.

    Tham số:
        sample: mẫu cần dựng.
        index: chỉ mục catalog.
        transcript: chữ của câu cuối và các câu người dùng trong lịch sử.
        installed_apps: app đã cài dùng để dựng danh sách tool; `None` thì lấy từ `sample.context`.
        compact: rút gọn schema tool.
        skip_turns: số lượt lịch sử cũ nhất bị bỏ (rút gọn khi quá dài).
    """
    context = dict(sample.context)
    apps = installed_apps if installed_apps is not None else (context.get("installed_apps") or {})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(context), "tools": tool_schemas(index, apps, compact=compact)}
    ]
    for i, turn in enumerate(sample.history):
        if i < skip_turns:
            continue
        if turn.role == "user":
            asr = transcript.history[i] if i < len(transcript.history) else None
            messages.append({"role": "user", "content": normalize_input_text(asr or turn.audio_text or "")})
        elif turn.role == "assistant":
            calls = list(turn.tool_calls)
            if calls:
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"call_hist_{k}",
                                "type": "function",
                                "function": {"name": c["name"], "arguments": arguments_text(c.get("arguments", {}))},
                            }
                            for k, c in enumerate(calls)
                        ],
                    }
                )
                for k, res in enumerate(turn.tool_results):
                    name = calls[k]["name"] if k < len(calls) else "tool"
                    messages.append(tool_message(f"call_hist_{k}", name, res))
            if turn.text:
                messages.append({"role": "assistant", "content": turn.text})
    messages.append({"role": "user", "content": normalize_input_text(transcript.final)})
    return messages


def parse_reply(text: str) -> list[Action] | ParseError | None:
    """Đọc lượt trả lời của mô hình.

    Trả về:
        Danh sách `Action` loại `call` (một cho mỗi khối `<tool_call>`); `None` khi không có khối nào, tức mô hình trả
        lời bằng lời và kết thúc lượt; `ParseError` khi có khối mà nội dung không phải
        `{"name": str, "arguments": {...}}`.
    """
    blocks = _TOOL_CALL_RE.findall(text)
    if not blocks:
        # Thẻ mở không có thẻ đóng: mô hình bị cắt giữa lời gọi, không phải câu kết.
        return ParseError(message="unclosed <tool_call>", raw=text) if "<tool_call>" in text else None
    actions: list[Action] = []
    for block in blocks:
        try:
            obj = json.loads(block)
        except (json.JSONDecodeError, RecursionError) as exc:
            return ParseError(message=f"invalid tool_call json: {exc}", raw=text)
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
            return ParseError(message="tool_call needs a non-empty string name", raw=text)
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                return ParseError(message=f"tool_call arguments are not json: {exc}", raw=text)
        if not isinstance(args, dict):
            return ParseError(message="tool_call arguments must be an object", raw=text)
        actions.append(Action(kind="call", name=obj["name"], arguments=args))
    return actions


# Khoá JSON schema giữ lại trong luật ép. `required`, `format`, giới hạn số và khoá riêng của BTC (`x-ref`, `x-match`,
# `x-one-of`) bị bỏ: guard (`env.validate`) vẫn kiểm tra các phần này. Không ép `required` vì mô hình đã mở lời gọi mà
# thiếu thông tin thì sẽ phải bịa giá trị, trong khi đáp án đúng là bị guard chặn rồi hỏi lại (`sys.ask_missing_info`).
_SCHEMA_KEYS = ("type", "enum", "items", "properties")
TOOL_CALL_BEGIN = "<tool_call>"


def _constraint_schema(schema: Any) -> Any:
    """Bản JSON schema dùng để ép: chỉ giữ `_SCHEMA_KEYS`; object có `properties` thì cấm khoá ngoài danh sách."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {k: schema[k] for k in _SCHEMA_KEYS if k in schema}
    if isinstance(out.get("items"), dict):
        out["items"] = _constraint_schema(out["items"])
    if isinstance(out.get("properties"), dict):
        out["properties"] = {name: _constraint_schema(p) for name, p in out["properties"].items()}
        out["additionalProperties"] = False
    elif out.get("type") == "object":
        out.setdefault("properties", {})
        out["additionalProperties"] = False
    return out


def tool_call_constraint(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Luật ép đầu ra dạng structural tag cho một danh sách tool (trường `tools` của tin `system`).

    Mỗi tool một cấu trúc: mở bằng đúng chuỗi mà chat template của Qwen in cho lời gọi
    (`<tool_call>\n{"name": "<tên>", "arguments": `), giữa là JSON tham số theo `_constraint_schema`, đóng bằng
    `}\n</tool_call>`. Luật chỉ bật khi mô hình viết `<tool_call>` (trigger); ngoài khối đó mô hình viết tự do.

    Trả về:
        Dict theo định dạng `structural_tag` của vLLM:
        `{"type": "structural_tag", "structures": [...], "triggers": [...]}`.
    """
    structures = []
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        structures.append(
            {
                "begin": f'{TOOL_CALL_BEGIN}\n{{"name": "{fn["name"]}", "arguments": ',
                "schema": _constraint_schema(params),
                "end": "}\n</tool_call>",
            }
        )
    return {"type": "structural_tag", "structures": structures, "triggers": [TOOL_CALL_BEGIN]}


class ChatMLPrompt:
    """Dựng hội thoại đầu tiên của một mẫu ở định dạng ChatML và rút gọn khi quá dài.

    Tham số:
        index: chỉ mục catalog.
        counter: bộ đếm token (đếm cả danh sách tool trong tin `system`).
        max_model_len: độ dài context tối đa.
        max_tokens: số token sinh tối đa mỗi bước; phải chừa chỗ này.
        compact: rút gọn schema tool ngay từ mức 0.
        held_out_apps: app bị giấu ở train mode.
    """

    def __init__(
        self,
        index: CatalogIndex,
        counter: TokenCounter,
        *,
        max_model_len: int,
        max_tokens: int,
        compact: bool = False,
        held_out_apps: frozenset[str] = frozenset(),
    ) -> None:
        self._index = index
        self._counter = counter
        self._max_model_len = max_model_len
        self._max_tokens = max_tokens
        self._compact = compact
        self._held_out = held_out_apps

    def installed(self, sample: Sample, train_mode: bool) -> dict[str, Any]:
        """`installed_apps` của mẫu; ở train mode bỏ app held-out."""
        apps: dict[str, Any] = dict(sample.context.get("installed_apps") or {})
        if train_mode:
            apps = {a: v for a, v in apps.items() if a not in self._held_out}
        return apps

    def allowed_tools(self, sample: Sample, train_mode: bool = False) -> tuple[str, ...]:
        """Tên các tool mẫu được phép gọi (cùng tập với danh sách `tools` trong prompt)."""
        return tuple(t["function"]["name"] for t in tool_schemas(self._index, self.installed(sample, train_mode)))

    def _check_leak(self, sample: Sample) -> None:
        """Báo `HeldOutLeak` khi lịch sử đã gọi tool của app held-out (mẫu không dùng được ở train mode)."""
        for turn in sample.history:
            for call in turn.tool_calls if turn.role == "assistant" else ():
                app = self._index.app_of(str(call.get("name", "")))
                if app in self._held_out:
                    raise HeldOutLeak(sample.id, app)

    def fits(self, messages: list[dict[str, Any]]) -> bool:
        """`True` khi prompt cộng `max_tokens` vẫn nằm trong `max_model_len`."""
        return self._counter.count(messages) + self._max_tokens <= self._max_model_len

    def start(
        self, sample: Sample, transcript: Transcript, *, train_mode: bool = False
    ) -> tuple[list[dict[str, Any]], int]:
        """Hội thoại đầu tiên và mức rút gọn: 0 nguyên bản; 1 schema rút gọn; 2 bỏ dần các cặp lượt lịch sử cũ nhất."""
        if train_mode:
            self._check_leak(sample)
        apps = self.installed(sample, train_mode)
        msgs = build_messages(sample, self._index, transcript, installed_apps=apps, compact=self._compact)
        if self.fits(msgs):
            return msgs, 0
        msgs = build_messages(sample, self._index, transcript, installed_apps=apps, compact=True)
        if self.fits(msgs):
            return msgs, SHRINK_COMPACT
        skip = 0
        while len(sample.history) - skip - 2 >= _MIN_HISTORY_TURNS:
            skip += 2
            msgs = build_messages(sample, self._index, transcript, installed_apps=apps, compact=True, skip_turns=skip)
            if self.fits(msgs):
                break
        return msgs, SHRINK_HISTORY

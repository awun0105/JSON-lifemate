"""Bản chép nguyên văn hàm dựng tin nhắn của repo lead, chỉ dùng làm đáp án trong `tests/test_chatml.py`.

Nguồn: ThanhDanh1510/ZaloAI2026 src/utils/prompt_utils.py@c703bd9 (`build_system_prompt`, `compact_tool_schema`,
`build_tools_schema_for_sample`, `build_messages`) và src/preprocessing/normalize.py (`normalize_input_text`). Không sửa
thân hàm: test so `lifemate.agent.chatml` với đúng mã này. Khi lead đổi `prompt_utils.py`, chép lại file này.
"""
# ruff: noqa
# mypy: ignore-errors
# fmt: off

import json
import random
import unicodedata
from typing import Dict, List, Any, Optional


def normalize_input_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def build_system_prompt(context: dict) -> str:
    """
    Xây dựng system prompt ngắn gọn, chính xác kèm ngữ cảnh của người dùng.
    """
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    user_prof = context.get("user_profile", {})
    location = context.get("location", {})

    prompt = (
        "Bạn là LifeMate, trợ lý AI thông minh điều khiển thiết bị và dịch vụ qua công cụ (tools).\n"
        "Quy tắc quan trọng:\n"
        "- Chỉ gọi các công cụ có trong danh sách được cung cấp.\n"
        "- Tận dụng thông tin và kết quả đã có trong lịch sử trò chuyện; không gọi lại các công cụ tra cứu nếu dữ liệu đã có sẵn.\n"
        "- Nếu thiếu thông tin bắt buộc không thể suy đoán từ ngữ cảnh, hãy gọi 'sys.ask_missing_info' để hỏi lại người dùng.\n"
        "- Khi hoàn tất mọi hành động hoặc nếu yêu cầu không cần gọi công cụ nào, hãy trả lời bằng văn bản kết thúc lượt.\n"
        f"- Thời gian hiện tại: {current_time} (Múi giờ: {timezone})\n"
    )
    if user_prof:
        prompt += f"- Hồ sơ người dùng: {json.dumps(user_prof, ensure_ascii=False)}\n"
    if location:
        prompt += f"- Vị trí hiện tại: {json.dumps(location, ensure_ascii=False)}\n"

    return prompt.strip()


def compact_tool_schema(tool_dict: dict) -> dict:
    """
    Rút gọn schema của tool theo Thang rút gọn Mức 2 (EDA Mục 26):
    - Giữ nguyên tên hàm, mô tả hàm, type, enum, required, items, format.
    - Lược bỏ description chi tiết của từng tham số con để tiết kiệm ~55-60% dung lượng prompt.
    """
    fn = tool_dict.get("function", {})
    params = fn.get("parameters", {})
    props = params.get("properties", {})

    compact_props = {}
    for p_name, p_info in props.items():
        if isinstance(p_info, dict):
            cleaned = {}
            for k in ["type", "enum", "items", "required", "format", "x-ref"]:
                if k in p_info:
                    cleaned[k] = p_info[k]
            compact_props[p_name] = cleaned
        else:
            compact_props[p_name] = p_info

    compact_params = {
        "type": params.get("type", "object"),
        "properties": compact_props
    }
    if "required" in params:
        compact_params["required"] = params["required"]

    return {
        "type": "function",
        "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": compact_params
        }
    }


def build_tools_schema_for_sample(
    context: dict,
    catalog_indexed: dict,
    shuffle: bool = False,
    seed: Optional[int] = None,
    compact: bool = False
) -> list:
    """
    Trích xuất schema các tool thuộc context.installed_apps của sample + sys.ask_missing_info.
    Hỗ trợ dynamic tool shuffling để chống thiên vị vị trí (positional bias) và compact schema.
    """
    installed_apps = context.get("installed_apps", {})
    available_tools = []

    # 1. Các tool từ installed_apps
    for app_id, app_info in installed_apps.items():
        allowed_tool_names = None
        if isinstance(app_info, dict) and "tools" in app_info and isinstance(app_info["tools"], list):
            allowed_tool_names = set(app_info["tools"])

        app_def = catalog_indexed["apps"].get(app_id, {})
        for tool in app_def.get("tools", []):
            t_name = tool["name"]
            if allowed_tool_names is None or t_name in allowed_tool_names:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    }
                })

    # 2. Luôn bổ sung sys.ask_missing_info
    ask_def = catalog_indexed["tools"].get("sys.ask_missing_info")
    if ask_def:
        available_tools.append({
            "type": "function",
            "function": {
                "name": "sys.ask_missing_info",
                "description": ask_def.get("description", ""),
                "parameters": ask_def.get("parameters", {}),
            }
        })

    # 3. Rút gọn schema nếu bật cờ compact
    if compact:
        available_tools = [compact_tool_schema(t) for t in available_tools]

    # 4. Dynamic Tool Shuffling khi huấn luyện
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(available_tools)
    else:
        # Giữ thứ tự cố định ổn định theo tên tool
        available_tools.sort(key=lambda x: x["function"]["name"])

    return available_tools


def build_messages(
    sample: dict,
    catalog_indexed: dict,
    current_transcript: Optional[str] = None,
    history_transcripts: Optional[Dict[str, str]] = None,
    trajectory_steps: Optional[List[Dict[str, Any]]] = None,
    compact_schema: bool = False,
    shuffle_tools: bool = False,
    seed: Optional[int] = None
) -> list:
    """
    Hàm DUY NHẤT dùng chung để dựng danh sách tin nhắn hội thoại (ChatML messages).
    Được gọi ở cả script sinh dữ liệu huấn luyện (builder.py) lẫn Agent suy luận thời gian thực (agent.py).
    
    Tham số:
    - sample: Dữ liệu sample thô gồm context, history, audio, audio_text...
    - catalog_indexed: Từ điển schema toàn bộ công cụ.
    - current_transcript: Câu nói của lượt hiện tại (ASR transcript hoặc text sạch).
    - history_transcripts: Dict ánh xạ audio_path -> ASR transcript cho các lượt history.
    - trajectory_steps: Danh sách các bước gọi tool đã thực hiện trong quá khứ của lượt hiện tại:
      [{'tool_call': {'name': ..., 'arguments': ...}, 'tool_result': {...}}, ...]
    - compact_schema: Bật/tắt rút gọn mô tả tham số con để tiết kiệm 50% token.
    - shuffle_tools: Xáo trộn ngẫu nhiên thứ tự tools (dùng lúc train).
    """
    context = sample.get("context", {})
    tools_schema = build_tools_schema_for_sample(
        context,
        catalog_indexed,
        shuffle=shuffle_tools,
        seed=seed,
        compact=compact_schema
    )
    system_text = build_system_prompt(context)

    messages = [
        {"role": "system", "content": system_text, "tools": tools_schema}
    ]

    # 1. Xử lý lịch sử (history)
    history = sample.get("history", [])
    for h_turn in history:
        role = h_turn.get("role")
        if role == "user":
            h_audio = h_turn.get("audio")
            user_text = ""
            if history_transcripts and h_audio in history_transcripts and history_transcripts[h_audio]:
                user_text = history_transcripts[h_audio]
            else:
                user_text = h_turn.get("audio_text", "")
            messages.append({"role": "user", "content": normalize_input_text(user_text)})

        elif role == "assistant":
            calls = h_turn.get("tool_calls", [])
            results = h_turn.get("tool_results", [])
            reply_text = h_turn.get("text", "")

            if calls:
                tool_calls_formatted = []
                for idx, c in enumerate(calls):
                    args_val = c.get("arguments", {})
                    args_str = json.dumps(args_val, ensure_ascii=False) if isinstance(args_val, dict) else str(args_val)
                    tool_calls_formatted.append({
                        "id": f"call_hist_{idx}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": args_str
                        }
                    })
                messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls_formatted
                })

                for idx, res in enumerate(results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_hist_{idx}",
                        "name": calls[idx]["name"] if idx < len(calls) else "tool",
                        "content": json.dumps(res, ensure_ascii=False) if isinstance(res, (dict, list)) else str(res)
                    })

            if reply_text:
                messages.append({
                    "role": "assistant",
                    "content": reply_text
                })

    # 2. Xử lý lượt hiện tại (current turn)
    curr_user_text = current_transcript if current_transcript is not None else sample.get("audio_text", "")
    messages.append({"role": "user", "content": normalize_input_text(curr_user_text)})

    # 3. Nạp quỹ đạo các bước tool call đã thực hiện trong quá khứ của lượt hiện tại (nếu có)
    if trajectory_steps:
        for step_idx, step in enumerate(trajectory_steps):
            call = step["tool_call"]
            res = step.get("tool_result", {"ok": True})
            call_id = f"call_curr_{step_idx}"

            args_val = call.get("arguments", {})
            args_str = json.dumps(args_val, ensure_ascii=False) if isinstance(args_val, dict) else str(args_val)

            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": args_str
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": call["name"],
                "content": json.dumps(res, ensure_ascii=False) if isinstance(res, (dict, list)) else str(res)
            })

    return messages

"""Dựng prompt, hội thoại chỉ nối thêm và thang thu gọn prompt (SPEC 4.6).

Nội dung: các hàm sinh khối ngữ cảnh, lịch, lịch sử hội thoại; ngoại lệ `HeldOutLeak`; lớp `Conversation`
(danh sách tin nhắn chỉ nối thêm, đếm token tăng dần) và lớp `PromptBuilder` dựng hội thoại đầu của mẫu.

Cách xây dựng:
- System prompt = phần quy tắc trước `{tool_docs}` + tài liệu tool của mẫu + phần đuôi. Phần quy tắc giống
  hệt cho mọi mẫu, nên backend tái dùng prefix cache giữa các mẫu.
- Tin nhắn user đầu tiên gồm `# Ngữ cảnh`, `# Lịch sử hội thoại` (nếu có) và `# Yêu cầu hiện tại`.
- `Conversation` chỉ nối thêm, không sửa tin cũ: tiền tố prompt giữ nguyên qua các bước (prefix cache), và
  prompt lúc chạy trùng từng byte với prompt dùng khi sinh dữ liệu huấn luyện.
- Token được đếm tăng dần: tin assistant dùng số completion token backend báo, tin user đếm riêng phần mới;
  `reconcile` lấy số của backend làm chuẩn khi lệch.
- Thang thu gọn khi prompt không vừa ngữ cảnh: 0 nguyên bản; 1 cắt kết quả tool còn 1/4; 2 bỏ thêm mô tả
  tham số; 3 bỏ dần các lượt lịch sử cũ theo cặp, giữ ít nhất `_MIN_HISTORY_TURNS` lượt.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

import structlog

from lifemate.agent.tooldocs import CatalogIndex, render_tools
from lifemate.core.jsonutil import canonical_json, truncate
from lifemate.core.types import Sample, TokenCounter, Transcript, Turn

log = structlog.get_logger(__name__)

TOOL_DOCS_SLOT = "{tool_docs}"
SHRINK_RESULTS = 1
SHRINK_PARAM_DESCRIPTIONS = 2
SHRINK_HISTORY = 3
_MIN_HISTORY_TURNS = 2


def truncate_result(result: Mapping[str, Any], max_chars: int) -> str:
    """JSON chuẩn hóa của một kết quả tool, cắt còn tối đa `max_chars` ký tự."""
    return truncate(canonical_json(result), max_chars)


def _value_text(value: Any) -> str:
    """Chuỗi giữ nguyên, giá trị khác đổi sang JSON chuẩn hóa."""
    return value if isinstance(value, str) else canonical_json(value)


_WEEKDAYS = ("thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật")
_CALENDAR_DAYS = 14


def render_calendar(current_time: str) -> list[str]:
    """Thứ và ngày của hôm qua, hôm nay và 13 ngày tới, tính từ `current_time` (ISO 8601).

    Mô hình hay sai "thứ sáu", "chủ nhật này" và "mai" khi phải tự đếm ngày; liệt kê sẵn giúp bỏ phép tính
    đó. Tuần bắt đầu từ thứ Hai, nên "tuần này" / "tuần sau" được ghi rõ.
    Trả về danh sách rỗng khi không đọc được `current_time`.
    """
    try:
        now = dt.datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return []
    today = now.date()
    # Thứ Hai của tuần hiện tại, mốc để tính "tuần này", "tuần sau"...
    week0 = today - dt.timedelta(days=today.weekday())
    names = {-1: "hôm qua", 0: "hôm nay", 1: "mai", 2: "mốt / ngày kia"}
    lines = [f"Lịch (bây giờ là {_WEEKDAYS[today.weekday()]} {today.isoformat()} {now.strftime('%H:%M')}):"]
    for offset in range(-1, _CALENDAR_DAYS):
        day = today + dt.timedelta(days=offset)
        weeks = (day - week0).days // 7
        week = {0: "tuần này", 1: "tuần sau", -1: "tuần trước"}.get(weeks, f"{weeks} tuần nữa")
        tag = f", {names[offset]}" if offset in names else ""
        lines.append(f"  {day.isoformat()} {_WEEKDAYS[day.weekday()]} ({week}{tag})")
    return lines


def render_context(context: Mapping[str, Any]) -> str:
    """Khối `# Ngữ cảnh`: thời gian, múi giờ, lịch, vị trí, user_profile; không bao giờ có installed_apps.

    installed_apps không cần đưa vào vì tài liệu tool trong system prompt đã chỉ gồm các tool được phép.
    """
    lines = ["# Ngữ cảnh"]
    for key in ("current_time", "timezone"):
        if key in context:
            lines.append(f"{key}: {_value_text(context[key])}")
    if isinstance(context.get("current_time"), str):
        lines.extend(render_calendar(context["current_time"]))
    loc = context.get("location")
    # Vị trí dạng {lat, lng} được in gọn; dạng khác in nguyên JSON.
    if isinstance(loc, Mapping) and "lat" in loc and "lng" in loc:
        lines.append(f"location: {loc['lat']}, {loc['lng']}")
    elif loc is not None:
        lines.append(f"location: {_value_text(loc)}")
    for key, value in (context.get("user_profile") or {}).items():
        lines.append(f"user_profile.{key}: {_value_text(value)}")
    return "\n".join(lines)


def render_history(history: Sequence[Turn], transcript_history: Sequence[str | None], max_result_chars: int) -> str:
    """Khối `# Lịch sử hội thoại`, kết quả tool bị cắt ngắn; chuỗi rỗng khi không có lịch sử.

    Args:
        history: các lượt hội thoại trước của mẫu.
        transcript_history: transcript ASR theo cùng chỉ số với `history`; None nghĩa là dùng `audio_text`.
        max_result_chars: số ký tự tối đa của mỗi kết quả tool.
    """
    if not history:
        return ""
    lines = ["# Lịch sử hội thoại"]
    for i, turn in enumerate(history):
        if turn.role == "user":
            # Ưu tiên transcript (giống điều kiện lúc chạy thật), thiếu thì dùng văn bản gốc.
            text = transcript_history[i] if i < len(transcript_history) else None
            lines.append(f"Người dùng: {text if text is not None else (turn.audio_text or '')}")
            continue
        lines.append(f"Trợ lý: {turn.text or ''}".rstrip())
        for j, call in enumerate(turn.tool_calls):
            line = f"  [đã gọi] {call.get('name')}({canonical_json(call.get('arguments') or {})})"
            if j < len(turn.tool_results):
                line += f" → {truncate_result(turn.tool_results[j], max_result_chars)}"
            lines.append(line)
    return "\n".join(lines)


class HeldOutLeak(Exception):  # noqa: N818 - name fixed by SPEC 4.5
    """Mẫu ở train mode có lịch sử gọi tool của một app held-out (SPEC 4.5, A4).

    Args:
        sample_id: id của mẫu bị rò rỉ.
        app: id app held-out xuất hiện trong lịch sử.
    """

    def __init__(self, sample_id: str, app: str) -> None:
        super().__init__(f"{sample_id}: history calls held-out app {app}")
        self.sample_id = sample_id
        self.app = app


class Conversation:
    """Danh sách tin nhắn chỉ nối thêm, có bộ đếm token prompt tăng dần.

    Args:
        messages: các tin nhắn ban đầu (được sao chép, tin gốc không bị sửa).
        counter: bộ đếm token theo chat template của mô hình.
    """

    def __init__(self, messages: Sequence[dict[str, Any]], counter: TokenCounter) -> None:
        # Lưu dạng tuple bất biến để tin cũ không bị sửa ngoài ý muốn (giữ prefix cache).
        self._messages: tuple[dict[str, Any], ...] = tuple(dict(m) for m in messages)
        self._counter = counter
        self._tokens = counter.count(list(self._messages))

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        """Các tin nhắn hiện có (chỉ đọc)."""
        return self._messages

    def as_list(self) -> list[dict[str, Any]]:
        """Bản sao các tin nhắn ở dạng list mà backend LLM nhận."""
        return [dict(m) for m in self._messages]

    def _overhead(self, role: str) -> int:
        """Số token chat template thêm vào quanh một tin nhắn của `role`.

        Dùng `template_overhead` của bộ đếm nếu có, nếu không thì đếm một tin nhắn rỗng.
        """
        fn = getattr(self._counter, "template_overhead", None)
        if callable(fn):
            return int(fn(role))
        return self._counter.count([{"role": role, "content": ""}])

    def append_assistant(self, text: str, completion_tokens: int) -> None:
        """Nối nguyên văn đầu ra của LLM; số token = completion token + overhead của template."""
        self._messages = (*self._messages, {"role": "assistant", "content": text})
        # Không đếm lại văn bản: số completion token backend báo là con số chính xác.
        self._tokens += completion_tokens + self._overhead("assistant")

    def append_user(self, text: str) -> None:
        """Nối một tin nhắn user (kết quả tool hoặc lỗi từ guard), chỉ đếm phần tin mới."""
        msg = {"role": "user", "content": text}
        self._messages = (*self._messages, msg)
        count_text = getattr(self._counter, "count_text", None)
        if callable(count_text):
            self._tokens += int(count_text(text)) + self._overhead("user")
        else:
            self._tokens += self._counter.count([msg])

    def prompt_tokens(self) -> int:
        """Ước lượng số token prompt hiện tại."""
        return self._tokens

    def reconcile(self, backend_prompt_tokens: int) -> bool:
        """Lấy số token prompt của backend làm chuẩn; trả về True (kèm cảnh báo) khi hai số khác nhau."""
        if backend_prompt_tokens == self._tokens:
            return False
        log.warning("prompt_tokens_mismatch", counted=self._tokens, backend=backend_prompt_tokens)
        self._tokens = backend_prompt_tokens
        return True


class PromptBuilder:
    """Dựng hội thoại đầu tiên của một mẫu và áp dụng thang thu gọn prompt (B1).

    Args:
        system_template: mẫu system prompt, phải chứa đúng một `{tool_docs}`.
        index: chỉ mục catalog để sinh tài liệu tool.
        counter: bộ đếm token.
        max_model_len: độ dài ngữ cảnh tối đa của mô hình.
        max_tokens: số token sinh tối đa mỗi bước, phải chừa chỗ trong ngữ cảnh.
        max_result_chars: số ký tự tối đa của mỗi kết quả tool trong lịch sử.
        held_out_apps: các app bị giấu ở train mode.

    Raises:
        ValueError: khi `system_template` không chứa đúng một `{tool_docs}`.
    """

    def __init__(
        self,
        system_template: str,
        index: CatalogIndex,
        counter: TokenCounter,
        *,
        max_model_len: int,
        max_tokens: int,
        max_result_chars: int,
        held_out_apps: frozenset[str] = frozenset(),
    ) -> None:
        if system_template.count(TOOL_DOCS_SLOT) != 1:
            raise ValueError("system template must contain exactly one {tool_docs}")
        self._rules, self._tail = system_template.split(TOOL_DOCS_SLOT)
        self._index = index
        self._counter = counter
        self._max_model_len = max_model_len
        self._max_tokens = max_tokens
        self._max_result_chars = max_result_chars
        self._held_out = held_out_apps

    @property
    def rules_prefix(self) -> str:
        """Phần system trước `{tool_docs}`; giống hệt cho mọi mẫu."""
        return self._rules

    def allowed_tools(self, sample: Sample, train_mode: bool = False) -> tuple[str, ...]:
        """Các tool mẫu được phép gọi; ở train mode các app held-out bị loại."""
        installed: dict[str, Any] = sample.context.get("installed_apps") or {}
        if train_mode:
            installed = {app: v for app, v in installed.items() if app not in self._held_out}
        return self._index.allowed_tools(installed)

    def _check_leak(self, sample: Sample) -> None:
        """Ném `HeldOutLeak` khi lịch sử của mẫu đã gọi tool thuộc app held-out.

        Mẫu như vậy để lộ app bị giấu qua prompt, nên không được dùng khi huấn luyện.
        """
        for turn in sample.history:
            if turn.role != "assistant":
                continue
            for call in turn.tool_calls:
                app = self._index.app_of(str(call.get("name", "")))
                if app in self._held_out:
                    raise HeldOutLeak(sample.id, app)

    def _build(
        self,
        sample: Sample,
        transcript: Transcript,
        allowed: tuple[str, ...],
        *,
        result_chars: int,
        drop_desc: bool,
        skip_turns: int,
    ) -> Conversation:
        """Dựng hội thoại [system, user] với một mức thu gọn cụ thể.

        Args:
            result_chars: số ký tự tối đa của kết quả tool trong lịch sử.
            drop_desc: True để bỏ mô tả tham số trong tài liệu tool.
            skip_turns: số lượt lịch sử đầu tiên bị bỏ.
        """
        docs = render_tools(self._index, allowed, drop_param_descriptions=drop_desc)
        system = self._rules + docs + self._tail
        parts = [render_context(sample.context)]
        # Cắt `transcript.history` cùng chỉ số với `sample.history` để hai danh sách vẫn khớp nhau.
        history = render_history(sample.history[skip_turns:], transcript.history[skip_turns:], result_chars)
        if history:
            parts.append(history)
        parts.append(f"# Yêu cầu hiện tại\n{transcript.final}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
        return Conversation(messages, self._counter)

    def _fits(self, conv: Conversation) -> bool:
        """True khi prompt cộng `max_tokens` token sinh ra vẫn nằm trong `max_model_len`."""
        return conv.prompt_tokens() + self._max_tokens <= self._max_model_len

    def start(self, sample: Sample, transcript: Transcript, *, train_mode: bool = False) -> tuple[Conversation, int]:
        """Hội thoại đầu tiên của mẫu và mức thu gọn đã dùng (0-3); mức 3 vẫn có thể chưa vừa.

        Raises:
            HeldOutLeak: ở train mode, khi lịch sử gọi tool của app held-out.
        """
        if train_mode:
            self._check_leak(sample)
        allowed = self.allowed_tools(sample, train_mode)
        full, quarter = self._max_result_chars, self._max_result_chars // 4
        conv = self._build(sample, transcript, allowed, result_chars=full, drop_desc=False, skip_turns=0)
        if self._fits(conv):
            return conv, 0
        conv = self._build(sample, transcript, allowed, result_chars=quarter, drop_desc=False, skip_turns=0)
        if self._fits(conv):
            return conv, SHRINK_RESULTS
        conv = self._build(sample, transcript, allowed, result_chars=quarter, drop_desc=True, skip_turns=0)
        if self._fits(conv):
            return conv, SHRINK_PARAM_DESCRIPTIONS
        # Bỏ lịch sử theo từng cặp (user + assistant) từ cũ nhất, giữ lại ít nhất `_MIN_HISTORY_TURNS` lượt.
        # Không vừa sau khi hết lịch sử để bỏ thì vẫn trả về mức 3; Budget sẽ dừng với CONTEXT_OVERFLOW.
        skip = 0
        while len(sample.history) - skip - 2 >= _MIN_HISTORY_TURNS:
            skip += 2
            conv = self._build(sample, transcript, allowed, result_chars=quarter, drop_desc=True, skip_turns=skip)
            if self._fits(conv):
                break
        return conv, SHRINK_HISTORY

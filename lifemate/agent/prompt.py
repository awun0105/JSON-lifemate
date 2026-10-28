"""Dựng prompt cho một mẫu, giữ hội thoại chỉ nối thêm và rút gọn prompt khi quá dài (SPEC 4.6).

Nội dung chính:
- `render_context`, `render_calendar`, `render_history`: sinh các khối văn bản ngữ cảnh, lịch, lịch sử hội thoại.
- `truncate_result`: cắt kết quả tool trước khi đưa vào prompt.
- `HeldOutLeak`: lỗi khi mẫu ở train mode để lộ app held-out qua lịch sử.
- `Conversation`: danh sách tin nhắn chỉ nối thêm, đếm token tăng dần.
- `PromptBuilder`: dựng hội thoại đầu tiên của mẫu và chọn mức rút gọn.

Cách hoạt động:
- System prompt = phần quy tắc trước `{tool_docs}` + mô tả tool của mẫu + phần đuôi. Phần quy tắc giống hệt
  cho mọi mẫu, nên backend dùng lại được prefix cache giữa các mẫu.
- Tin nhắn user đầu tiên gồm `# Ngữ cảnh`, `# Lịch sử hội thoại` (nếu có) và `# Yêu cầu hiện tại`.
- `Conversation` chỉ nối thêm, không sửa tin cũ. Nhờ vậy phần đầu prompt giữ nguyên qua các bước (prefix cache),
  và prompt lúc chạy khớp từng byte với prompt dùng khi sinh dữ liệu train.
- Token được đếm tăng dần: tin assistant lấy số completion token do backend báo, tin user chỉ đếm phần mới.
  Khi số đếm lệch với backend, `reconcile` lấy số của backend làm chuẩn.
- Khi prompt không vừa context, rút gọn theo bốn mức: 0 giữ nguyên; 1 cắt kết quả tool còn 1/4; 2 bỏ thêm mô tả
  tham số; 3 bỏ dần các lượt lịch sử cũ theo từng cặp, giữ ít nhất `_MIN_HISTORY_TURNS` lượt.

Lưu ý:
- Các chuỗi tiếng Việt trong file này là một phần của prompt; sửa chúng là đổi prompt và lệch với dữ liệu train.
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
    """Chuyển kết quả tool thành JSON chuẩn tắc rồi cắt còn tối đa `max_chars` ký tự."""
    return truncate(canonical_json(result), max_chars)


def _value_text(value: Any) -> str:
    """Giữ nguyên giá trị kiểu chuỗi; giá trị kiểu khác đổi sang JSON chuẩn tắc."""
    return value if isinstance(value, str) else canonical_json(value)


_WEEKDAYS = ("thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật")
_CALENDAR_DAYS = 14


def render_calendar(current_time: str) -> list[str]:
    """Liệt kê thứ và ngày của hôm qua, hôm nay và 13 ngày tới, tính từ `current_time`.

    Model hay tính sai "thứ sáu", "chủ nhật này" hay "mai" khi phải tự đếm ngày. Liệt kê sẵn giúp model khỏi phải
    tính. Tuần bắt đầu từ thứ Hai, và mỗi ngày ghi rõ thuộc "tuần này" hay "tuần sau".

    Tham số:
        current_time: thời điểm hiện tại theo ISO 8601.

    Trả về:
        Các dòng văn bản của khối lịch; danh sách rỗng khi không đọc được `current_time`.
    """
    try:
        now = dt.datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return []
    today = now.date()
    # Thứ Hai của tuần hiện tại, làm mốc để tính "tuần này", "tuần sau"...
    week0 = today - dt.timedelta(days=today.weekday())
    names = {-1: "hôm qua", 0: "hôm nay", 1: "mai", 2: "mốt / ngày kia"}
    lines = [f"Lịch (bây giờ là {_WEEKDAYS[today.weekday()]} {today.isoformat()} {now.strftime('%H:%M')}):"]
    for offset in range(-1, _CALENDAR_DAYS):
        day = today + dt.timedelta(days=offset)
        # Phép chia lấy phần nguyên làm tròn xuống, nên ngày trước `week0` cho số tuần âm ("tuần trước").
        weeks = (day - week0).days // 7
        week = {0: "tuần này", 1: "tuần sau", -1: "tuần trước"}.get(weeks, f"{weeks} tuần nữa")
        tag = f", {names[offset]}" if offset in names else ""
        lines.append(f"  {day.isoformat()} {_WEEKDAYS[day.weekday()]} ({week}{tag})")
    return lines


def render_context(context: Mapping[str, Any]) -> str:
    """Sinh khối `# Ngữ cảnh`: thời gian, múi giờ, lịch, vị trí và `user_profile`.

    Khối này không bao giờ chứa `installed_apps`: mô tả tool trong system prompt đã chỉ gồm các tool được phép.

    Tham số:
        context: trường `context` của mẫu.

    Trả về:
        Văn bản của khối, các dòng nối bằng ký tự xuống dòng.
    """
    lines = ["# Ngữ cảnh"]
    for key in ("current_time", "timezone"):
        if key in context:
            lines.append(f"{key}: {_value_text(context[key])}")
    if isinstance(context.get("current_time"), str):
        lines.extend(render_calendar(context["current_time"]))
    loc = context.get("location")
    # Vị trí dạng `{lat, lng}` được in gọn; dạng khác in nguyên JSON.
    if isinstance(loc, Mapping) and "lat" in loc and "lng" in loc:
        lines.append(f"location: {loc['lat']}, {loc['lng']}")
    elif loc is not None:
        lines.append(f"location: {_value_text(loc)}")
    for key, value in (context.get("user_profile") or {}).items():
        lines.append(f"user_profile.{key}: {_value_text(value)}")
    return "\n".join(lines)


def render_history(history: Sequence[Turn], transcript_history: Sequence[str | None], max_result_chars: int) -> str:
    """Sinh khối `# Lịch sử hội thoại`, với kết quả tool đã cắt ngắn.

    Tham số:
        history: các lượt hội thoại trước của mẫu.
        transcript_history: transcript ASR theo cùng chỉ số với `history`; `None` nghĩa là dùng `audio_text`.
        max_result_chars: số ký tự tối đa của mỗi kết quả tool.

    Trả về:
        Văn bản của khối; chuỗi rỗng khi mẫu không có lịch sử.
    """
    if not history:
        return ""
    lines = ["# Lịch sử hội thoại"]
    for i, turn in enumerate(history):
        if turn.role == "user":
            # Ưu tiên transcript (giống điều kiện lúc chạy thật); không có thì dùng văn bản gốc.
            text = transcript_history[i] if i < len(transcript_history) else None
            lines.append(f"Người dùng: {text if text is not None else (turn.audio_text or '')}")
            continue
        lines.append(f"Trợ lý: {turn.text or ''}".rstrip())
        for j, call in enumerate(turn.tool_calls):
            line = f"  [đã gọi] {call.get('name')}({canonical_json(call.get('arguments') or {})})"
            # Lời gọi có thể không có kết quả đi kèm; khi đó chỉ in lời gọi.
            if j < len(turn.tool_results):
                line += f" → {truncate_result(turn.tool_results[j], max_result_chars)}"
            lines.append(line)
    return "\n".join(lines)


class HeldOutLeak(Exception):  # noqa: N818 - name fixed by SPEC 4.5
    """Lỗi khi mẫu ở train mode có lịch sử gọi tool của một app held-out (SPEC 4.5, A4).

    App held-out là app được giấu khi train để đo khả năng tổng quát. Mẫu có lịch sử gọi tool của app đó sẽ làm
    lộ app qua prompt.

    Tham số:
        sample_id: id của mẫu bị lộ.
        app: id app held-out xuất hiện trong lịch sử.

    Thuộc tính:
        sample_id: như tham số cùng tên.
        app: như tham số cùng tên.
    """

    def __init__(self, sample_id: str, app: str) -> None:
        super().__init__(f"{sample_id}: history calls held-out app {app}")
        self.sample_id = sample_id
        self.app = app


class Conversation:
    """Danh sách tin nhắn chỉ nối thêm, kèm bộ đếm token prompt tăng dần.

    Tham số:
        messages: các tin nhắn ban đầu; được sao chép nên tin gốc không bị sửa.
        counter: bộ đếm token theo chat template của model.
    """

    def __init__(self, messages: Sequence[dict[str, Any]], counter: TokenCounter) -> None:
        # Lưu dạng tuple để tin cũ không bị sửa ngoài ý muốn, giữ cho prefix cache còn dùng được.
        self._messages: tuple[dict[str, Any], ...] = tuple(dict(m) for m in messages)
        self._counter = counter
        self._tokens = counter.count(list(self._messages))

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        """Các tin nhắn hiện có, chỉ đọc."""
        return self._messages

    def as_list(self) -> list[dict[str, Any]]:
        """Trả về bản sao các tin nhắn dưới dạng list, đúng kiểu backend LLM nhận."""
        return [dict(m) for m in self._messages]

    def _overhead(self, role: str) -> int:
        """Trả về số token chat template chèn thêm quanh một tin nhắn của `role`.

        Dùng `template_overhead` của bộ đếm nếu có; không có thì đếm một tin nhắn rỗng.
        """
        fn = getattr(self._counter, "template_overhead", None)
        if callable(fn):
            return int(fn(role))
        return self._counter.count([{"role": role, "content": ""}])

    def append_assistant(self, text: str, completion_tokens: int) -> None:
        """Nối nguyên văn đầu ra của LLM vào hội thoại.

        Số token cộng thêm = số completion token + phần template chèn quanh tin assistant.

        Tham số:
            text: văn bản LLM sinh ra.
            completion_tokens: số completion token do backend báo.
        """
        self._messages = (*self._messages, {"role": "assistant", "content": text})
        # Không đếm lại văn bản: số completion token do backend báo đã chính xác.
        self._tokens += completion_tokens + self._overhead("assistant")

    def append_user(self, text: str) -> None:
        """Nối một tin nhắn user (kết quả tool hoặc lỗi từ guard) và chỉ đếm token của tin mới."""
        msg = {"role": "user", "content": text}
        self._messages = (*self._messages, msg)
        # Bộ đếm có `count_text` (như `HFTokenCounter`) thì đếm văn bản cộng phần template; không có thì đếm cả tin.
        count_text = getattr(self._counter, "count_text", None)
        if callable(count_text):
            self._tokens += int(count_text(text)) + self._overhead("user")
        else:
            self._tokens += self._counter.count([msg])

    def prompt_tokens(self) -> int:
        """Trả về số token prompt hiện tại theo cách đếm tăng dần."""
        return self._tokens

    def reconcile(self, backend_prompt_tokens: int) -> bool:
        """Lấy số token prompt của backend làm chuẩn.

        Tham số:
            backend_prompt_tokens: số token prompt do backend báo.

        Trả về:
            `True` khi số đếm khác số của backend (kèm ghi cảnh báo), `False` khi khớp.
        """
        if backend_prompt_tokens == self._tokens:
            return False
        log.warning("prompt_tokens_mismatch", counted=self._tokens, backend=backend_prompt_tokens)
        self._tokens = backend_prompt_tokens
        return True


class PromptBuilder:
    """Dựng hội thoại đầu tiên của một mẫu và rút gọn prompt khi cần (B1).

    Tham số:
        system_template: mẫu system prompt, phải chứa đúng một `{tool_docs}`.
        index: chỉ mục catalog, dùng để sinh mô tả tool.
        counter: bộ đếm token.
        max_model_len: độ dài context tối đa của model, tính bằng token.
        max_tokens: số token sinh tối đa mỗi bước; phải chừa chỗ này trong context.
        max_result_chars: số ký tự tối đa của mỗi kết quả tool trong lịch sử.
        held_out_apps: các app bị giấu ở train mode.

    Lỗi:
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
        """Phần system prompt đứng trước `{tool_docs}`; giống hệt cho mọi mẫu."""
        return self._rules

    def allowed_tools(self, sample: Sample, train_mode: bool = False) -> tuple[str, ...]:
        """Trả về các tool mẫu được phép gọi; ở train mode, tool của app held-out bị loại."""
        installed: dict[str, Any] = sample.context.get("installed_apps") or {}
        if train_mode:
            installed = {app: v for app, v in installed.items() if app not in self._held_out}
        return self._index.allowed_tools(installed)

    def _check_leak(self, sample: Sample) -> None:
        """Báo `HeldOutLeak` khi lịch sử của mẫu đã gọi tool thuộc app held-out.

        Mẫu như vậy làm lộ app bị giấu qua prompt, nên không được dùng khi train.
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
        """Dựng hội thoại `[system, user]` ở một mức rút gọn cụ thể.

        Tham số:
            sample: mẫu cần dựng prompt.
            transcript: transcript của mẫu.
            allowed: các tool được phép.
            result_chars: số ký tự tối đa của kết quả tool trong lịch sử.
            drop_desc: `True` để bỏ mô tả tham số trong mô tả tool.
            skip_turns: số lượt lịch sử đầu tiên bị bỏ.

        Trả về:
            `Conversation` mới, đã đếm token.
        """
        docs = render_tools(self._index, allowed, drop_param_descriptions=drop_desc)
        system = self._rules + docs + self._tail
        parts = [render_context(sample.context)]
        # Cắt `transcript.history` cùng vị trí với `sample.history` để hai danh sách vẫn khớp chỉ số.
        history = render_history(sample.history[skip_turns:], transcript.history[skip_turns:], result_chars)
        if history:
            parts.append(history)
        parts.append(f"# Yêu cầu hiện tại\n{transcript.final}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
        return Conversation(messages, self._counter)

    def _fits(self, conv: Conversation) -> bool:
        """Trả về `True` khi prompt cộng `max_tokens` token sinh ra vẫn nằm trong `max_model_len`."""
        return conv.prompt_tokens() + self._max_tokens <= self._max_model_len

    def start(self, sample: Sample, transcript: Transcript, *, train_mode: bool = False) -> tuple[Conversation, int]:
        """Dựng hội thoại đầu tiên của mẫu, thử lần lượt các mức rút gọn tới khi vừa context.

        Tham số:
            sample: mẫu cần dựng prompt.
            transcript: transcript của mẫu.
            train_mode: `True` khi dựng prompt cho dữ liệu train; khi đó app held-out bị giấu.

        Trả về:
            Cặp (hội thoại, mức rút gọn đã dùng từ 0 tới 3). Ở mức 3, prompt vẫn có thể chưa vừa context.

        Lỗi:
            HeldOutLeak: ở train mode, khi lịch sử đã gọi tool của app held-out.
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
        # Bỏ lịch sử theo từng cặp (user + assistant), bắt đầu từ cặp cũ nhất, giữ lại ít nhất `_MIN_HISTORY_TURNS`
        # lượt.
        # Bỏ hết mức cho phép mà vẫn không vừa thì vẫn trả về mức 3; khi đó `Budget` sẽ dừng với `CONTEXT_OVERFLOW`.
        skip = 0
        while len(sample.history) - skip - 2 >= _MIN_HISTORY_TURNS:
            skip += 2
            conv = self._build(sample, transcript, allowed, result_chars=quarter, drop_desc=True, skip_turns=skip)
            if self._fits(conv):
                break
        return conv, SHRINK_HISTORY

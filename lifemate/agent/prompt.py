"""Prompt assembly, append-only conversation and the shrink ladder (SPEC 4.6)."""

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
    """Canonical JSON of a tool result, cut to `max_chars`."""
    return truncate(canonical_json(result), max_chars)


def _value_text(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value)


_WEEKDAYS = ("thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật")
_CALENDAR_DAYS = 14


def render_calendar(current_time: str) -> list[str]:
    """Weekday and date of yesterday, today and the next 13 days, from `current_time` (ISO 8601).

    The model got "thứ sáu", "chủ nhật này" and "mai" wrong when it had to count days itself; listing
    them removes that arithmetic. Weeks start on Monday, so "tuần này" / "tuần sau" are explicit.
    Empty when `current_time` cannot be parsed.
    """
    try:
        now = dt.datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return []
    today = now.date()
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
    """`# Ngữ cảnh` block: time, timezone, calendar, location, user_profile; never installed_apps."""
    lines = ["# Ngữ cảnh"]
    for key in ("current_time", "timezone"):
        if key in context:
            lines.append(f"{key}: {_value_text(context[key])}")
    if isinstance(context.get("current_time"), str):
        lines.extend(render_calendar(context["current_time"]))
    loc = context.get("location")
    if isinstance(loc, Mapping) and "lat" in loc and "lng" in loc:
        lines.append(f"location: {loc['lat']}, {loc['lng']}")
    elif loc is not None:
        lines.append(f"location: {_value_text(loc)}")
    for key, value in (context.get("user_profile") or {}).items():
        lines.append(f"user_profile.{key}: {_value_text(value)}")
    return "\n".join(lines)


def render_history(history: Sequence[Turn], transcript_history: Sequence[str | None], max_result_chars: int) -> str:
    """`# Lịch sử hội thoại` block, tool results truncated; empty string when there is no history."""
    if not history:
        return ""
    lines = ["# Lịch sử hội thoại"]
    for i, turn in enumerate(history):
        if turn.role == "user":
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
    """A train-mode sample whose history calls a tool of a held-out app (SPEC 4.5, A4)."""

    def __init__(self, sample_id: str, app: str) -> None:
        super().__init__(f"{sample_id}: history calls held-out app {app}")
        self.sample_id = sample_id
        self.app = app


class Conversation:
    """Append-only message list with an incremental prompt-token count."""

    def __init__(self, messages: Sequence[dict[str, Any]], counter: TokenCounter) -> None:
        self._messages: tuple[dict[str, Any], ...] = tuple(dict(m) for m in messages)
        self._counter = counter
        self._tokens = counter.count(list(self._messages))

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        """Messages so far (read-only view)."""
        return self._messages

    def as_list(self) -> list[dict[str, Any]]:
        """Copy of the messages in the list form LLM backends take."""
        return [dict(m) for m in self._messages]

    def _overhead(self, role: str) -> int:
        fn = getattr(self._counter, "template_overhead", None)
        if callable(fn):
            return int(fn(role))
        return self._counter.count([{"role": role, "content": ""}])

    def append_assistant(self, text: str, completion_tokens: int) -> None:
        """Append the exact LLM output; count = completion tokens + template overhead."""
        self._messages = (*self._messages, {"role": "assistant", "content": text})
        self._tokens += completion_tokens + self._overhead("assistant")

    def append_user(self, text: str) -> None:
        """Append a user message (tool result or guard error), counting only the new message."""
        msg = {"role": "user", "content": text}
        self._messages = (*self._messages, msg)
        count_text = getattr(self._counter, "count_text", None)
        if callable(count_text):
            self._tokens += int(count_text(text)) + self._overhead("user")
        else:
            self._tokens += self._counter.count([msg])

    def prompt_tokens(self) -> int:
        """Current prompt-token estimate."""
        return self._tokens

    def reconcile(self, backend_prompt_tokens: int) -> bool:
        """Adopt the backend's prompt-token count; True (and a warning) when it differed."""
        if backend_prompt_tokens == self._tokens:
            return False
        log.warning("prompt_tokens_mismatch", counted=self._tokens, backend=backend_prompt_tokens)
        self._tokens = backend_prompt_tokens
        return True


class PromptBuilder:
    """Builds the first conversation of a sample and applies the shrink ladder (B1)."""

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
        """System text before `{tool_docs}`; identical for every sample."""
        return self._rules

    def allowed_tools(self, sample: Sample, train_mode: bool = False) -> tuple[str, ...]:
        """Tools this sample may call; held-out apps removed in train mode."""
        installed: dict[str, Any] = sample.context.get("installed_apps") or {}
        if train_mode:
            installed = {app: v for app, v in installed.items() if app not in self._held_out}
        return self._index.allowed_tools(installed)

    def _check_leak(self, sample: Sample) -> None:
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
        docs = render_tools(self._index, allowed, drop_param_descriptions=drop_desc)
        system = self._rules + docs + self._tail
        parts = [render_context(sample.context)]
        history = render_history(sample.history[skip_turns:], transcript.history[skip_turns:], result_chars)
        if history:
            parts.append(history)
        parts.append(f"# Yêu cầu hiện tại\n{transcript.final}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
        return Conversation(messages, self._counter)

    def _fits(self, conv: Conversation) -> bool:
        return conv.prompt_tokens() + self._max_tokens <= self._max_model_len

    def start(self, sample: Sample, transcript: Transcript, *, train_mode: bool = False) -> tuple[Conversation, int]:
        """First conversation of a sample and the shrink level used (0-3); level 3 may still not fit."""
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
        skip = 0
        while len(sample.history) - skip - 2 >= _MIN_HISTORY_TURNS:
            skip += 2
            conv = self._build(sample, transcript, allowed, result_chars=quarter, drop_desc=True, skip_turns=skip)
            if self._fits(conv):
                break
        return conv, SHRINK_HISTORY

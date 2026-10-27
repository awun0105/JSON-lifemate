"""Clean ASR text before it reaches the prompt or the SFT data (SPEC 6.1).

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/sanitizer.py@7033c9f.
Whisper invents YouTube-style subtitles on short or silent clips ("hãy subscribe cho kênh...", "cảm ơn các bạn đã
theo dõi") and sometimes loops on numbers or phrases; the lead measured that 98% of the worst ASR errors were of
this kind. Changes from the source:

- The "đăng ký" pattern only fires when it is about a channel ("đăng ký kênh", "đăng ký cho kênh"). The original
  matched "đăng ký <anything>" and cut real requests such as "đăng ký cho anh cái khóa..." or "hủy gói đăng ký".
- `entities_preserved` works on gold tool calls and folds diacritics, like the scorer's fuzzy matching does.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

_CHANNEL = r"(?:kênh|channel)"
HALLUCINATION_PATTERNS = [
    re.compile(
        r"(?i)(?:và\s+|mà\s+|ừ,\s+)?(?:chào\s+quý\s+vị\s+và\s+)?(?:các\s+bạn\s+)?(?:hãy\s+|nhớ\s+)?"
        r"(?:subscribe|đăng\s*(?:ký|kí))\s+(?:cho\s+)?" + _CHANNEL + r".*"
    ),
    re.compile(r"(?i)(?:hãy\s+|nhớ\s+)?subscribe\b.*"),
    re.compile(r"(?i)cảm\s+ơn\s+các\s+bạn\s+đã\s+(?:theo\s+dõi|xem|ủng\s+hộ|đăng\s*(?:ký|kí)).*"),
    re.compile(r"(?i)(?:hẹn\s+gặp\s+lại\s+các\s+bạn\s+)?trong\s+những\s+video\s+(?:tiếp\s+theo|hấp\s+dẫn).*"),
    re.compile(r"(?i)hẹn\s+gặp\s+lại\s+các\s+bạn.*"),
    re.compile(r"(?i)\bla\s*la\s*school\b.*"),
    re.compile(r"(?i)\bghiền\s+mì\s+gõ\b.*"),
    re.compile(r"(?i)like\s+and\s+subscribe.*"),
    re.compile(r"(?i)chúc\s+các\s+bạn\s+(?:một\s+ngày|xem\s+video)\s+vui\s+vẻ.*"),
    re.compile(r"(?i)ủng\s+hộ\s+kênh.*"),
    # the decoding prompt echoed back on a silent turn
    re.compile(r"(?i)^trợ\s+lý\s+ảo\s+tiếng\s+việt.*"),
    re.compile(r"(?i)^lifemate\s*:?\s*trợ\s+lý\s+ảo.*"),
]
_KEYWORDS = (
    "subscribe",
    "kênh",
    "lalaschool",
    "la la school",
    "ghiền mì gõ",
    "video hấp dẫn",
    "video tiếp theo",
    "cảm ơn các bạn đã",
    "hẹn gặp lại các bạn",
    "trợ lý ảo tiếng việt",
    "lifemate",
)


def normalize(text: str) -> str:
    """Unicode NFC and single spaces."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def strip_hallucinations(text: str) -> str:
    """Cut the subtitle-style phrases out of `text`, keeping the real speech around them."""
    cleaned, matched = text, False
    for pattern in HALLUCINATION_PATTERNS:
        if pattern.search(cleaned):
            matched = True
            cleaned = pattern.sub("", cleaned)
    if not matched:
        return text.strip()
    cleaned = re.sub(r"[\s,;]+$", "", cleaned)
    cleaned = re.sub(r"^[\s,;]+", "", cleaned)
    return normalize(cleaned)


def is_pure_hallucination(text: str) -> bool:
    """True when the whole text is subtitle boilerplate (at most one word left after stripping)."""
    norm = normalize(text)
    if not any(k in norm.lower() for k in _KEYWORDS):
        return False
    return len([w for w in re.split(r"\W+", strip_hallucinations(norm)) if w]) <= 1


_DIGIT_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười"}


def _collapse_word(m: re.Match[str]) -> str:
    word = m.group(1)
    repeats = len(re.findall(r"\w+", m.group(0)))
    # Spoken phone numbers repeat digit words legitimately ("không, không không, không"): only a real loop
    # (8+ in a row) is collapsed for digits and digit words. Changed from the source, which cut them at 4.
    if (word.isdigit() or word.lower() in _DIGIT_WORDS) and repeats < 8:
        return m.group(0)
    return word


def truncate_loops(text: str) -> str:
    """Collapse runaway repetitions: one word 4+ times (8+ for digits), a 2-3 word phrase 3+ times, or a long tail
    of numbers."""
    text = re.sub(r"\b(\w+)(?:[\s,]+\1){3,}\b", _collapse_word, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    tokens = text.split()
    if len(tokens) > 20 and sum(1 for t in tokens[10:] if re.fullmatch(r"\d+[,.]?", t)) > 15:
        text = " ".join(tokens[:10])
    return text.strip()


def sanitize(text: str, *, is_history: bool) -> str:
    """Full cleaning of one ASR turn. A history turn that is pure boilerplate becomes "" (the assistant turn after
    it still carries the context); a final turn keeps whatever real speech is left."""
    norm = normalize(text)
    if not norm:
        return ""
    if is_pure_hallucination(norm):
        return "" if is_history else strip_hallucinations(norm)
    return normalize(truncate_loops(strip_hallucinations(norm)))


def _fold(s: str) -> str:
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def gold_entities(tool_calls: Iterable[Mapping[str, Any]]) -> list[str]:
    """Argument values a transcript must contain for the call to be recoverable: strings of 2+ characters that
    are not IDs, and numbers of 4+ digits (phone numbers, amounts)."""
    out: list[str] = []
    for call in tool_calls:
        if call.get("name") == "sys.ask_missing_info":
            continue
        for key, value in (call.get("arguments") or {}).items():
            if key.endswith("_id") or key in {"time", "date", "start", "end", "due"}:
                continue
            if isinstance(value, str) and len(value.strip()) > 1:
                out.append(value)
            elif isinstance(value, int | float) and not isinstance(value, bool) and len(str(int(value))) >= 4:
                out.append(str(int(value)))
    return out


def entities_preserved(tool_calls: Iterable[Mapping[str, Any]], transcript: str) -> bool:
    """True when every gold entity appears in `transcript` (case- and diacritic-insensitive)."""
    hay = _fold(transcript)
    compact = hay.replace(" ", "")
    for ent in gold_entities(tool_calls):
        e = _fold(ent)
        if e not in hay and not (e.replace(" ", "").isdigit() and e.replace(" ", "") in compact):
            return False
    return True

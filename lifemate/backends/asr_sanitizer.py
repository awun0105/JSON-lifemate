"""Làm sạch văn bản ASR trước khi đưa vào prompt hoặc dữ liệu SFT (SPEC 6.1).

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/sanitizer.py@7033c9f.
Chuyển thể từ repo của trưởng nhóm, đường dẫn và commit như dòng trên.

Whisper hay bịa phụ đề kiểu YouTube trên clip ngắn hoặc im lặng ("hãy subscribe cho kênh...", "cảm ơn các bạn đã
theo dõi") và đôi khi lặp vòng trên số hoặc cụm từ; trưởng nhóm đo được 98% lỗi ASR nặng nhất thuộc loại này.

Cách xây dựng: một danh sách regex `HALLUCINATION_PATTERNS` cắt các cụm phụ đề, `_KEYWORDS` cho phép nhận biết
nhanh văn bản chỉ toàn phụ đề, `truncate_loops` gộp các lần lặp. Luồng xử lý trong `sanitize`: chuẩn hóa NFC ->
nếu toàn phụ đề thì trả "" (lượt lịch sử) hoặc phần còn lại (lượt cuối) -> nếu không thì cắt phụ đề rồi gộp lặp.
`gold_entities` và `entities_preserved` dùng khi lọc dữ liệu SFT: kiểm tra bản ghi ASR còn giữ các thực thể của
lời gọi tool gốc hay không.

Thay đổi so với mã gốc:

- Mẫu "đăng ký" chỉ kích hoạt khi nói về kênh ("đăng ký kênh", "đăng ký cho kênh"). Bản gốc khớp
  "đăng ký <bất kỳ>" và cắt mất yêu cầu thật như "đăng ký cho anh cái khóa..." hay "hủy gói đăng ký".
- `entities_preserved` làm việc trên lời gọi tool gốc và bỏ dấu tiếng Việt, giống cách so khớp mờ của bộ chấm điểm.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

# Các mẫu được áp dụng theo thứ tự; mỗi mẫu cắt từ cụm phụ đề đến hết câu (`.*`)
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
    # prompt giải mã bị mô hình lặp lại nguyên văn trên lượt im lặng
    re.compile(r"(?i)^trợ\s+lý\s+ảo\s+tiếng\s+việt.*"),
    re.compile(r"(?i)^lifemate\s*:?\s*trợ\s+lý\s+ảo.*"),
]
# Từ khóa lọc nhanh: văn bản không chứa từ nào ở đây thì chắc chắn không phải phụ đề thuần
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
    """Chuẩn hóa Unicode NFC và gộp khoảng trắng thành một dấu cách."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def strip_hallucinations(text: str) -> str:
    """Cắt các cụm kiểu phụ đề khỏi `text`, giữ lại lời nói thật xung quanh.

    Khi không mẫu nào khớp, văn bản được trả nguyên (chỉ bỏ khoảng trắng hai đầu).
    """
    cleaned, matched = text, False
    for pattern in HALLUCINATION_PATTERNS:
        if pattern.search(cleaned):
            matched = True
            cleaned = pattern.sub("", cleaned)
    if not matched:
        return text.strip()
    # Bỏ dấu phẩy, chấm phẩy và khoảng trắng còn sót ở hai đầu sau khi cắt
    cleaned = re.sub(r"[\s,;]+$", "", cleaned)
    cleaned = re.sub(r"^[\s,;]+", "", cleaned)
    return normalize(cleaned)


def is_pure_hallucination(text: str) -> bool:
    """True khi toàn bộ văn bản là phụ đề rập khuôn (sau khi cắt chỉ còn tối đa một từ)."""
    norm = normalize(text)
    if not any(k in norm.lower() for k in _KEYWORDS):
        return False
    return len([w for w in re.split(r"\W+", strip_hallucinations(norm)) if w]) <= 1


# Các từ đọc chữ số, kể cả biến thể khi đọc số điện thoại ("mốt", "tư", "lăm")
_DIGIT_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười"}


def _collapse_word(m: re.Match[str]) -> str:
    """Hàm thay thế cho regex lặp một từ: trả một lần từ đó, hoặc giữ nguyên chuỗi khi là chữ số lặp chưa đủ 8."""
    word = m.group(1)
    repeats = len(re.findall(r"\w+", m.group(0)))
    # Số điện thoại đọc thành lời lặp từ chữ số là chuyện bình thường ("không, không không, không"): với chữ số và
    # từ chữ số, chỉ gộp khi thật sự lặp vòng (8 lần liên tiếp trở lên). Khác mã gốc, vốn cắt từ 4 lần.
    if (word.isdigit() or word.lower() in _DIGIT_WORDS) and repeats < 8:
        return m.group(0)
    return word


def truncate_loops(text: str) -> str:
    """Gộp các lần lặp mất kiểm soát: một từ lặp 4+ lần (8+ với chữ số), cụm 2-3 từ lặp 3+ lần, hoặc đuôi số dài.

    Đuôi số dài: câu trên 20 token mà phần từ token thứ 11 trở đi có hơn 15 token là số; khi đó chỉ giữ 10 token
    đầu.
    """
    # Thứ tự: một từ, rồi cụm hai từ, rồi cụm ba từ; mỗi bước giữ lại một lần xuất hiện
    text = re.sub(r"\b(\w+)(?:[\s,]+\1){3,}\b", _collapse_word, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    tokens = text.split()
    if len(tokens) > 20 and sum(1 for t in tokens[10:] if re.fullmatch(r"\d+[,.]?", t)) > 15:
        text = " ".join(tokens[:10])
    return text.strip()


def sanitize(text: str, *, is_history: bool) -> str:
    """Làm sạch đầy đủ một lượt ASR.

    Lượt lịch sử chỉ toàn phụ đề thành "" (lượt assistant phía sau vẫn mang ngữ cảnh); lượt cuối giữ lại phần lời
    nói thật còn sót, vì lượt cuối rỗng sẽ khiến agent không có yêu cầu để xử lý.
    """
    norm = normalize(text)
    if not norm:
        return ""
    if is_pure_hallucination(norm):
        return "" if is_history else strip_hallucinations(norm)
    return normalize(truncate_loops(strip_hallucinations(norm)))


def _fold(s: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt (kể cả đ -> d) và gộp khoảng trắng, dùng để so khớp không phân biệt dấu."""
    # NFD tách dấu thành ký tự tổ hợp riêng; "đ" không tách được nên thay tay
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def gold_entities(tool_calls: Iterable[Mapping[str, Any]]) -> list[str]:
    """Các giá trị tham số mà bản ghi phải chứa để lời gọi còn khôi phục được.

    Gồm chuỗi từ 2 ký tự trở lên không phải ID, và số có từ 4 chữ số trở lên (số điện thoại, số tiền). Bỏ qua
    `sys.ask_missing_info`, các khóa kết thúc bằng `_id` và các khóa thời gian (thường được nói khác cách ghi).
    """
    out: list[str] = []
    for call in tool_calls:
        if call.get("name") == "sys.ask_missing_info":
            continue
        for key, value in (call.get("arguments") or {}).items():
            if key.endswith("_id") or key in {"time", "date", "start", "end", "due"}:
                continue
            if isinstance(value, str) and len(value.strip()) > 1:
                out.append(value)
            # bool là lớp con của int nên phải loại riêng
            elif isinstance(value, int | float) and not isinstance(value, bool) and len(str(int(value))) >= 4:
                out.append(str(int(value)))
    return out


def entities_preserved(tool_calls: Iterable[Mapping[str, Any]], transcript: str) -> bool:
    """True khi mọi thực thể gốc xuất hiện trong `transcript` (không phân biệt hoa thường và dấu).

    Thực thể dạng số còn được so trên bản bỏ hết dấu cách, vì ASR có thể tách số điện thoại thành nhiều nhóm.
    """
    hay = _fold(transcript)
    compact = hay.replace(" ", "")
    for ent in gold_entities(tool_calls):
        e = _fold(ent)
        if e not in hay and not (e.replace(" ", "").isdigit() and e.replace(" ", "") in compact):
            return False
    return True

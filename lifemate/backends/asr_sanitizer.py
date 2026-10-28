"""Lọc câu ảo giác và đoạn lặp trong văn bản ASR trước khi đưa vào prompt hoặc dữ liệu SFT (SPEC 6.1).

Nội dung chính:
- `sanitize`: làm sạch đầy đủ một lượt ASR; đây là hàm chính.
- `strip_hallucinations`, `is_pure_hallucination`: cắt câu phụ đề bịa và nhận biết văn bản chỉ toàn phụ đề.
- `truncate_loops`: gộp các đoạn lặp mất kiểm soát.
- `gold_entities`, `entities_preserved`: kiểm tra bản ASR còn giữ các thực thể của lời gọi tool gốc hay không;
  dùng khi lọc dữ liệu SFT.
- `HALLUCINATION_PATTERNS`: danh sách regex của các câu phụ đề hay bị bịa.

Cách hoạt động:
- Whisper hay bịa phụ đề kiểu YouTube trên clip ngắn hoặc im lặng ("hãy subscribe cho kênh...", "cảm ơn các bạn
  đã theo dõi"), và đôi khi lặp vòng trên số hoặc cụm từ. Trưởng nhóm đo được 98% lỗi ASR nặng nhất thuộc loại này.
- `sanitize` chuẩn hoá NFC trước. Nếu văn bản chỉ toàn phụ đề, trả về "" (lượt lịch sử) hoặc phần còn sót (lượt
  cuối). Nếu không, cắt phụ đề rồi gộp đoạn lặp.
- `_KEYWORDS` giúp loại nhanh văn bản chắc chắn không phải phụ đề, khỏi phải chạy regex.

Lưu ý:
- Nguồn: Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/asr/sanitizer.py@7033c9f.
- Khác mã gốc ở hai điểm:
  - Mẫu "đăng ký" chỉ khớp khi nói về kênh ("đăng ký kênh", "đăng ký cho kênh"). Bản gốc khớp "đăng ký <bất kỳ>"
    nên cắt mất yêu cầu thật như "đăng ký cho anh cái khóa..." hay "hủy gói đăng ký".
  - `entities_preserved` làm việc trên lời gọi tool gốc và bỏ dấu tiếng Việt, giống cách so khớp mờ của bộ chấm.
- Bộ đọc số giữ nguyên chữ số lặp dưới 8 lần (mã gốc gộp từ 4 lần), vì số điện thoại đọc thành lời hay lặp số.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

# Các mẫu được áp dụng theo thứ tự; mỗi mẫu cắt từ đầu cụm phụ đề tới hết câu (`.*`).
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
    # Prompt giải mã (`initial_prompt`) mà model lặp lại nguyên văn khi gặp lượt im lặng.
    re.compile(r"(?i)^trợ\s+lý\s+ảo\s+tiếng\s+việt.*"),
    re.compile(r"(?i)^lifemate\s*:?\s*trợ\s+lý\s+ảo.*"),
]
# Từ khoá lọc nhanh: văn bản không chứa từ nào trong đây thì chắc chắn không phải toàn phụ đề.
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
    """Chuẩn hoá Unicode NFC và gộp các khoảng trắng liền nhau thành một dấu cách."""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def strip_hallucinations(text: str) -> str:
    """Cắt các cụm kiểu phụ đề khỏi `text`, giữ lại lời nói thật xung quanh.

    Tham số:
        text: văn bản ASR.

    Trả về:
        Văn bản đã cắt và chuẩn hoá. Khi không mẫu nào khớp, trả về nguyên văn bản, chỉ bỏ khoảng trắng hai đầu.
    """
    cleaned, matched = text, False
    for pattern in HALLUCINATION_PATTERNS:
        if pattern.search(cleaned):
            matched = True
            cleaned = pattern.sub("", cleaned)
    if not matched:
        return text.strip()
    # Sau khi cắt, bỏ dấu phẩy, chấm phẩy và khoảng trắng còn sót ở hai đầu.
    cleaned = re.sub(r"[\s,;]+$", "", cleaned)
    cleaned = re.sub(r"^[\s,;]+", "", cleaned)
    return normalize(cleaned)


def is_pure_hallucination(text: str) -> bool:
    """Trả về `True` khi cả văn bản là phụ đề rập khuôn, tức sau khi cắt chỉ còn nhiều nhất một từ."""
    norm = normalize(text)
    if not any(k in norm.lower() for k in _KEYWORDS):
        return False
    return len([w for w in re.split(r"\W+", strip_hallucinations(norm)) if w]) <= 1


# Các từ đọc chữ số, kể cả biến thể khi đọc số điện thoại ("mốt", "tư", "lăm").
_DIGIT_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười"}


def _collapse_word(m: re.Match[str]) -> str:
    """Hàm thay thế cho regex bắt một từ lặp: trả về từ đó một lần, hoặc giữ nguyên nếu là chữ số lặp dưới 8 lần."""
    word = m.group(1)
    repeats = len(re.findall(r"\w+", m.group(0)))
    # Số điện thoại đọc thành lời thường lặp chữ số ("không, không không, không"). Vì vậy với chữ số và từ đọc chữ số,
    # chỉ gộp khi thật sự lặp vòng (từ 8 lần liên tiếp). Mã gốc gộp từ 4 lần.
    if (word.isdigit() or word.lower() in _DIGIT_WORDS) and repeats < 8:
        return m.group(0)
    return word


def truncate_loops(text: str) -> str:
    """Gộp các đoạn lặp mất kiểm soát trong văn bản ASR.

    Ba trường hợp được gộp:
    - Một từ lặp từ 4 lần trở lên (từ 8 lần với chữ số).
    - Cụm 2-3 từ lặp từ 3 lần trở lên.
    - Đuôi số dài: câu trên 20 token mà phần từ token thứ 11 trở đi có hơn 15 token là số. Khi đó chỉ giữ 10 token
      đầu.

    Tham số:
        text: văn bản cần xử lý.

    Trả về:
        Văn bản đã gộp, bỏ khoảng trắng hai đầu.
    """
    # Gộp theo thứ tự: một từ, rồi cụm hai từ, rồi cụm ba từ; mỗi bước giữ lại một lần xuất hiện.
    # `{3,}` nghĩa là thêm ít nhất 3 lần lặp sau lần đầu (tổng từ 4 lần); `{2,}` là tổng từ 3 lần.
    text = re.sub(r"\b(\w+)(?:[\s,]+\1){3,}\b", _collapse_word, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+\s+\w+\s+\w+)(?:[\s,]+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    tokens = text.split()
    if len(tokens) > 20 and sum(1 for t in tokens[10:] if re.fullmatch(r"\d+[,.]?", t)) > 15:
        text = " ".join(tokens[:10])
    return text.strip()


def sanitize(text: str, *, is_history: bool) -> str:
    """Làm sạch đầy đủ một lượt ASR: cắt phụ đề bịa rồi gộp đoạn lặp.

    Tham số:
        text: văn bản ASR của một lượt.
        is_history: `True` với lượt trong lịch sử, `False` với lượt cuối.

    Trả về:
        Văn bản đã làm sạch. Lượt lịch sử chỉ toàn phụ đề thành "", vì lượt assistant phía sau vẫn giữ ngữ cảnh.
        Lượt cuối thì giữ lại phần lời nói thật còn sót, vì lượt cuối rỗng sẽ khiến agent không có yêu cầu để xử lý.
    """
    norm = normalize(text)
    if not norm:
        return ""
    if is_pure_hallucination(norm):
        return "" if is_history else strip_hallucinations(norm)
    return normalize(truncate_loops(strip_hallucinations(norm)))


def _fold(s: str) -> str:
    """Đổi về chữ thường, bỏ dấu tiếng Việt (cả "đ" thành "d") và gộp khoảng trắng, để so khớp không phân biệt dấu."""
    # NFD tách dấu thành ký tự tổ hợp riêng để bỏ đi; "đ" không tách được nên phải thay tay.
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def gold_entities(tool_calls: Iterable[Mapping[str, Any]]) -> list[str]:
    """Lấy các giá trị tham số mà bản ASR phải chứa thì mới khôi phục được lời gọi tool.

    Gồm chuỗi từ 2 ký tự trở lên không phải ID, và số có từ 4 chữ số trở lên (số điện thoại, số tiền). Bỏ qua
    `sys.ask_missing_info`, các khoá kết thúc bằng `_id` và các khoá thời gian, vì thời gian thường được nói khác
    với cách ghi.

    Tham số:
        tool_calls: các lời gọi tool gốc của mẫu.

    Trả về:
        Danh sách giá trị dạng chuỗi.
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
            # bool là lớp con của int nên phải loại riêng.
            elif isinstance(value, int | float) and not isinstance(value, bool) and len(str(int(value))) >= 4:
                out.append(str(int(value)))
    return out


def entities_preserved(tool_calls: Iterable[Mapping[str, Any]], transcript: str) -> bool:
    """Kiểm tra mọi thực thể gốc có xuất hiện trong `transcript` không, bỏ qua hoa thường và dấu.

    Thực thể dạng số còn được so trên bản đã bỏ hết dấu cách, vì ASR có thể tách số điện thoại thành nhiều nhóm.

    Tham số:
        tool_calls: các lời gọi tool gốc của mẫu.
        transcript: văn bản ASR cần kiểm tra.

    Trả về:
        `True` khi mọi thực thể đều có mặt.
    """
    hay = _fold(transcript)
    compact = hay.replace(" ", "")
    for ent in gold_entities(tool_calls):
        e = _fold(ent)
        if e not in hay and not (e.replace(" ", "").isdigit() and e.replace(" ", "") in compact):
            return False
    return True

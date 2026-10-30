"""Đổi số đọc bằng chữ trong văn bản ASR thành chữ số (benchmark ASR lần 4, thử nghiệm L1).

Nội dung chính:
- `digitize`: hàm chính; đổi các cụm từ đọc số trong một câu thành chữ số, giữ nguyên phần còn lại.
- `parse_number`: đọc một dãy từ số tiếng Việt thành số nguyên, hoặc `None` khi dãy không đúng ngữ pháp số.

Cách hoạt động:
- ChunkFormer RNNT viết mọi số bằng chữ ("năm rưỡi", "ba mươi phần trăm", "không chín không một hai ba bốn năm sáu
  bảy"), trong khi giá trị tham số của lời gọi tool và lời chuẩn của BTC viết chữ số ("5", "30", "0901234567"). Hàm
  này làm bước đổi đó bằng luật, để LLM không phải tự làm.
- Câu được tách thành từ theo khoảng trắng. Các dãy từ số liền nhau được gom lại rồi xử lý theo ba trường hợp:
  1. Dãy chỉ gồm chữ số đọc rời (không có "mươi", "trăm", ...) dài từ 4 từ, hoặc từ 3 từ và bắt đầu bằng "không": ghép
     thành dãy chữ số (số điện thoại, mã).
  2. Dãy từ 2 từ trở lên đọc được thành một số theo ngữ pháp (`parse_number`): thay bằng số đó. Từ đơn vị tiền
     ("nghìn", "triệu", "tỷ") được giữ lại thành chữ khi số chia hết cho đơn vị đó ("năm trăm nghìn" thành "500 nghìn",
     "một triệu hai" thành "1 triệu 2"), vì lời chuẩn viết như vậy.
  3. Dãy một từ: chỉ đổi khi từ đứng sau là đơn vị đo trong `UNITS_AFTER` ("năm giờ" thành "5 giờ", "tám rưỡi" thành
     "8 rưỡi") hoặc từ đứng trước là danh từ đánh số trong `UNITS_BEFORE` ("kênh ba" thành "kênh 3"); "một lít", "hai
     chỗ" giữ nguyên vì lời chuẩn thường viết bằng chữ và LLM không nhầm. Dãy chữ số rời ngắn không đọc được thành một
     số ("sáu tám giờ") được xét từng từ theo cùng luật ("sáu 8 giờ").
- "không" đứng đầu một dãy ngắn hoặc đứng riêng là từ phủ định ("à không năm rưỡi", "à không giờ anh"), không phải
  số 0. Nhiệt độ đọc tắt hai chữ số ("hai sáu độ") thành "26 độ".
- Các từ vừa là số vừa có nghĩa khác được xử lý riêng: "năm" (year) không đổi khi đứng đầu dãy mà từ sau là một chữ số
  đọc rời ("năm hai nghìn hai mươi sáu") hoặc khi dãy chỉ có "năm" và từ sau không phải đơn vị; "tư", "mốt", "lăm" chỉ
  là số khi đứng sau "mươi" hoặc "mười" ("thứ tư" giữ nguyên); "ba" đơn (bố) và "chín" đơn (chín muồi) chỉ đổi trước
  đơn vị.
- Dãy không đọc được theo ngữ pháp ("ba bốn chỗ") giữ nguyên.

Lưu ý:
- Không tính giờ ("bảy giờ kém mười lăm" thành "7 giờ kém 15", không thành 6:45): việc đó cần ngữ cảnh sáng, chiều
  và thuộc về LLM.
- Hàm chỉ xử lý văn bản đã viết thường, không dấu câu (đầu ra RNNT). Với văn bản có dấu câu, từ dính dấu câu không
  được coi là từ số.
"""

from __future__ import annotations

import re

DIGITS = {
    "không": 0,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
}
# Biến thể chỉ dùng sau "mươi" / "mười": "hai mươi mốt", "hai mươi tư", "mười lăm".
TAIL_DIGITS = {"mốt": 1, "tư": 4, "lăm": 5}
MULTIPLIERS = {"mươi": 10, "trăm": 100}
BIG_UNITS = {"nghìn": 1_000, "ngàn": 1_000, "triệu": 1_000_000, "tỷ": 1_000_000_000, "tỉ": 1_000_000_000}
CONNECTORS = {"linh", "lẻ"}
NUMBER_WORDS = DIGITS.keys() | TAIL_DIGITS.keys() | MULTIPLIERS.keys() | BIG_UNITS.keys() | CONNECTORS | {"mười"}

# Từ đứng sau mà một từ số đơn ("năm", "tám") được đổi thành chữ số: đơn vị của giờ, nhiệt độ, phần trăm, tiền, kênh.
UNITS_AFTER = frozenset(
    {"giờ", "h", "rưỡi", "phút", "giây", "độ", "phần", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "kênh"}
)
# Từ đứng trước mà một chữ số đơn được đổi ("kênh ba", "số bốn", "tầng hai"); cho phép chen một mã chữ cái ("kênh vtv
# sáu").
UNITS_BEFORE = frozenset({"kênh", "số", "phòng", "tầng", "lầu", "line", "bàn"})
# Từ đứng sau "năm" cho thấy "năm" là year, không phải số 5.
YEAR_AFTER = frozenset({"nay", "sau", "trước", "ngoái", "kia", "tới", "này", "học", "nhất", "mới", "rồi", "qua"})

_WS = re.compile(r"\s+")


def _parse_tens(words: list[str], i: int) -> tuple[int, int] | None:
    """Đọc hàng chục bắt đầu ở `words[i]` ("mười lăm", "hai mươi mốt"): (giá trị, vị trí sau); (0, i) khi không có hàng
    chục ở đó; `None` khi sai ngữ pháp ("một mươi", "mười không")."""
    n = len(words)
    if i < n and words[i] == "mười":
        value, i = 10, i + 1
    elif i + 1 < n and words[i] in DIGITS and words[i + 1] == "mươi":
        if DIGITS[words[i]] < 2:
            return None
        value, i = DIGITS[words[i]] * 10, i + 2
    else:
        return 0, i
    if i < n and (words[i] in DIGITS or words[i] in TAIL_DIGITS):
        d = DIGITS.get(words[i], TAIL_DIGITS.get(words[i], 0))
        if d == 0:
            return None
        value, i = value + d, i + 1
    return value, i


def _parse_group(words: list[str]) -> int | None:
    """Đọc một nhóm dưới 1 000 ("ba trăm linh năm", "hai mươi mốt", "mười lăm"); `None` khi sai ngữ pháp."""
    value = 0
    i = 0
    n = len(words)
    # Hàng trăm.
    if i + 1 < n and words[i] in DIGITS and words[i + 1] == "trăm":
        value += DIGITS[words[i]] * 100
        i += 2
        if i < n and words[i] in CONNECTORS:
            i += 1
            if i < n and words[i] in DIGITS:
                return value + DIGITS[words[i]] if i + 1 == n else None
            return None
    # Hàng chục.
    tens = _parse_tens(words, i)
    if tens is None:
        return None
    if tens[1] > i:
        value += tens[0]
        i = tens[1]
    elif i < n and words[i] in DIGITS and (i == 0 or value):
        # Một chữ số đứng riêng (đầu nhóm) hoặc sau hàng trăm không có "linh" ("hai trăm năm" là 250: xử lý ở dưới).
        if value and i + 1 == n:
            # "hai trăm năm" đọc tắt của 250; "một trăm hai" là 120.
            return value + DIGITS[words[i]] * 10
        if value:
            return None
        value += DIGITS[words[i]]
        i += 1
    return value if i == n and n > 0 else None


def parse_number(words: list[str]) -> int | None:
    """Đọc một dãy từ số thành số nguyên; `None` khi dãy không theo ngữ pháp số tiếng Việt.

    Tham số:
        words: các từ viết thường, chỉ gồm từ trong `NUMBER_WORDS`.

    Trả về:
        Số nguyên, hoặc `None`.
    """
    total = 0
    group: list[str] = []
    last_unit = 10**12
    for w in words:
        if w in BIG_UNITS:
            unit = BIG_UNITS[w]
            if not group or unit >= last_unit:
                return None
            g = _parse_group(group)
            if g is None:
                return None
            total += g * unit
            last_unit = unit
            group = []
        else:
            group.append(w)
    if group:
        g = _parse_group(group)
        if g is None:
            return None
        if last_unit < 10**12 and len(group) == 1 and group[0] in DIGITS:
            # "một triệu hai" là 1 200 000; "năm trăm nghìn hai" là 500 200: chữ số lẻ sau đơn vị là phần mười của
            # đơn vị.
            total += g * (last_unit // 10)
        else:
            total += g
    return total


def _format(value: int, words: list[str]) -> str:
    """Viết số theo kiểu của lời chuẩn: giữ chữ "nghìn", "triệu", "tỷ" khi số chia hết cho đơn vị đó."""
    units = [w for w in words if w in BIG_UNITS]
    if units:
        word = units[0]
        unit = BIG_UNITS[word]
        if value % unit == 0:
            return f"{value // unit} {word}"
        head, rest = divmod(value, unit)
        if rest % (unit // 10) == 0:
            return f"{head} {word} {rest // (unit // 10)}"
    return str(value)


def _single(w: str, before: list[str], after: str | None) -> str | None:
    """Chữ số cho một từ số đứng riêng, theo từ đứng trước (`before`, các từ đã ra) và từ đứng sau."""
    if w == "mười":
        return "10"
    if w not in DIGITS or w == "không":
        # "không" đứng riêng là phủ định ("à không giờ anh"), không phải số 0.
        return None
    if w == "năm" and after in YEAR_AFTER:
        return None
    if after in UNITS_AFTER:
        return str(DIGITS[w])
    prev = before[-1] if before else ""
    prev2 = before[-2] if len(before) >= 2 else ""
    if prev in UNITS_BEFORE or (prev.isascii() and prev.isalpha() and prev2 in UNITS_BEFORE):
        return str(DIGITS[w])
    return None


def _convert_run(words: list[str], before: list[str], after: str | None) -> str | None:
    """Chữ số cho một dãy từ số, hoặc `None` khi dãy nên giữ nguyên."""
    bare = all(w in DIGITS for w in words)
    if bare and (len(words) >= 4 or (len(words) >= 3 and words[0] == "không")):
        return "".join(str(DIGITS[w]) for w in words)
    if bare and len(words) == 2 and after == "độ" and 1 <= DIGITS[words[0]] <= 3:
        # Nhiệt độ đọc tắt hai chữ số: "hai sáu độ" là 26 độ.
        return "".join(str(DIGITS[w]) for w in words)
    if len(words) == 1:
        return _single(words[0], before, after)
    if words[0] in TAIL_DIGITS or any(w in CONNECTORS for w in words[:1]):
        return None
    value = parse_number(words)
    if value is None:
        return None
    return _format(value, words)


def digitize(text: str) -> str:
    """Đổi các cụm từ đọc số trong `text` thành chữ số; phần còn lại giữ nguyên.

    Tham số:
        text: một câu ASR đã viết thường (đầu ra RNNT); từ dính dấu câu không được coi là từ số.

    Trả về:
        Câu sau khi đổi; khoảng trắng gộp về một dấu cách.
    """
    words = _WS.split(text.strip()) if text.strip() else []
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] not in NUMBER_WORDS:
            out.append(words[i])
            i += 1
            continue
        j = i
        while j < len(words) and words[j] in NUMBER_WORDS:
            j += 1
        run = words[i:j]
        after = words[j] if j < len(words) else None
        # "năm" đứng đầu dãy mà từ sau là chữ số đọc rời ("năm hai nghìn...") là year; "không" đứng đầu dãy ngắn là
        # phủ định ("à không năm rưỡi"): tách từ đó ra khỏi dãy.
        if len(run) >= 2 and (
            (run[0] == "năm" and run[1] in DIGITS) or (run[0] == "không" and len(run) < 3 and run[1] in DIGITS)
        ):
            out.append(run[0])
            i += 1
            continue
        # "tư", "mốt", "lăm", "linh", "lẻ" đứng đầu dãy không phải số ("thứ tư"): giữ từ đó, xét tiếp từ sau.
        if run[0] in TAIL_DIGITS or run[0] in CONNECTORS:
            out.append(run[0])
            i += 1
            continue
        converted = _convert_run(run, out, after)
        if converted is not None:
            out.append(converted)
            i = j
            continue
        if len(run) >= 2:
            # Thử bỏ từ cuối ("hai mươi ba bốn" là "23" + "bốn").
            converted_head = _convert_run(run[:-1], out, run[-1])
            if converted_head is not None:
                out.append(converted_head)
                i = j - 1
                continue
        if all(w in DIGITS for w in run):
            # Dãy chữ số rời ngắn không phải một số ("sáu tám giờ"): xét từng từ theo ngữ cảnh.
            for k, w in enumerate(run):
                nxt = run[k + 1] if k + 1 < len(run) else after
                out.append(_single(w, out, nxt) or w)
            i = j
            continue
        out.extend(run)
        i = j
    return " ".join(out)

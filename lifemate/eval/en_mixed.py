"""Tìm từ tiếng Anh (và từ ngoại khác) chêm trong câu tiếng Việt, xếp theo nhóm (benchmark ASR lần 2).

Nội dung chính:
- `EnWord`: một từ ngoại tìm được, kèm dạng đã gập và nhóm.
- `english_words`: các từ ngoại trong một câu.
- `has_english`: câu có ít nhất một từ ngoại hay không.
- `letter_words`, `letter_tokens`: từ đã gập của một câu (tập / dãy), cách tách giống `english_words`.
- `hits`: từ ngoại nào của đáp án có trong chữ ASR, đếm theo số lần xuất hiện; tuỳ chọn chấp nhận cách viết tương
  đương (`EQUIVALENTS`, tên ghép bị tách đôi).

Cách hoạt động (với từng từ chỉ gồm chữ cái, dài từ 2 ký tự):
1. Từ có dấu tiếng Việt là tiếng Việt, bỏ qua.
2. Từ trong `LOANWORDS` (từ mượn đã Việt hoá, viết theo chính tả tiếng Việt) thuộc nhóm `loanword`.
3. Từ viết hoa toàn bộ (SMS, SJC, VTV) thuộc nhóm `acronym`.
4. Từ không dấu mà không ghép được thành một âm tiết tiếng Việt (phụ âm đầu + nguyên âm + phụ âm cuối hợp lệ), hoặc
   có chữ f/j/w/z, hoặc nằm trong `EN_LIKE_VI`, là từ ngoại. Từ viết hoa chữ đầu không đứng đầu câu (Sarah, MinhMart)
   thuộc nhóm `name`; từ viết thường (check, online) thuộc nhóm `common`. Từ viết hoa đứng đầu câu ("Check…",
   "Sarah…") không phân biệt được, nên xếp `common` khi từ đó có xuất hiện viết thường trong `lowercase_seen`, còn
   lại xếp `name`.

Lưu ý:
- Đây là quy tắc, không phải danh sách rà tay, vì danh sách lấy từ dữ liệu thi không được đưa lên git. Quy tắc bỏ
  sót từ tiếng Anh trông giống âm tiết Việt mà không có trong `EN_LIKE_VI` (ví dụ "man", "ban").
- Chữ số tách từ: "RON95" cho từ "RON".
- Nhóm của từ viết hoa đầu câu phụ thuộc `lowercase_seen`, tức phụ thuộc tập đang chấm; tổng số từ và số từ chép
  đúng thì không phụ thuộc.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

GROUPS = ("common", "acronym", "name", "loanword")

# Từ mượn đã Việt hoá: viết theo chính tả tiếng Việt, mô hình tiếng Việt chép như âm tiết Việt.
LOANWORDS = frozenset({"tivi", "gara"})
# Từ tiếng Anh ghép được thành một âm tiết Việt không dấu, nên quy tắc âm tiết không bắt được.
EN_LIKE_VI = frozenset({"chat", "top", "hot"})
# Cách viết tương đương (đã gập) mà LLM vẫn hiểu đúng; chỉ dùng cho số "chép đúng" nới lỏng (`hits(lenient=True)`).
EQUIVALENTS: dict[str, tuple[str, ...]] = {"tivi": ("tv",), "ok": ("okay", "oke"), "email": ("mail",)}

# Một âm tiết tiếng Việt sau khi bỏ dấu: phụ âm đầu (tuỳ chọn), 1 đến 3 nguyên âm, phụ âm cuối (tuỳ chọn).
_VI_SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$"
)
_FOREIGN_LETTER = re.compile(r"[fjwz]")
# Từ chỉ gồm chữ cái; chữ số và dấu câu là ranh giới từ.
_WORD = re.compile(r"[^\W\d_]+")
# Ranh giới câu: đầu chuỗi, hoặc sau . ! ? … và khoảng trắng.
_SENTENCE_END = re.compile(r"[.!?…]\s*$")


def _fold(s: str) -> str:
    """Đổi về chữ thường, bỏ dấu tiếng Việt, đổi "đ" thành "d"."""
    t = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(c for c in t if not unicodedata.combining(c))


def letter_tokens(text: str) -> list[str]:
    """Tách `text` thành dãy từ chỉ gồm chữ cái, đã gập (chữ thường, bỏ dấu), giữ thứ tự và từ lặp."""
    return _WORD.findall(_fold(unicodedata.normalize("NFC", text)))


def letter_words(text: str) -> set[str]:
    """Tập từ của `letter_tokens`; dùng để so chữ ASR với từ ngoại."""
    return set(letter_tokens(text))


@dataclass(frozen=True, slots=True)
class EnWord:
    """Một từ ngoại trong câu.

    Thuộc tính:
        text: từ như viết trong câu.
        folded: từ đã gập (chữ thường, bỏ dấu), dùng để so khớp.
        group: một tên trong `GROUPS`.
    """

    text: str
    folded: str
    group: str


def english_words(text: str, lowercase_seen: Collection[str] = ()) -> list[EnWord]:
    """Tìm các từ ngoại trong `text`, theo thứ tự xuất hiện (có thể trùng).

    Tham số:
        text: câu tiếng Việt.
        lowercase_seen: các từ (đã gập) từng xuất hiện viết thường ở đâu đó trong kho; dùng để xếp nhóm từ viết hoa
            đứng đầu câu.

    Trả về:
        Danh sách `EnWord`.
    """
    out: list[EnWord] = []
    # Vị trí "đầu câu" phải tính trên đúng chuỗi đã chuẩn hoá mà regex chạy trên đó.
    text = unicodedata.normalize("NFC", text)
    for m in _WORD.finditer(text):
        w = m.group()
        if len(w) < 2:
            continue
        folded = _fold(w)
        if folded != w.lower():  # có dấu tiếng Việt (kể cả "đ")
            continue
        if folded in LOANWORDS:
            out.append(EnWord(w, folded, "loanword"))
        elif w.isupper():
            out.append(EnWord(w, folded, "acronym"))
        elif not _VI_SYLLABLE.match(folded) or _FOREIGN_LETTER.search(folded) or folded in EN_LIKE_VI:
            if w[0].islower():
                group = "common"
            elif m.start() == 0 or _SENTENCE_END.search(text[: m.start()]):
                group = "common" if folded in lowercase_seen else "name"
            else:
                group = "name"
            out.append(EnWord(w, folded, group))
    return out


def has_english(text: str) -> bool:
    """`True` khi `text` có ít nhất một từ ngoại."""
    return bool(english_words(text))


def hits(words: Sequence[EnWord], hyp: str, lenient: bool = False) -> list[bool]:
    """Đánh dấu từng từ ngoại của đáp án có được chép đúng trong `hyp` không, theo số lần xuất hiện.

    Từ xuất hiện k lần trong đáp án chỉ được tính đúng tối đa bằng số lần nó có trong `hyp`; các lần đúng gán cho
    các lần xuất hiện đầu tiên.

    Tham số:
        words: từ ngoại của đáp án (`english_words`).
        hyp: chữ ASR.
        lenient: `True` thì chấp nhận thêm cách viết trong `EQUIVALENTS` và tên ghép bị tách thành 2 hoặc 3 từ liền
            nhau ("Minh Mart" cho "MinhMart").

    Trả về:
        Danh sách `bool`, cùng thứ tự với `words`.
    """
    toks = letter_tokens(hyp)
    avail = Counter(toks)
    if lenient:
        for n in (2, 3):
            avail.update("".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    used: Counter[str] = Counter()
    out: list[bool] = []
    for w in words:
        n_avail = avail[w.folded] + (sum(avail[e] for e in EQUIVALENTS.get(w.folded, ())) if lenient else 0)
        ok = used[w.folded] < n_avail
        used[w.folded] += int(ok)
        out.append(ok)
    return out

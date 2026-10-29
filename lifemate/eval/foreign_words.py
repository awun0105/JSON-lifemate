"""Tìm từ ngoại (chủ yếu là tiếng Anh) chêm trong câu tiếng Việt, xếp theo nhóm (benchmark ASR lần 2).

Dùng ở đâu:
- `lifemate.eval.make_splits`: chọn các mẫu có câu chêm tiếng Anh (`has_foreign_word`) để tạo tập `*_en_mixed.txt`.
- `lifemate.eval.eval_asr`: đếm từ ngoại của đáp án mà ASR chép đúng (`match_words_in_transcript`) và từ của câu mồi
  bị ASR chèn vào (`contains_word`).
- `lifemate.eval.asr_merge`: nhận ra từ tiếng Anh trong chữ ASR khi ghép hai model (`find_foreign_words`).

Nội dung chính:
- `ForeignWord`: một từ ngoại tìm được: từ như viết trong câu, dạng đã chuẩn hoá để so khớp, và nhóm.
- `find_foreign_words`: các từ ngoại trong một câu, theo thứ tự xuất hiện.
- `has_foreign_word`: câu có ít nhất một từ ngoại hay không.
- `fold_text`, `folded_words`, `folded_word_set`: chuẩn hoá chữ (chữ thường, bỏ dấu, "đ" thành "d") và tách câu thành từ
  đã chuẩn hoá, cùng cách tách với `find_foreign_words`.
- `match_words_in_transcript`: từ ngoại nào của đáp án có trong chữ ASR, đếm theo số lần xuất hiện; tuỳ chọn chấp nhận
  cách viết tương đương (`EQUIVALENT_SPELLINGS`) và tên ghép bị tách ("Minh Mart" cho "MinhMart").
- `equivalent_spellings`, `contains_word`: các cách viết tương đương của một từ; một dãy từ có chứa một từ hay không,
  tính cả cách viết tương đương và từ bị tách ("ti vi" cho "tivi").

Cách hoạt động (`find_foreign_words`, với từng từ chỉ gồm chữ cái, dài từ 2 ký tự):
1. Từ có dấu tiếng Việt (kể cả "đ") là tiếng Việt, bỏ qua.
2. Từ trong `VIETNAMIZED_LOANWORDS` (từ mượn viết theo chính tả tiếng Việt: tivi, gara) thuộc nhóm `loanword`.
3. Từ viết hoa toàn bộ (SMS, SJC, VTV) thuộc nhóm `acronym`.
4. Từ không dấu mà không ghép được thành một âm tiết tiếng Việt (`_VIETNAMESE_SYLLABLE`: phụ âm đầu + nguyên âm + phụ
   âm cuối hợp lệ), hoặc có chữ f/j/w/z, hoặc nằm trong `ENGLISH_SPELLED_LIKE_VIETNAMESE` (chat, top, hot), là từ
   ngoại. Từ viết hoa chữ đầu không đứng đầu câu (Sarah, MinhMart) thuộc nhóm `name`; từ viết thường (check, online)
   thuộc nhóm `common`. Từ viết hoa đứng đầu câu ("Check…", "Sarah…") không phân biệt được bằng chữ hoa, nên xếp
   `common` khi từ đó có xuất hiện viết thường trong `lowercase_vocabulary`, còn lại xếp `name`.

Lưu ý:
- Đây là quy tắc chứ không phải danh sách từ, vì phải chạy được trên chữ ASR (câu chưa từng thấy), không chỉ trên đáp
  án. Quy tắc bỏ sót từ tiếng Anh trông giống âm tiết Việt mà không có trong `ENGLISH_SPELLED_LIKE_VIETNAMESE` (ví dụ
  "man", "ban").
- Chữ số tách từ: "RON95" cho từ "RON".
- Nhóm của từ viết hoa đầu câu phụ thuộc `lowercase_vocabulary`, tức phụ thuộc tập đang chấm; tổng số từ và số từ
  chép đúng thì không phụ thuộc.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

WORD_GROUPS = ("common", "acronym", "name", "loanword")

# Từ mượn đã Việt hoá: viết theo chính tả tiếng Việt, mô hình tiếng Việt chép như âm tiết Việt, nên tách thành nhóm
# riêng để không làm đẹp số của từ tiếng Anh thật.
VIETNAMIZED_LOANWORDS = frozenset({"tivi", "gara"})
# Từ tiếng Anh ghép được thành một âm tiết Việt không dấu, nên quy tắc âm tiết không bắt được; liệt kê tay.
ENGLISH_SPELLED_LIKE_VIETNAMESE = frozenset({"chat", "top", "hot"})
# Từ (đã chuẩn hoá) -> các cách viết khác mà LLM vẫn hiểu đúng. Tra theo cả hai chiều (`equivalent_spellings`). Dùng cho số
# "chép đúng" nới lỏng (`match_words_in_transcript(lenient=True)`) và để không tính nhầm một cách viết khác là từ câu
# mồi chèn vào (`contains_word`).
EQUIVALENT_SPELLINGS: dict[str, tuple[str, ...]] = {"tivi": ("tv",), "ok": ("okay", "oke"), "email": ("mail",)}
# Mỗi cách viết -> cả nhóm cách viết của nó, dựng từ `EQUIVALENT_SPELLINGS` ("tv" -> {"tivi", "tv"}).
_SPELLING_FAMILY: dict[str, frozenset[str]] = {
    spelling: frozenset((word, *others))
    for word, others in EQUIVALENT_SPELLINGS.items()
    for spelling in (word, *others)
}

# Một âm tiết tiếng Việt sau khi bỏ dấu: phụ âm đầu (tuỳ chọn), 1 đến 3 nguyên âm, phụ âm cuối (tuỳ chọn).
_VIETNAMESE_SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$"
)
# Chữ cái không có trong bảng chữ cái tiếng Việt.
_NON_VIETNAMESE_LETTER = re.compile(r"[fjwz]")
# Một từ là một dãy chữ cái liền nhau; chữ số và dấu câu là ranh giới từ.
_LETTER_RUN = re.compile(r"[^\W\d_]+")
# Chuỗi kết thúc bằng dấu hết câu (. ! ? …), tức từ đứng ngay sau là từ đầu câu.
_SENTENCE_END = re.compile(r"[.!?…]\s*$")


def fold_text(text: str) -> str:
    """Chuẩn hoá chữ để so khớp: chữ thường, bỏ dấu tiếng Việt, đổi "đ" thành "d"."""
    decomposed = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def folded_words(text: str) -> list[str]:
    """Tách `text` thành dãy từ chỉ gồm chữ cái, đã chuẩn hoá (`fold_text`), giữ thứ tự và từ lặp."""
    return _LETTER_RUN.findall(fold_text(unicodedata.normalize("NFC", text)))


def folded_word_set(text: str) -> set[str]:
    """Tập các từ của `folded_words(text)`."""
    return set(folded_words(text))


@dataclass(frozen=True, slots=True)
class ForeignWord:
    """Một từ ngoại trong câu.

    Thuộc tính:
        text: từ như viết trong câu ("Zalo").
        folded: từ đã chuẩn hoá, dùng để so khớp ("zalo").
        group: một tên trong `WORD_GROUPS`.
    """

    text: str
    folded: str
    group: str


def find_foreign_words(text: str, lowercase_vocabulary: Collection[str] = ()) -> list[ForeignWord]:
    """Tìm các từ ngoại trong `text`, theo thứ tự xuất hiện (có thể trùng).

    Tham số:
        text: câu tiếng Việt.
        lowercase_vocabulary: các từ (đã chuẩn hoá) từng xuất hiện viết thường ở đâu đó trong tập đang chấm; dùng để xếp
            nhóm từ viết hoa đứng đầu câu.

    Trả về:
        Danh sách `ForeignWord`.
    """
    found: list[ForeignWord] = []
    # Vị trí "đầu câu" phải tính trên đúng chuỗi đã chuẩn hoá mà regex chạy trên đó.
    text = unicodedata.normalize("NFC", text)
    for match in _LETTER_RUN.finditer(text):
        word = match.group()
        if len(word) < 2:
            continue
        folded = fold_text(word)
        if folded != word.lower():  # có dấu tiếng Việt (kể cả "đ")
            continue
        if folded in VIETNAMIZED_LOANWORDS:
            found.append(ForeignWord(word, folded, "loanword"))
        elif word.isupper():
            found.append(ForeignWord(word, folded, "acronym"))
        elif (
            not _VIETNAMESE_SYLLABLE.match(folded)
            or _NON_VIETNAMESE_LETTER.search(folded)
            or folded in ENGLISH_SPELLED_LIKE_VIETNAMESE
        ):
            if word[0].islower():
                group = "common"
            elif match.start() == 0 or _SENTENCE_END.search(text[: match.start()]):
                group = "common" if folded in lowercase_vocabulary else "name"
            else:
                group = "name"
            found.append(ForeignWord(word, folded, group))
    return found


def has_foreign_word(text: str) -> bool:
    """`True` khi `text` có ít nhất một từ ngoại."""
    return bool(find_foreign_words(text))


def match_words_in_transcript(
    reference_words: Sequence[ForeignWord], transcript: str, lenient: bool = False
) -> list[bool]:
    """Đánh dấu từng từ ngoại của đáp án có được chép đúng trong `transcript` không, theo số lần xuất hiện.

    Từ xuất hiện k lần trong đáp án chỉ được tính đúng tối đa bằng số lần nó có trong `transcript`; các lần đúng gán
    cho các lần xuất hiện đầu tiên.

    Tham số:
        reference_words: từ ngoại của đáp án (`find_foreign_words`).
        transcript: chữ ASR của cùng câu.
        lenient: `True` thì chấp nhận thêm cách viết trong `EQUIVALENT_SPELLINGS` và tên ghép bị tách thành 2 hoặc 3
            từ liền nhau ("Minh Mart" cho "MinhMart").

    Trả về:
        Danh sách `bool`, cùng thứ tự với `reference_words`.
    """
    transcript_words = folded_words(transcript)
    available = Counter(transcript_words)
    if lenient:
        available.update(_adjacent_joins(transcript_words))
    used: Counter[str] = Counter()
    matched: list[bool] = []
    for word in reference_words:
        spellings = equivalent_spellings(word.folded) if lenient else frozenset({word.folded})
        # Đếm lần dùng theo cả nhóm cách viết, để "TV" và "tivi" cùng câu không dùng chung một lần xuất hiện hai lần.
        family = min(spellings)
        ok = used[family] < sum(available[s] for s in spellings)
        used[family] += int(ok)
        matched.append(ok)
    return matched


def _adjacent_joins(words: Sequence[str]) -> list[str]:
    """Các chuỗi ghép từ 2 hoặc 3 từ liền nhau của `words` (["minh", "mart"] cho "minhmart")."""
    return ["".join(words[i : i + n]) for n in (2, 3) for i in range(len(words) - n + 1)]


def equivalent_spellings(folded: str) -> frozenset[str]:
    """Các cách viết tương đương của một từ đã chuẩn hoá theo `EQUIVALENT_SPELLINGS`, tra cả hai chiều; luôn gồm chính nó."""
    return _SPELLING_FAMILY.get(folded, frozenset({folded}))


def contains_word(folded: str, words: Sequence[str]) -> bool:
    """`True` khi dãy từ đã chuẩn hoá `words` có từ `folded`, tính cả cách viết tương đương và từ bị tách 2 hoặc 3 phần."""
    present = set(words) | set(_adjacent_joins(words))
    return any(s in present for s in equivalent_spellings(folded))

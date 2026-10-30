"""Sửa tên riêng chép lệch trong chữ ASR theo danh sách tên có sẵn của mẫu (benchmark ASR lần 4, thử nghiệm L2).

Nội dung chính:
- `candidates_from_strings`: lập danh sách tên ứng viên từ các chuỗi có sẵn của mẫu.
- `sample_candidates`: danh sách ứng viên của một mẫu tại một lượt: app đã cài, mục catalog của các app đó (tên, giá
  trị `enum`), các trường khác của ngữ cảnh, và lượt trợ lý **trước lượt đó** trong lịch sử (câu trả lời, lời gọi
  tool, kết quả tool như danh bạ, thiết bị).
- `correct`: hàm chính; thay các cụm chữ ASR gần một tên ứng viên bằng tên đó, trả về chữ mới và danh sách thay
  thế.

Cách hoạt động:
- Hai mô hình ASR thường chép tên riêng lệch nhưng gần đúng về âm: "Olivia Broad" cho Olivia Brown, "Harbour View"
  cho Harbor View, "scg" cho SJC. Tên đúng có sẵn trong mẫu lúc thi, nên chỉ cần so.
- Ứng viên: mọi chuỗi giá trị (không lấy khoá JSON) dài từ 1 tới `MAX_WORDS` từ, có ít nhất `MIN_CHARS` chữ cái sau khi
  chuẩn hoá (bỏ dấu, viết thường, bỏ khoảng trắng), hoặc là chữ viết tắt 3 chữ in hoa ("SJC", "VTV"). Chuỗi dài hơn
  (mô tả tool, câu trả lời) và chuỗi dạng tên trường ("vouchers[]", "playlist_id") không là ứng viên; không có từ điển
  tay.
- Với mỗi cửa sổ 1 tới `MAX_WORDS` từ của chữ ASR, ưu tiên cửa sổ dài, tính khoảng cách sửa (số ký tự phải đổi)
  giữa dạng chuẩn hoá của cửa sổ và của từng ứng viên **có cùng số từ**. Thay khi: (1) cửa sổ có ít nhất một từ
  không phải âm tiết tiếng Việt (không có dấu và không theo mẫu âm tiết, hoặc có chữ f, j, w, z), để không đụng câu
  tiếng Việt thường; (2) khoảng cách nhỏ nhất ≤ ngưỡng theo độ dài ứng viên (`Thresholds`); (3) ứng viên gần nhì cách
  xa hơn ít nhất 1; (4) cửa sổ chưa trùng ứng viên nào.
- Cửa sổ đã thay không được xét lại; các cửa sổ xét từ trái sang phải.

Lưu ý:
- Không sửa tên là âm tiết tiếng Việt hợp lệ ("len" cho Lan): không có tín hiệu âm thanh để phân biệt với từ thường.
- Chỉ dùng chữ trong mẫu và đặc tả tool, là thứ bộ thi riêng cũng có; không học gì từ lỗi của train.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lifemate.core.types import Sample
from lifemate.eval.eval_asr import _edit_distance
from lifemate.eval.foreign_words import fold_text

MAX_WORDS = 4
MIN_CHARS = 4

_VIETNAMESE_SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$"
)
_NON_VIETNAMESE_LETTER = re.compile(r"[fjwz]")
_WORD = re.compile(r"[^\W_]+")
# Chuỗi là tên trường hay con trỏ trong schema ("vouchers[]", "playlist_id", "a.b"), không phải tên người dùng nói ra.
_FIELD_LIKE = re.compile(r"[\[\]_./:{}<>=]")


@dataclass(frozen=True, slots=True)
class Candidate:
    """Một tên ứng viên: chữ gốc để chèn vào câu, dạng chuẩn hoá để so, số từ."""

    text: str
    folded: str
    n_words: int


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Khoảng cách sửa tối đa theo số chữ cái của ứng viên: `(độ dài tối thiểu, khoảng cách)` tăng dần."""

    steps: tuple[tuple[int, int], ...] = ((MIN_CHARS, 0), (6, 1), (10, 2))
    acronym: int = 1

    def allow(self, candidate: Candidate) -> int:
        """Khoảng cách tối đa cho một ứng viên: theo số chữ cái; chữ viết tắt 3 chữ in hoa ("SJC") dùng `acronym`."""
        if candidate.text.isupper() and len(candidate.folded) == 3:
            return self.acronym
        allowed = 0
        for min_len, dist in self.steps:
            if len(candidate.folded) >= min_len:
                allowed = dist
        return allowed


@dataclass(frozen=True, slots=True)
class Replacement:
    """Một lần thay: cụm chữ ASR, tên ứng viên, khoảng cách."""

    source: str
    target: str
    distance: int


def _fold_key(text: str) -> str:
    """Dạng so khớp: bỏ dấu, viết thường, chỉ giữ chữ cái và chữ số, bỏ khoảng trắng."""
    return "".join(_WORD.findall(fold_text(text)))


def _is_vietnamese_word(word: str) -> bool:
    """`True` khi `word` (một từ của chữ ASR) là một âm tiết tiếng Việt hợp lệ hoặc có dấu tiếng Việt."""
    folded = fold_text(word)
    if folded != word.lower():
        return True
    return bool(_VIETNAMESE_SYLLABLE.match(folded)) and not _NON_VIETNAMESE_LETTER.search(folded)


def candidates_from_strings(strings: Iterable[str]) -> list[Candidate]:
    """Danh sách ứng viên từ các chuỗi, bỏ trùng theo dạng chuẩn hoá, giữ thứ tự xuất hiện.

    Tham số:
        strings: các chuỗi giá trị có sẵn của mẫu.

    Trả về:
        Các `Candidate` có từ 1 tới `MAX_WORDS` từ và ít nhất `MIN_CHARS` chữ cái.
    """
    seen: set[str] = set()
    out: list[Candidate] = []
    for s in strings:
        text = " ".join(str(s).split())
        words = _WORD.findall(text)
        if not words or len(words) > MAX_WORDS or _FIELD_LIKE.search(text):
            continue
        folded = _fold_key(text)
        acronym = text.isupper() and len(folded) == 3
        if (len(folded) < MIN_CHARS and not acronym) or not any(c.isalpha() for c in folded) or folded in seen:
            continue
        seen.add(folded)
        out.append(Candidate(text, folded, len(words)))
    return out


def _strings(value: Any) -> Iterator[str]:
    """Mọi chuỗi giá trị trong một giá trị JSON lồng nhau; không lấy khoá dict (như `asr_merge._strings`)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def sample_candidates(
    sample: Sample, catalog_apps: Mapping[str, Mapping[str, Any]], turn: str = "final"
) -> list[Candidate]:
    """Ứng viên của `sample` tại lượt `turn` ("final" hoặc chỉ số lượt trong lịch sử).

    Chỉ lấy lượt trợ lý đứng trước `turn`, vì lúc chép lượt đó hệ thống chưa có kết quả tool của các lượt sau.
    """
    context = dict(sample.context)
    installed = context.pop("installed_apps", None) or {}
    texts: list[str] = []
    for app in installed:
        texts.append(str(app).replace("_", " "))
        texts.extend(_strings(catalog_apps.get(app, {})))
    texts.extend(_strings(context))
    limit = len(sample.history) if turn == "final" else int(turn)
    for prior in sample.history[:limit]:
        if prior.role == "assistant":
            texts.extend(_strings([prior.text, prior.tool_calls, prior.tool_results]))
    return candidates_from_strings(texts)


DEFAULT_THRESHOLDS = Thresholds()


def correct(
    text: str, candidates: Sequence[Candidate], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> tuple[str, list[Replacement]]:
    """Thay các cụm chữ ASR gần một tên ứng viên bằng tên đó.

    Tham số:
        text: một câu ASR.
        candidates: ứng viên của mẫu tại lượt đó (`sample_candidates`).
        thresholds: ngưỡng khoảng cách theo độ dài.

    Trả về:
        (câu sau khi thay, các lần thay theo thứ tự).
    """
    words = text.split()
    if not words or not candidates:
        return text, []
    by_key = {c.folded: c for c in candidates}
    out: list[str] = []
    replacements: list[Replacement] = []
    i = 0
    while i < len(words):
        chosen: tuple[int, Candidate, int] | None = None
        for n in range(min(MAX_WORDS, len(words) - i), 0, -1):
            window = words[i : i + n]
            if not any(not _is_vietnamese_word(w) for w in window):
                continue
            key = _fold_key(" ".join(window))
            if not key or key in by_key:
                break
            best: tuple[int, Candidate] | None = None
            second = 10**9
            for c in candidates:
                # Chỉ so cửa sổ cùng số từ với ứng viên, để không nuốt từ đứng cạnh ("robot hút bụi cả").
                if c.n_words != n or abs(len(c.folded) - len(key)) > thresholds.allow(c):
                    continue
                d = _edit_distance(key, c.folded)
                if best is None or d < best[0]:
                    second = best[0] if best is not None else second
                    best = (d, c)
                elif d < second:
                    second = d
            if best is not None and best[0] <= thresholds.allow(best[1]) and second >= best[0] + 1:
                chosen = (n, best[1], best[0])
                break
        if chosen is None:
            out.append(words[i])
            i += 1
            continue
        n, cand, dist = chosen
        replacements.append(Replacement(" ".join(words[i : i + n]), cand.text, dist))
        out.append(cand.text)
        i += n
    return " ".join(out), replacements

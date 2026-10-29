"""Chấm điểm hai phương án của mỗi vùng khác nhau bằng âm thanh (benchmark ASR lần 3, thử nghiệm G).

Nội dung chính:
- `scoring_text`: đưa một câu về kiểu viết dùng để chấm (chữ thường, bỏ dấu câu).
- `candidate_pairs`: mỗi vùng cần quyết định của một câu cho một cặp câu ứng viên A (giữ chữ RNNT) và B (lấy chữ
  large-v3 ở vùng đó).
- `decide_g`: quyết định của G, lấy B khi điểm âm thanh của B cao hơn A.

Cách hoạt động:
- Vùng cần quyết định là vùng mà chữ large-v3 có từ tiếng Anh mới (`asr_merge.candidate_regions`), giống tập vùng mà
  các cách quyết định khác (C0, F, H) xét.
- Hai câu ứng viên của một vùng chỉ khác nhau ở đúng vùng đó; các vùng khác giữ chữ RNNT.
- Cả hai câu cùng kiểu viết, vì large-v3 quen viết hoa và có dấu câu: nếu chỉ câu B giữ kiểu viết đó thì B được lợi
  chỉ vì kiểu viết.
- Điểm âm thanh do `FasterWhisperASR.score_texts` tính trên GPU (phiên K3); module này không chạy mô hình.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from lifemate.eval.asr_merge import Region, assemble, candidate_regions, split_regions

_WORD = re.compile(r"[^\W_]+")


def scoring_text(text: str) -> str:
    """Chữ thường, NFC, bỏ dấu câu, các từ cách nhau một dấu cách."""
    return " ".join(_WORD.findall(unicodedata.normalize("NFC", text.lower())))


@dataclass(frozen=True)
class CandidatePair:
    """Một vùng cần quyết định và hai câu ứng viên của vùng đó.

    Thuộc tính:
        part_index: chỉ số của vùng trong `split_regions(primary, secondary)`.
        region: vùng khác nhau.
        text_a: câu giữ chữ RNNT ở mọi vùng, đã đưa về kiểu viết chấm điểm.
        text_b: câu lấy chữ large-v3 ở đúng vùng này, cùng kiểu viết.
    """

    part_index: int
    region: Region
    text_a: str
    text_b: str


def candidate_pairs(primary: str, secondary: str) -> list[CandidatePair]:
    """Các cặp câu ứng viên của một câu nói, theo thứ tự vùng từ trái sang phải.

    Tham số:
        primary: chữ RNNT.
        secondary: chữ large-v3 (đã qua bộ lọc câu bịa).
    """
    parts = split_regions(primary, secondary)
    text_a = scoring_text(assemble(parts, lambda _k, _r: False))
    out = []
    for k in candidate_regions(parts):
        region = parts[k]
        assert isinstance(region, Region)
        text_b = scoring_text(assemble(parts, lambda i, _r, k=k: i == k))
        out.append(CandidatePair(k, region, text_a, text_b))
    return out


def decide_g(score_a: float, score_b: float) -> bool:
    """Quyết định của G: lấy chữ large-v3 khi điểm âm thanh của câu B cao hơn câu A (hoà thì giữ RNNT)."""
    return score_b > score_a

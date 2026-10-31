"""Ghép chữ hai mô hình ASR, bản 2 (benchmark ASR lần 4 mục 6c, D66, D67): đổi số trước khi gióng, tách thêm vùng
chỉ khác dấu thanh, và quyết định cả vùng chữ Việt.

Nội dung chính:
- `prepare`: đưa một câu ASR về dạng dùng để gióng và ghép: chữ thường, bỏ dấu câu dính hai đầu từ, đổi số đọc thành
  chữ số (`asr_numbers.digitize`).
- `split_regions_v2`: như `asr_merge.split_regions`, nhưng hai từ chỉ khác dấu thanh ("tắt"/"tất") không được coi là
  khớp, nên thành vùng khác nhau.
- `region_kind`: `"en"` khi phía large-v3 có từ tiếng Anh mới (vùng của H-Anh), `"vi"` cho các vùng còn lại (H-Việt).
- `candidate_pairs_v2`: câu ứng viên A (giữ RNNT ở mọi vùng) và B_k (lấy large-v3 ở đúng vùng k) cho mọi vùng, ở dạng
  chấm điểm; dùng cho phiên GPU chấm điểm âm thanh (K5) và cho engine `MergedASR` bản 2.
- `vi_features`, `VI_FEATURES`: tín hiệu của H-Việt.

Cách hoạt động:
- Cả hai bản được `prepare` trước khi gióng, nên "năm giờ" (RNNT) và "5 giờ" (large-v3) khớp nhau thay vì thành một
  vùng giả; câu ghép không cần đổi số ở cuối nữa.
- Gióng giữ nguyên cách của bản 1 (Levenshtein theo từ trên dạng bỏ dấu, `asr_merge.align_words`), rồi các cặp được
  ghép với nhau mà khác dấu thanh bị coi là không khớp; như vậy ranh giới vùng ổn định như bản 1 và chỉ thêm vùng.
- Bản 1 (`asr_merge`, `asr_v3_analysis`) không đổi, để các số của lần 3 và `configs/asr_merge/h_v1.json` vẫn tái lập.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lifemate.backends.asr_numbers import digitize
from lifemate.eval.asr_merge import Region, align_words, assemble
from lifemate.eval.eval_asr import _edit_distance
from lifemate.eval.foreign_words import find_foreign_words, fold_text

_EDGE = re.compile(r"^[^\w]+|[^\w]+$")
_WORD = re.compile(r"[^\W_]+")


def prepare(text: str) -> str:
    """Chữ thường (NFC), bỏ dấu câu dính hai đầu mỗi từ, gộp khoảng trắng, rồi đổi số đọc thành chữ số."""
    words = [_EDGE.sub("", w) for w in unicodedata.normalize("NFC", text.lower()).split()]
    return digitize(" ".join(w for w in words if w))


def _fold_key(word: str) -> str:
    """Khoá so khớp bỏ dấu (như bản 1)."""
    return fold_text(_EDGE.sub("", word))


def _tone_key(word: str) -> str:
    """Khoá so khớp giữ dấu thanh: chữ thường, NFC, bỏ dấu câu hai đầu."""
    return unicodedata.normalize("NFC", _EDGE.sub("", word).lower())


def split_regions_v2(primary: str, secondary: str) -> list[str | Region]:
    """Đặt hai câu (đã `prepare`) cạnh nhau; từ khớp phải giống cả dấu thanh. Trả về như `asr_merge.split_regions`."""
    pw, sw = primary.split(), secondary.split()
    if not sw:
        return list(pw)
    pk, sk = [_fold_key(w) for w in pw], [_fold_key(w) for w in sw]
    out: list[str | Region] = []
    pr: list[str] = []
    sr: list[str] = []

    def close() -> None:
        if not pr and not sr:
            return
        new_english = {w.folded for w in find_foreign_words(" ".join(sr))} - {
            w.folded for w in find_foreign_words(" ".join(pr))
        }
        out.append(Region(tuple(pr), tuple(sr), frozenset(new_english)))
        pr.clear()
        sr.clear()

    for i, j in align_words(pk, sk):
        if i is not None and j is not None and pk[i] == sk[j] and _tone_key(pw[i]) == _tone_key(sw[j]):
            close()
            out.append(pw[i])
            continue
        if i is not None:
            pr.append(pw[i])
        if j is not None:
            sr.append(sw[j])
    close()
    return out


def region_kind(region: Region) -> str:
    """`"en"` khi phía large-v3 có từ tiếng Anh mới (như `asr_merge.candidate_regions`), còn lại `"vi"`."""
    return "en" if region.secondary and region.new_english else "vi"


def region_indices(parts: Sequence[str | Region]) -> dict[str, list[int]]:
    """Chỉ số các vùng theo loại: `{"en": [...], "vi": [...]}`."""
    out: dict[str, list[int]] = {"en": [], "vi": []}
    for k, p in enumerate(parts):
        if isinstance(p, Region):
            out[region_kind(p)].append(k)
    return out


def scoring_text(text: str) -> str:
    """Kiểu viết dùng để chấm điểm âm thanh: chữ thường, NFC, chỉ giữ chữ và chữ số."""
    return " ".join(_WORD.findall(unicodedata.normalize("NFC", text.lower())))


@dataclass(frozen=True)
class Pair:
    """Một vùng và hai câu ứng viên ở dạng chấm điểm (giống `candidate_scoring.CandidatePair`), kèm loại vùng."""

    part_index: int
    kind: str
    text_a: str
    text_b: str


def candidate_pairs_v2(parts: Sequence[str | Region]) -> list[Pair]:
    """Cặp câu ứng viên cho mọi vùng của một câu (đã tách bằng `split_regions_v2`), theo thứ tự vùng."""
    text_a = scoring_text(assemble(parts, lambda _k, _r: False))
    out = []
    for k, p in enumerate(parts):
        if isinstance(p, Region):
            text_b = scoring_text(assemble(parts, lambda i, _r, k=k: i == k))
            out.append(Pair(k, region_kind(p), text_a, text_b))
    return out


# Tín hiệu của H-Việt (asr_v4.md mục 6c), theo thứ tự trong vector.
VI_FEATURES = (
    "delta_acoustic",
    "delta_tokens",
    "n_primary",
    "n_secondary",
    "char_distance",
    "is_insert",
    "is_delete",
    "at_end",
    "in_context",
    "avg_logprob",
    "is_final",
)


def vi_features(
    parts: Sequence[str | Region],
    k: int,
    acoustic: tuple[float, float, int, int, bool] | None,
    context: frozenset[str],
    v3_scores: Mapping[str, float],
    is_final: bool,
) -> list[float]:
    """Vector tín hiệu `VI_FEATURES` của vùng `parts[k]`.

    Tham số:
        acoustic: (điểm A, điểm B, số token A, số token B, câu dài hơn 30 giây) hoặc `None` khi chưa chấm; câu dài
            hoặc chưa chấm thì hai tín hiệu âm thanh bằng 0.
        context: từ điển ngữ cảnh của mẫu (`asr_merge.sample_context_words`, đã bỏ dấu).
        v3_scores: điểm tin cậy của large-v3 cho cả câu.
    """
    region = parts[k]
    assert isinstance(region, Region)
    a, b = " ".join(region.primary), " ".join(region.secondary)
    fa, fb = fold_text(a).replace(" ", ""), fold_text(b).replace(" ", "")
    char_dist = _edit_distance(fa, fb) / max(1, len(fa), len(fb))
    if acoustic is None or acoustic[4]:
        d_ac, d_tok = 0.0, 0.0
    else:
        d_ac, d_tok = acoustic[1] - acoustic[0], float(acoustic[3] - acoustic[2])
    b_words = {fold_text(w) for w in region.secondary}
    return [
        d_ac,
        d_tok,
        float(len(region.primary)),
        float(len(region.secondary)),
        char_dist,
        float(not region.primary),
        float(not region.secondary),
        float(k == len(parts) - 1),
        float(bool(b_words & context)),
        float(v3_scores.get("avg_logprob", 0.0)),
        float(is_final),
    ]

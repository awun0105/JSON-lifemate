"""Ghép chữ của hai ASR: giữ chữ của model chính, chỉ lấy từ model phụ các đoạn có từ tiếng Anh mới.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.asr_merge --primary <rnnt>.hyps.jsonl --secondary <whisper>.hyps.jsonl --out merged.hyps.jsonl

Nội dung chính:
- `align_words`: gióng hàng hai dãy từ theo khoảng cách Levenshtein (số từ phải thêm, xoá, thay), có truy vết.
- `merge_transcripts`: ghép hai câu theo quy tắc ở dưới.
- `merge_hyp_rows`: ghép hai danh sách dòng `hyps.jsonl` (mỗi dòng là chữ ASR của một câu) theo từng câu.
- `main`: CLI ghép hai file `hyps.jsonl`; file ra chấm được bằng `lifemate.eval.rescore_asr`.

Cách hoạt động (`merge_transcripts`):
1. Tách hai câu thành từ (giữ nguyên cách viết để ghép lại), gióng hàng trên dạng đã gập (chữ thường, bỏ dấu, bỏ dấu
   câu).
2. Gom các cặp không khớp nằm giữa hai từ khớp thành một "vùng khác nhau".
3. Ở mỗi vùng: nếu phía model phụ có từ tiếng Anh (`foreign_words.find_foreign_words`) mà phía model chính không
   có, và vùng phía phụ không dài quá `2 x (số từ phía chính) + 2` (chặn câu bịa), thì lấy chữ của model phụ cho cả
   vùng; còn lại giữ model chính.
4. Câu của model phụ bị bỏ hẳn khi chữ trước lọc khác chữ sau lọc (bộ lọc ảo giác đã cắt) hoặc rỗng.

Lưu ý:
- Model chính là model có WER thấp và không bịa câu (ChunkFormer RNNT); model phụ là model chép từ tiếng Anh tốt hơn
  (whisper-large-v3 hoặc PhoWhisper). Quy tắc chỉ sửa từ tiếng Anh, không sửa phần tiếng Việt.
- Tất định: không có ngẫu nhiên, hoà khi truy vết ưu tiên thay thế rồi xoá rồi chèn.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lifemate.eval.foreign_words import find_foreign_words, fold_text

# Dấu câu dính ở hai đầu một từ ("ví," hay "(Zalo)").
_EDGE_PUNCTUATION = re.compile(r"^[^\w]+|[^\w]+$")


def _match_key(word: str) -> str:
    """Dạng so khớp của một từ: bỏ dấu câu hai đầu rồi gập chữ (chữ thường, bỏ dấu tiếng Việt)."""
    return fold_text(_EDGE_PUNCTUATION.sub("", word))


def align_words(a: Sequence[str], b: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """Gióng hàng hai dãy khoá từ; trả về các cặp (chỉ số trong `a` hoặc None, chỉ số trong `b` hoặc None).

    Cặp có cả hai chỉ số là hai từ đứng cùng chỗ (khớp hoặc bị thay); `None` ở một bên là từ chỉ có ở bên kia.
    """
    n, m = len(a), len(b)
    # cost[i][j]: số từ ít nhất phải thêm, xoá hoặc thay để biến `a[:i]` thành `b[:j]`.
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def merge_transcripts(primary: str, secondary: str) -> str:
    """Ghép câu của model chính với câu của model phụ theo quy tắc ở docstring module."""
    primary_words, secondary_words = primary.split(), secondary.split()
    if not secondary_words:
        return primary
    primary_keys = [_match_key(w) for w in primary_words]
    secondary_keys = [_match_key(w) for w in secondary_words]
    merged: list[str] = []
    # Vùng khác nhau đang gom: các từ của mỗi bên nằm giữa hai từ khớp.
    primary_region: list[str] = []
    secondary_region: list[str] = []

    def close_region() -> None:
        """Quyết định lấy bên nào cho vùng khác nhau đang gom, ghi vào `merged`, rồi xoá vùng."""
        secondary_text, primary_text = " ".join(secondary_region), " ".join(primary_region)
        # Phía phụ mang tới từ tiếng Anh mà phía chính không có. Không đòi phía chính "không có từ ngoại", vì chữ rác
        # của model chính ("s tp") cũng không ghép được âm tiết Việt nên bị coi là từ ngoại.
        new_english = {w.folded for w in find_foreign_words(secondary_text)} - {
            w.folded for w in find_foreign_words(primary_text)
        }
        not_too_long = len(secondary_region) <= 2 * len(primary_region) + 2
        take_secondary = bool(secondary_region) and bool(new_english) and not_too_long
        merged.extend(secondary_region if take_secondary else primary_region)
        primary_region.clear()
        secondary_region.clear()

    for i, j in align_words(primary_keys, secondary_keys):
        if i is not None and j is not None and primary_keys[i] == secondary_keys[j]:
            close_region()
            merged.append(primary_words[i])
            continue
        if i is not None:
            primary_region.append(primary_words[i])
        if j is not None:
            secondary_region.append(secondary_words[j])
    close_region()
    return " ".join(merged)


def merge_hyp_rows(primary: Sequence[dict[str, Any]], secondary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ghép hai danh sách dòng `hyps.jsonl` theo (id, turn); câu không có ở model phụ thì giữ model chính.

    Mỗi dòng là `{id, turn, ref, hyp, raw}`: id mẫu, lượt ("final" hoặc chỉ số lượt trong lịch sử), đáp án, chữ ASR
    sau và trước bộ lọc ảo giác. Câu của model phụ bị bỏ khi rỗng hoặc khi bộ lọc ảo giác đã cắt (`raw` khác `hyp` sau
    khi gộp khoảng trắng).
    """
    secondary_by_turn = {(str(r["id"]), str(r["turn"])): r for r in secondary}
    merged_rows = []
    for row in primary:
        other = secondary_by_turn.get((str(row["id"]), str(row["turn"])))
        hyp = str(row["hyp"])
        if other is not None:
            other_hyp = str(other["hyp"])
            filtered = " ".join(str(other.get("raw", other_hyp)).split()) != " ".join(other_hyp.split())
            if other_hyp.strip() and not filtered:
                hyp = merge_transcripts(hyp, other_hyp)
        merged_rows.append({**row, "hyp": hyp, "raw": hyp})
    return merged_rows


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI ghép hai file `hyps.jsonl`; trả về mã thoát 0."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_merge", description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--secondary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    def load(path: Path) -> list[dict[str, Any]]:
        """Đọc một file `hyps.jsonl`."""
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    rows = merge_hyp_rows(load(args.primary), load(args.secondary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"merged {len(rows)} utterances -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

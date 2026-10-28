"""Ghép chữ của hai ASR: giữ chữ của model chính, chỉ lấy từ model phụ các đoạn có từ tiếng Anh mới.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.asr_merge --primary <rnnt>.hyps.jsonl --secondary <whisper>.hyps.jsonl --out merged.hyps.jsonl

Nội dung chính:
- `align`: gióng hàng hai dãy từ theo khoảng cách Levenshtein, có truy vết.
- `merge`: ghép hai câu theo quy tắc ở dưới.
- `main`: CLI ghép hai file `hyps.jsonl` theo từng câu; file ra chấm được bằng `lifemate.eval.rescore_asr`.

Cách hoạt động (`merge`):
1. Tách hai câu thành từ (giữ nguyên cách viết để ghép lại), gióng hàng trên dạng đã gập (chữ thường, bỏ dấu, bỏ dấu
   câu).
2. Gom các cặp không khớp nằm giữa hai từ khớp thành một "vùng khác nhau".
3. Ở mỗi vùng: nếu phía model phụ có từ tiếng Anh (`en_mixed.english_words`) mà phía model chính không có, và vùng phía
   phụ không dài quá `2 x (số từ phía chính) + 2` (chặn câu bịa), thì lấy chữ của model phụ cho cả vùng; còn lại giữ
   model chính.
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
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lifemate.eval.en_mixed import english_words

_PUNCT_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


def _key(word: str) -> str:
    """Dạng so khớp của một từ: bỏ dấu câu hai đầu, chữ thường, bỏ dấu tiếng Việt."""
    t = unicodedata.normalize("NFD", _PUNCT_EDGE.sub("", word).lower()).replace("đ", "d")
    return "".join(c for c in t if not unicodedata.combining(c))


def align(a: Sequence[str], b: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """Gióng hàng hai dãy khoá từ; trả về các cặp (chỉ số trong `a` hoặc None, chỉ số trong `b` hoặc None)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def merge(primary: str, secondary: str) -> str:
    """Ghép câu của model chính với câu của model phụ theo quy tắc ở docstring module."""
    pw, sw = primary.split(), secondary.split()
    if not sw:
        return primary
    pk, sk = [_key(w) for w in pw], [_key(w) for w in sw]
    out: list[str] = []
    region_p: list[str] = []
    region_s: list[str] = []

    def flush() -> None:
        """Quyết định lấy phía nào cho vùng khác nhau đang gom, rồi xoá vùng."""
        s_text, p_text = " ".join(region_s), " ".join(region_p)
        # Phía phụ mang tới từ tiếng Anh mà phía chính không có. Không đòi phía chính "không có từ ngoại", vì chữ rác
        # của model chính ("s tp") cũng không ghép được âm tiết Việt nên bị coi là từ ngoại.
        new_en = {w.folded for w in english_words(s_text)} - {w.folded for w in english_words(p_text)}
        take = bool(region_s) and bool(new_en) and len(region_s) <= 2 * len(region_p) + 2
        out.extend(region_s if take else region_p)
        region_p.clear()
        region_s.clear()

    for i, j in align(pk, sk):
        if i is not None and j is not None and pk[i] == sk[j]:
            flush()
            out.append(pw[i])
            continue
        if i is not None:
            region_p.append(pw[i])
        if j is not None:
            region_s.append(sw[j])
    flush()
    return " ".join(out)


def merge_rows(primary: Sequence[dict[str, Any]], secondary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ghép hai danh sách dòng `hyps.jsonl` theo (id, turn); câu không có ở model phụ thì giữ model chính.

    Câu của model phụ bị bỏ khi rỗng hoặc khi bộ lọc ảo giác đã cắt (`raw` khác `hyp` sau khi gộp khoảng trắng).
    """
    sec = {(str(r["id"]), str(r["turn"])): r for r in secondary}
    out = []
    for r in primary:
        s = sec.get((str(r["id"]), str(r["turn"])))
        hyp = str(r["hyp"])
        if s is not None:
            s_hyp = str(s["hyp"])
            filtered = " ".join(str(s.get("raw", s_hyp)).split()) != " ".join(s_hyp.split())
            if s_hyp.strip() and not filtered:
                hyp = merge(hyp, s_hyp)
        out.append({**r, "hyp": hyp, "raw": hyp})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI ghép hai file `hyps.jsonl`; trả về mã thoát 0."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_merge", description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--secondary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    def load(p: Path) -> list[dict[str, Any]]:
        """Đọc một file `hyps.jsonl`."""
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    rows = merge_rows(load(args.primary), load(args.secondary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"merged {len(rows)} utterances -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

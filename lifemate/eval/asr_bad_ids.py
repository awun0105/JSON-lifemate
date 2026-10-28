"""Lấy id các mẫu có câu ASR hỏng từ một file `hyps.jsonl` của `eval_asr`, để chạy lại riêng các mẫu đó.

Cách dùng:
  python -m lifemate.eval.asr_bad_ids <hyps.jsonl> --out bad_ids.txt [--kind unk|empty]

Nội dung chính:
- `bad_ids`: id (không trùng, theo thứ tự xuất hiện) của các mẫu có ít nhất một câu hỏng.
- `main`: CLI ghi danh sách id, mỗi dòng một id.

Lưu ý:
- `unk`: chữ ASR chỉ gồm token không biết (`⁇`), dấu hiệu lỗi model hoặc engine.
- `empty`: đáp án có chữ nhưng ASR ra rỗng sau chuẩn hoá (gồm cả trường hợp `unk`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from lifemate.eval.eval_asr import is_unk_only, normalize_text

KINDS = ("unk", "empty")


def bad_ids(rows: Iterable[dict[str, Any]], kind: str = "unk") -> list[str]:
    """Trả về id các mẫu có ít nhất một câu hỏng theo `kind`.

    Tham số:
        rows: các dòng của `hyps.jsonl` (cần `id`, `ref`, `hyp`).
        kind: `unk` hoặc `empty` (xem đầu module).

    Lỗi:
        ValueError: khi `kind` không hợp lệ.
    """
    if kind not in KINDS:
        raise ValueError(f"kind: {kind!r} not in {'|'.join(KINDS)}")
    out: list[str] = []
    for r in rows:
        if kind == "unk":
            bad = is_unk_only(r["hyp"])
        else:
            bad = bool(normalize_text(r["ref"])) and not normalize_text(r["hyp"])
        if bad and r["id"] not in out:
            out.append(r["id"])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: đọc `hyps.jsonl`, ghi id các mẫu hỏng ra `--out` và in số lượng."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_bad_ids", description=__doc__.splitlines()[0])
    ap.add_argument("hyps", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=KINDS, default="unk")
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.hyps.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = bad_ids(rows, args.kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    print(f"{len(ids)} samples with {args.kind} utterances -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

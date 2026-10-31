"""Tạo bộ dữ liệu riêng cho một split: bản chép đầy đủ từng mẫu, kèm lời chuẩn đã chuẩn hoá như bản 2 dùng để học và
chấm (máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.prepared_dataset --ids configs/splits/train.txt --out data/derived/prepared/train.jsonl
  python -m lifemate.eval.prepared_dataset --ids configs/splits/val_seen.txt --out data/derived/prepared/val_seen.jsonl

Nội dung chính:
- `prepared_record`: một dòng của bộ dữ liệu: nguyên bản ghi của mẫu trong `train.jsonl` của BTC, thêm trường
  `audio_text_prepared` cho lượt cuối và cho từng lượt người dùng trong `history`.
- `main`: chép các mẫu của file id theo đúng thứ tự, ghi file JSON Lines và `<out>.meta.json`.

Cách hoạt động:
- Chuẩn hoá bằng `asr_merge_v2.prepare` (chữ thường, bỏ dấu câu dính hai đầu từ, đổi số đọc thành chữ số), đúng hàm
  dùng cho lời chuẩn khi tạo nhãn và tính WER của bộ ghép bản 2, nên xem lại được chính xác thứ bộ ghép đã so.
- Bản ghi gốc giữ nguyên (gồm `label`); trường mới chỉ được thêm, không sửa trường cũ.

Lưu ý:
- Đây là bản sao dữ liệu thi của BTC: chỉ ghi trong `data/` (git bỏ qua), không commit, không đưa vào notebook hay log.
- Không dùng cho `val_unseen` khi chưa chốt bài nộp (D53); lệnh từ chối file id của `val_unseen`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT, repo_path
from lifemate.eval.asr_merge_v2 import prepare
from lifemate.io.data import read_ids


def prepared_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Bản sao của `record` (một dòng `train.jsonl`) thêm `audio_text_prepared` ở lượt cuối và từng lượt người dùng."""
    out = copy.deepcopy(dict(record))
    out["audio_text_prepared"] = prepare(str(out.get("audio_text") or ""))
    for turn in out.get("history") or []:
        if turn.get("role") == "user":
            turn["audio_text_prepared"] = prepare(str(turn.get("audio_text") or ""))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: xem docstring module."""
    from lifemate.core.config import load_profile

    ap = argparse.ArgumentParser(prog="lifemate.eval.prepared_dataset", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)
    if "val_unseen" in args.ids.name:
        raise SystemExit("val_unseen bị khoá (D53)")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if not out.resolve().is_relative_to((REPO_ROOT / "data").resolve()):
        raise SystemExit("bản sao dữ liệu thi chỉ được ghi trong data/ (git bỏ qua)")
    root = repo_path(load_profile(args.env, REPO_ROOT).data_root, REPO_ROOT)
    source = root / args.split / f"{args.split}.jsonl"
    ids = read_ids(args.ids if args.ids.is_absolute() else REPO_ROOT / args.ids)
    by_id: dict[str, dict[str, Any]] = {}
    with open(source, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_id[str(rec["id"])] = rec
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} ids not in {source}: {missing[:3]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps(prepared_record(by_id[i]), ensure_ascii=False) + "\n")
    meta = {
        "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "ids_file": str(args.ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "n_samples": len(ids),
        "added_fields": ["audio_text_prepared (lượt cuối và từng lượt user trong history)"],
        "prepare": "lifemate.eval.asr_merge_v2.prepare",
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ids)} samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tạo cặp dữ liệu fine-tune ASR (file wav, `audio_text` nguyên văn) cho lượt cuối và mọi lượt user trong lịch sử (SPEC
6.4.1).

Nội dung chính:
- `sample_pairs`: các cặp của một mẫu.
- `build_pairs`: các cặp của nhiều mẫu.
- `parse_args`, `run`, `main`: CLI tạo file cặp kèm file meta.

Cách hoạt động:
1. Đọc file id. Chặn id thuộc `val_seen`/`val_unseen`, trừ khi có `--allow-eval` (chỉ dành cho file đo).
2. Nạp mẫu theo đúng thứ tự trong file id.
3. Mỗi mẫu cho các dòng `u1, u2, …` rồi `final`.
4. Ghi file jsonl và `<out>.meta.json` (sha256, argv, phiên bản code).

Lưu ý:
- Đường dẫn audio giữ tương đối với thư mục split; file audio không bị sao chép.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_profile
from lifemate.core.envinfo import code_version
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample
from lifemate.io.data import load_split, read_ids
from lifemate.train.common import abs_path, sha256_file
from lifemate.train.sft_data import eval_ids, write_jsonl

log = structlog.get_logger(__name__)

FINAL_TURN = "final"


def sample_pairs(sample: Sample, split_tag: str) -> list[dict[str, Any]]:
    """Tạo các dòng `{id, turn, audio, text, split}` của một mẫu.

    Thứ tự giống thứ tự ASR (SPEC 6.1): các lượt user trong lịch sử `u1, u2, …` trước, lượt cuối sau.

    Tham số:
        sample: mẫu có `audio_text` ở mọi lượt user.
        split_tag: giá trị của trường `split` trong mỗi dòng.

    Trả về:
        Danh sách dòng.

    Lỗi:
        ValueError: khi một lượt bất kỳ thiếu `audio` hoặc `audio_text`.
    """
    rows: list[dict[str, Any]] = []
    k = 0
    for turn in sample.history:
        if turn.role != "user":
            continue
        k += 1
        if turn.audio is None or turn.audio_text is None:
            raise ValueError(f"{sample.id}: user turn u{k} lacks audio or audio_text")
        rows.append(
            {"id": sample.id, "turn": f"u{k}", "audio": turn.audio, "text": turn.audio_text, "split": split_tag}
        )
    if sample.audio_text is None:
        raise ValueError(f"{sample.id}: final turn has no audio_text")
    rows.append(
        {"id": sample.id, "turn": FINAL_TURN, "audio": sample.audio, "text": sample.audio_text, "split": split_tag}
    )
    return rows


def build_pairs(samples: Sequence[Sample], split_tag: str) -> list[dict[str, Any]]:
    """Tạo các cặp của mọi mẫu theo thứ tự; đường dẫn audio giữ tương đối với thư mục split (không sao chép file)."""
    return [row for s in samples for row in sample_pairs(s, split_tag)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_data", description="Build ASR pairs (6.4.1).")
    ap.add_argument("--env", default="local", help="env profile name or path")
    ap.add_argument("--ids", required=True, type=Path, help="file with one sample id per line")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl (e.g. data/asr/v1.jsonl)")
    ap.add_argument("--split", default="train", help="data split folder holding the ids")
    ap.add_argument("--tag", help="value of the `split` field (default: ids file stem, e.g. train)")
    ap.add_argument("--root", help="data root override")
    ap.add_argument("--allow-eval", action="store_true", help="allow val ids (measurement files only)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Tạo một file cặp ASR kèm `<out>.meta.json`.

    Tham số:
        args: tham số dòng lệnh đã đọc.
        repo_root: thư mục gốc repo.
        argv: tham số gốc, ghi vào file meta; `None` thì lấy `sys.argv`.

    Trả về:
        Dict meta đã ghi.
    """
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = abs_path(profile.data_root, repo_root)
    ids_path = abs_path(args.ids, repo_root)
    ids = read_ids(ids_path)
    # Chống lộ dữ liệu đánh giá: id của tập val chỉ được dùng khi cố ý tạo file đo.
    leaked = sorted(set(ids) & eval_ids(repo_root))
    if leaked and not args.allow_eval:
        raise SystemExit(f"refusing: {len(leaked)} ids are in val_seen/val_unseen (e.g. {leaked[:3]}); --allow-eval")
    # `load_split` trả mẫu theo thứ tự trong file dữ liệu, không theo file id; sắp lại để đầu ra ổn định.
    order = {sid: i for i, sid in enumerate(ids)}
    samples = sorted(load_split(root, args.split, ids), key=lambda s: order[s.id])
    rows = build_pairs(samples, args.tag or ids_path.stem)
    out = abs_path(args.out, repo_root)
    write_jsonl(out, rows)
    version, dirty = code_version(repo_root)
    meta = {
        "file": out.name,
        "sha256": sha256_file(out),
        "argv": list(argv if argv is not None else sys.argv),
        "code_version": version,
        "code_dirty": dirty,
        "split": args.split,
        "ids": ids,
        "n_samples": len(samples),
        "n_rows": len(rows),
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("asr_data_done", rows=len(rows), samples=len(samples))
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI và in số dòng, số mẫu đã tạo; trả về mã thoát 0."""
    meta = run(parse_args(argv), argv=argv)
    print(f"rows={meta['n_rows']} samples={meta['n_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

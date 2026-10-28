"""Chấm lại chữ ASR đã có (`hyps.jsonl` của một lần `eval_asr`) bằng các chỉ số hiện tại, không chạy lại model.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.rescore_asr --env local --hyps <run>/large-v3__report_val_seen.hyps.jsonl \\
      --ids configs/splits/val_seen_en_mixed.txt --config <run>/large-v3__report_val_seen.toml --out x.json

Nội dung chính:
- `ReplayTranscriber`: transcriber trả lại chữ đã ghi trong `hyps.jsonl`.
- `main`: CLI.

Cách hoạt động:
- Giải mã ASR là tất định, nên chạy lại cùng config cho cùng chữ; chấm lại chữ đã có cho cùng số với chạy lại.
- `--ids` chọn tập con, ví dụ `*_en_mixed.txt`; mẫu trong `--ids` phải có trong `hyps.jsonl`.
- `--config` (tuỳ chọn) là config của lần chạy gốc, chỉ để lấy câu mồi (`initial_prompt`) cho các chỉ số câu mồi.

Lưu ý:
- Thời gian (`p50_s`, `p95_s`, `max_s`) bằng 0 vì không chạy model; lấy thời gian từ lần chạy gốc.
- `n_prompt_retries` bằng 0: lần chạy gốc chưa có lớp bảo vệ câu mồi.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT, repo_path
from lifemate.core.types import Sample, Transcript
from lifemate.eval.eval_asr import evaluate


class ReplayTranscriber:
    """Trả lại chữ đã ghi trong `hyps.jsonl` thay vì chạy ASR.

    Tham số:
        rows: các dòng `{id, turn, hyp, raw}`; `turn` là "final" hoặc chỉ số lượt trong lịch sử (dạng chuỗi).
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = {(str(r["id"]), str(r["turn"])): r for r in rows}

    def warmup(self) -> None:
        """Không cần làm gì."""

    def transcribe(self, sample: Sample) -> Transcript:
        """Dựng `Transcript` từ các dòng của mẫu; lượt user không có dòng nào thì là chuỗi rỗng.

        Lỗi:
            KeyError: khi `hyps.jsonl` không có lượt cuối của mẫu.
        """
        history: list[str | None] = []
        raw: list[str] = []
        for i, turn in enumerate(sample.history):
            if turn.role != "user":
                history.append(None)
                continue
            row = self._rows.get((sample.id, str(i)))
            hyp = str(row["hyp"]) if row else ""
            history.append(hyp)
            raw.append(str(row.get("raw", hyp)) if row else hyp)
        final = self._rows[(sample.id, "final")]
        raw.append(str(final.get("raw", final["hyp"])))
        return Transcript(final=str(final["hyp"]), history=tuple(history), seconds=0.0, raw=tuple(raw))


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI chấm lại và in bản tóm tắt; trả về mã thoát 0."""
    from lifemate.core.config import load_config, load_profile
    from lifemate.io.data import load_split

    ap = argparse.ArgumentParser(prog="lifemate.eval.rescore_asr", description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--hyps", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="config của lần chạy gốc, để lấy câu mồi")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    profile = load_profile(args.env, REPO_ROOT).with_overrides(root=args.root)
    root = repo_path(profile.data_root, REPO_ROOT)
    ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    rows = [json.loads(line) for line in args.hyps.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = ""
    if args.config is not None:
        # `load_config` đọc cả `asr.initial_prompt_file`, nên câu mồi để trong file cũng lấy được.
        prompt = str(load_config(args.config).asr.generate_kwargs.get("initial_prompt") or "")
    report = evaluate(samples, ReplayTranscriber(rows), prompt=prompt)
    print(report.to_text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

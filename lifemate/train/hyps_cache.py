"""Dựng cache ASR cho dữ liệu SFT từ chữ đã lưu của các phiên GPU benchmark (máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.train.hyps_cache --k1 <K1> --k2 <K2> --k3 <K3> --winner E2 \\
      --splits train val_seen --out data/asr/train_asr_v3h.json

Nội dung chính:
- `cache_from_rows`: gom các dòng `hyps.jsonl` (một dòng một câu) thành mục cache của từng mẫu, đúng định dạng của
  `lifemate.train.asr_cache` (`{"final": str, "history": [str | None, ...]}`).
- `main`: với mỗi tập, dựng chữ ghép H (học lại trên cả train như D54) rồi đổi số (L1, D56) bằng các hàm của
  `lifemate.eval.asr_v4_report`, ghi một file cache và file meta.

Cách hoạt động:
- `asr_cache.py` chạy ASR thật trên GPU; module này thay bước đó bằng chữ đã có từ phiên K2 (RNNT, large-v3 câu mồi
  thắng) và điểm âm thanh của phiên K3, qua đúng chuỗi xử lý của pipeline đã chốt: ghép H rồi `digitize`. Nhờ vậy dữ
  liệu SFT (`transcript_source = "asr"` hoặc `"mix"`) học trên chữ giống chữ lúc thi.
- Lượt lịch sử không phải của người dùng là `None`; lượt người dùng thiếu dòng chữ là chuỗi rỗng và được đếm vào meta.

Lưu ý:
- `val_unseen` chỉ dựng khi chốt cấu hình nộp bài (D53); mặc định không dựng.
- Khi `MergedASR` của hệ thống thật xong, cache này phải được dựng lại bằng `asr_cache.py` và so từng byte với cache
  này trên vài mẫu.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample


def cache_from_rows(samples: Sequence[Sample], rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], int]:
    """Mục cache của từng mẫu từ các dòng `{id, turn, hyp}`.

    Trả về:
        (cache `{id: {"final", "history"}}`, số lượt người dùng không có dòng chữ).
    """
    hyp = {(str(r["id"]), str(r["turn"])): str(r["hyp"]) for r in rows}
    cache: dict[str, Any] = {}
    missing = 0
    for s in samples:
        history: list[str | None] = []
        for i, turn in enumerate(s.history):
            if turn.role != "user":
                history.append(None)
                continue
            text = hyp.get((s.id, str(i)))
            missing += text is None
            history.append(text or "")
        final = hyp.get((s.id, "final"))
        missing += final is None
        cache[s.id] = {"final": final or "", "history": history}
    return cache, missing


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: dựng cache cho các tập `--splits` và ghi `--out` cùng `<out>.meta.json`."""
    from lifemate.eval import asr_v3_analysis as A  # noqa: N812
    from lifemate.eval.asr_v4_report import SPLIT_PARTS, digitized, h_full_train, load_parts

    ap = argparse.ArgumentParser(prog="lifemate.train.hyps_cache", description=__doc__.split("\n\n")[0])
    for k in ("--k1", "--k2", "--k3"):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument("--winner", default="E2")
    ap.add_argument("--splits", nargs="+", default=["train", "val_seen"], choices=sorted(SPLIT_PARTS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if "val_unseen" in args.splits:
        raise SystemExit("val_unseen bị khoá (D53): chỉ dựng khi chốt cấu hình nộp bài")
    make_h, h_info = h_full_train(args)
    cache: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "source": "hyps_cache: RNNT + large-v3 câu mồi thắng, ghép H học trên cả train (D54), đổi số L1 (D56)",
        "winner_prompt": args.winner,
        "h": h_info,
        "k1": str(args.k1),
        "k2": str(args.k2),
        "k3": str(args.k3),
        "splits": {},
    }
    for split in args.splits:
        data = load_parts(args, SPLIT_PARTS[split])
        rows = digitized(A.apply_decider(data["rnnt"], data["utts"], make_h(data["context"])))
        part, missing = cache_from_rows(data["samples"], rows)
        cache |= part
        meta["splits"][split] = {"n_samples": len(part), "missing_turns": missing}
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["splits"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Giải thích vì sao một lần chạy mất điểm, theo từng lời gọi: thiếu hoặc thừa lời gọi, đọc thừa, sai tham số.

Nội dung: script so sánh lời gọi dự đoán trong `submission.jsonl` với lời gọi gold của từng mẫu có nhãn, rồi gom
các khác biệt thành bảng thống kê theo loại lỗi, theo tool và theo tham số.

Cách xây dựng:
  - Ghép cặp theo tên tool: mỗi lời gọi gold lấy lời gọi dự đoán đầu tiên cùng tên chưa được dùng. Lời gọi gold
    không có cặp là MISSING_CALL; lời gọi dự đoán không có cặp là EXTRA_CALL, hoặc REDUNDANT_READ khi tool thuộc
    loại `read` trong catalog và lịch sử hội thoại đã có kết quả tool.
  - Với mỗi cặp (trừ `sys.ask_missing_info`), so từng tham số: thiếu, thừa, khác kiểu, chỉ khác chữ hoa/thường
    (CASING_ONLY) hoặc khác giá trị.
  - Gold rỗng mà vẫn gọi tool thì ghi một lỗi riêng CALLED_ON_EMPTY_LABEL.
  - Luồng dữ liệu: đọc split, catalog (lấy `kind` của tool) và submission, kiểm tra từng id, ghi jsonl chi tiết
    nếu có `--out`, cuối cùng in bản tóm tắt JSON.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/evaluation/audit_labels.py@c703bd9. Changes from
the source: predicted and gold calls are paired by tool name, not by position (the scorer ignores call order);
a read call is recognised by its catalog `kind`, not by words in its name; out-of-scope answers that called tools
get their own type. The score itself stays in `lifemate.eval.score`; this script only lists differences, so it can
also point at doubtful gold labels (for example a value that differs only in letter case).
(Chuyển thể từ repo của trưởng nhóm. Khác bản gốc: ghép cặp theo tên tool thay vì theo vị trí, vì bộ chấm bỏ qua
thứ tự lời gọi; lời gọi đọc được nhận ra qua `kind` trong catalog; câu trả lời ngoài phạm vi mà vẫn gọi tool có loại
lỗi riêng. Điểm số vẫn nằm ở `lifemate.eval.score`; script này chỉ liệt kê khác biệt, nên còn giúp phát hiện nhãn
gold đáng ngờ, ví dụ giá trị chỉ khác chữ hoa/thường.)

Usage:
  python -m lifemate.eval.audit artifacts/runs/<run>/submission.jsonl --ids configs/splits/val_seen.txt \\
      --out artifacts/runs/<run>/audit.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.eval.score import DEFAULT_ROOT
from lifemate.io.data import read_ids

ASK_TOOL = "sys.ask_missing_info"


def _history_has_results(sample: Mapping[str, Any]) -> bool:
    """Kiểm tra lịch sử hội thoại của mẫu đã có lượt nào chứa kết quả tool hay chưa.

    Khi đã có kết quả, một lời gọi đọc thừa được xem là đọc lại dữ liệu sẵn có (REDUNDANT_READ).
    """
    return any(turn.get("tool_results") for turn in sample.get("history") or [])


def _pair(pred: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> tuple[list, list[int], list[int]]:
    """Ghép cặp lời gọi dự đoán với lời gọi gold cùng tên tool.

    Trả về (các cặp (pred, gold) cùng tên, chỉ số gold không có cặp, chỉ số pred không có cặp). Ghép tham lam:
    mỗi lời gọi gold, theo thứ tự, lấy lời gọi dự đoán đầu tiên cùng tên chưa dùng; thứ tự lời gọi không quan trọng.
    """
    used: set[int] = set()
    pairs, missing = [], []
    for gi, g in enumerate(gold):
        # Lời gọi dự đoán đầu tiên cùng tên và chưa được ghép.
        pi = next((i for i, p in enumerate(pred) if i not in used and p.get("name") == g.get("name")), None)
        if pi is None:
            missing.append(gi)
        else:
            used.add(pi)
            pairs.append((pred[pi], g))
    return pairs, missing, [i for i in range(len(pred)) if i not in used]


def _arg_issues(tool: str, p_args: Mapping[str, Any], g_args: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Liệt kê khác biệt tham số giữa một lời gọi dự đoán và lời gọi gold cùng tên.

    Mỗi khóa trong hợp hai bộ tham số được xếp vào một loại: MISSING_ARGUMENT, EXTRA_ARGUMENT, TYPE_MISMATCH,
    CASING_ONLY (chỉ khác chữ hoa/thường hoặc khoảng trắng hai đầu) hoặc VALUE_MISMATCH.
    """
    out: list[dict[str, Any]] = []
    for k in sorted(set(p_args) | set(g_args)):
        if k not in p_args:
            out.append({"type": "MISSING_ARGUMENT", "tool": tool, "param": k, "gold": g_args[k]})
        elif k not in g_args:
            out.append({"type": "EXTRA_ARGUMENT", "tool": tool, "param": k, "pred": p_args[k]})
        else:
            p, g = p_args[k], g_args[k]
            # So kiểu chính xác: bool khác int, int khác float.
            if type(p) is not type(g):
                out.append({"type": "TYPE_MISMATCH", "tool": tool, "param": k, "pred": p, "gold": g})
            elif p != g:
                # Chỉ khác chữ hoa/thường thường là dấu hiệu nhãn gold đáng ngờ, nên tách thành loại riêng.
                kind = "CASING_ONLY" if str(p).strip().lower() == str(g).strip().lower() else "VALUE_MISMATCH"
                out.append({"type": kind, "tool": tool, "param": k, "pred": p, "gold": g})
    return out


def audit_sample(
    sample: Mapping[str, Any], pred: Sequence[Mapping[str, Any]], tool_kind: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Liệt kê khác biệt giữa lời gọi dự đoán và lời gọi gold của một mẫu có nhãn.

    `tool_kind` ánh xạ tên tool sang `kind` trong catalog (ví dụ `read`), dùng để nhận ra lời gọi đọc thừa.
    Tham số của `sys.ask_missing_info` không được so, vì nội dung câu hỏi lại không cần trùng khớp.
    """
    gold = list((sample.get("label") or {}).get("tool_calls") or [])
    # Gold rỗng (ngoài phạm vi) mà vẫn gọi tool: một lỗi duy nhất, không xét từng lời gọi.
    if not gold and pred:
        return [{"type": "CALLED_ON_EMPTY_LABEL", "tools": [p.get("name") for p in pred]}]
    issues: list[dict[str, Any]] = []
    pairs, missing, extra = _pair(pred, gold)
    for gi in missing:
        issues.append({"type": "MISSING_CALL", "tool": gold[gi].get("name")})
    for pi in extra:
        name = str(pred[pi].get("name"))
        # Lời gọi đọc thừa khi lịch sử đã có kết quả tool được tính là đọc lại, không phải gọi sai tool.
        redundant = tool_kind.get(name) == "read" and _history_has_results(sample)
        issues.append({"type": "REDUNDANT_READ" if redundant else "EXTRA_CALL", "tool": name})
    for p, g in pairs:
        if g.get("name") != ASK_TOOL:
            issues += _arg_issues(str(g.get("name")), p.get("arguments") or {}, g.get("arguments") or {})
    return issues


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Đếm lỗi theo loại, theo (loại, tool) và theo (loại, tool, tham số), kèm số mẫu không có lỗi.

    Bảng theo tool và theo tham số chỉ giữ 15 mục phổ biến nhất.
    """
    by_type: collections.Counter[str] = collections.Counter()
    by_tool: collections.Counter[str] = collections.Counter()
    by_param: collections.Counter[str] = collections.Counter()
    for row in rows:
        for i in row["issues"]:
            by_type[i["type"]] += 1
            by_tool[f"{i['type']} {i.get('tool', '')}"] += 1
            if "param" in i:
                by_param[f"{i['type']} {i['tool']}.{i['param']}"] += 1
    return {
        "samples": len(rows),
        "clean": sum(1 for r in rows if not r["issues"]),
        "by_type": dict(by_type.most_common()),
        "top_tools": dict(by_tool.most_common(15)),
        "top_params": dict(by_param.most_common(15)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào dòng lệnh: đọc dữ liệu, kiểm tra từng mẫu, ghi chi tiết (tùy chọn) và in bản tóm tắt."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path, help="submission.jsonl of a run on a labelled split")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding the split folders")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, help="only these ids (default: every id in the submission)")
    ap.add_argument("--out", type=Path, help="write one jsonl row per sample with its issues")
    args = ap.parse_args(argv)

    with open(args.root / args.split / f"{args.split}.jsonl", encoding="utf-8") as f:
        samples = {s["id"]: s for s in map(json.loads, filter(str.strip, f))}
    # Bảng tên tool -> kind (ví dụ "read"), lấy từ catalog của split.
    with open(args.root / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8") as f:
        tool_kind = {t["name"]: t["kind"] for app in json.load(f)["apps"] for t in app["tools"]}
    with open(args.submission, encoding="utf-8") as f:
        preds = {r["id"]: r.get("tool_calls") or [] for r in map(json.loads, filter(str.strip, f))}
    # Id thiếu trong submission được xem như không gọi tool nào.
    ids = read_ids(args.ids) if args.ids else sorted(preds)
    rows = [{"id": i, "issues": audit_sample(samples[i], preds.get(i, []), tool_kind)} for i in ids]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

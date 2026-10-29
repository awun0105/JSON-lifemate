"""Thử nghiệm J của benchmark ASR lần 3: điểm thi khi chạy agent trên chữ ASR đã lưu (báo cáo `asr_v3.md`).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.asr_v3_agent diff --base j1.hyps.jsonl --other j3.hyps.jsonl --out j3_ids.txt
  python -m lifemate.eval.asr_v3_agent score --ids configs/splits/val_seen.txt --base J1=<run J1> \\
      --source J0=<run J0> --source J3=<run J3, chạy trên j3_ids.txt> ... --out j_val_seen.json

Nội dung chính:
- `differing_ids`: các mẫu có ít nhất một lượt người dùng mà chữ của nguồn khác chữ của nguồn mốc.
- `source_scores`: điểm từng mẫu của một nguồn; mẫu không chạy lại thì lấy điểm của nguồn mốc.
- `summarize`: điểm trung bình, CI 95% bootstrap theo mẫu, chênh lệch so với mốc, đếm loại lỗi, dấu ngoặc lọt ra.

Cách hoạt động:
- Agent giải mã tất định (nhiệt độ 0, cùng seed), nên mẫu có chữ giống hệt nguồn mốc cho cùng lời gọi tool. Vì vậy
  nguồn J2, J3, J4 chỉ chạy lại các mẫu có chữ khác J1 (RNNT); các mẫu còn lại dùng điểm của J1. J0 (chữ chuẩn) và J1
  chạy đủ mọi mẫu.
- Điểm một mẫu là trường `score` của `details.jsonl` (0 tới 1); điểm tập là trung bình nhân 100, như `score.txt`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    """Đọc một file JSON Lines."""
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def differing_ids(base: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> list[str]:
    """Mã các mẫu có ít nhất một câu mà `hyp` của `other` khác `hyp` của `base` (so sau khi gộp khoảng trắng)."""
    b = {(str(r["id"]), str(r["turn"])): " ".join(str(r["hyp"]).split()) for r in base}
    out = {
        str(r["id"]) for r in other if " ".join(str(r["hyp"]).split()) != b.get((str(r["id"]), str(r["turn"])))
    }
    return sorted(out)


def source_scores(
    ids: Sequence[str], base_details: Mapping[str, Mapping[str, Any]], run_details: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Dòng `details` của từng mẫu cho một nguồn: lấy từ lần chạy của nguồn nếu có, không thì lấy của nguồn mốc."""
    return {i: run_details.get(i, base_details[i]) for i in ids}


def _ci(xs: Sequence[float]) -> list[float]:
    """Khoảng 2,5% – 97,5% của các giá trị bootstrap."""
    s = sorted(xs)
    return [s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]]


def summarize(
    ids: Sequence[str],
    sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline: str,
    submissions: Mapping[str, Path | None] | None = None,
    rounds: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Điểm của từng nguồn, CI 95%, chênh lệch so với `baseline` (bootstrap ghép cặp theo mẫu), và đếm lỗi.

    Tham số:
        ids: các mẫu của tập.
        sources: tên nguồn -> mã mẫu -> dòng `details`.
        baseline: tên nguồn mốc để tính chênh lệch.
        submissions: tên nguồn -> `submission.jsonl` của lần chạy (để đếm dấu ngoặc `{` `|` `}` lọt vào tham số).
    """
    rng = random.Random(seed)
    picks = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(rounds)]
    out: dict[str, Any] = {}
    for name, rows in sources.items():
        scores = [float(rows[i]["score"]) for i in ids]
        boot = [sum(float(rows[i]["score"]) for i in pick) / len(pick) for pick in picks]
        base_boot = [sum(float(sources[baseline][i]["score"]) for i in pick) / len(pick) for pick in picks]
        errors: dict[str, int] = {}
        for i in ids:
            for e in rows[i].get("errors") or []:
                errors[e] = errors.get(e, 0) + 1
        entry: dict[str, Any] = {
            "score": 100 * sum(scores) / len(scores),
            "ci": [100 * v for v in _ci(boot)],
            "diff_ci": [100 * v for v in _ci([a - b for a, b in zip(boot, base_boot, strict=True)])],
            "perfect": sum(s >= 1.0 for s in scores),
            "lost_vs_baseline": sum(float(rows[i]["score"]) < float(sources[baseline][i]["score"]) for i in ids),
            "gained_vs_baseline": sum(float(rows[i]["score"]) > float(sources[baseline][i]["score"]) for i in ids),
            "errors": errors,
        }
        sub = (submissions or {}).get(name)
        if sub is not None and Path(sub).is_file():
            # Mẫu có giá trị tham số dạng chuỗi chứa dấu ngoặc `{` `}` hoặc `|` của định dạng hai phương án.
            leaked = 0
            for row in _rows(Path(sub)):
                values = [
                    v for call in row.get("tool_calls") or [] for v in (call.get("arguments") or {}).values()
                    if isinstance(v, str)
                ]  # fmt: skip
                leaked += any(ch in v for v in values for ch in "{}|")
            entry["bracket_leaks"] = leaked
        out[name] = entry
    return out


def _details(run_dir: Path) -> dict[str, dict[str, Any]]:
    """`details.jsonl` của một thư mục run, theo mã mẫu."""
    return {str(r["id"]): r for r in _rows(run_dir / "details.jsonl")}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `diff` (mẫu cần chạy lại) hoặc `score` (điểm các nguồn)."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_v3_agent", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pd = sub.add_parser("diff")
    pd.add_argument("--base", type=Path, required=True)
    pd.add_argument("--other", type=Path, required=True)
    pd.add_argument("--out", type=Path, required=True)
    ps = sub.add_parser("score")
    ps.add_argument("--ids", type=Path, required=True)
    ps.add_argument("--base", required=True, metavar="NAME=RUN_DIR")
    ps.add_argument("--source", action="append", default=[], metavar="NAME=RUN_DIR")
    ps.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "diff":
        ids = differing_ids(_rows(args.base), _rows(args.other))
        args.out.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        print(f"{len(ids)} samples differ -> {args.out}")
        return 0
    ids = [x for x in args.ids.read_text(encoding="utf-8").split() if x]
    base_name, _, base_dir = args.base.partition("=")
    base = _details(Path(base_dir))
    sources: dict[str, Mapping[str, Mapping[str, Any]]] = {base_name: source_scores(ids, base, base)}
    subs: dict[str, Path | None] = {base_name: Path(base_dir) / "submission.jsonl"}
    for spec in args.source:
        name, _, run_dir = spec.partition("=")
        sources[name] = source_scores(ids, base, _details(Path(run_dir)))
        subs[name] = Path(run_dir) / "submission.jsonl"
    res = summarize(ids, sources, base_name, subs)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({n: [round(r["score"], 2), [round(x, 2) for x in r["diff_ci"]]] for n, r in res.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

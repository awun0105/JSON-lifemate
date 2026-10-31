"""So nhiều lần chạy agent trên cùng một tập mẫu, đủ mọi chỉ số đo được từ thư mục run (máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.agent_report --base L0=<run L0> --run L0v2=<run> --run J5v2=<run> --out <file.json>

Nội dung chính:
- `run_metrics`: các chỉ số của một lần chạy: điểm và điểm theo loại câu, độ chính xác quyết định (thực thi / hỏi lại /
  không làm gì), loại lỗi của bộ chấm, lý do dừng, số mẫu hết giờ, thời gian (tổng, bước đầu, đọc trước), số bước,
  số token, số lần guard chặn theo mã.
- `paired`: chênh lệch điểm so với lần chạy mốc, CI 95% bootstrap ghép cặp theo mẫu, số mẫu tăng / giảm điểm.
- `markdown`: bảng tóm tắt để dán vào báo cáo.

Cách hoạt động:
- Đọc `details.jsonl` (điểm và lỗi từng mẫu, do `lifemate/eval/score.py` ghi), `trace.jsonl` (từng bước của agent) và
  `summary.json` (điểm `strict`, `canon`) của mỗi run.
- Mọi run phải có cùng tập mẫu với run mốc; thiếu mẫu thì báo lỗi, để không so hai tập khác nhau.
- Bootstrap: 1 000 vòng, seed 0, chọn lại mẫu có hoàn lại, như các báo cáo benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DECISIONS = ("execute", "ask_missing_info", "out_of_scope")


def _rows(path: Path) -> list[dict[str, Any]]:
    """Đọc một file JSON Lines."""
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _pct(values: Sequence[float], q: float) -> float:
    """Phân vị theo hạng gần nhất; 0 khi rỗng."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q / 100 * len(s)) - 1))]


def load_run(run_dir: Path) -> dict[str, Any]:
    """Nội dung cần thiết của một thư mục run: `details`, `trace` theo mã mẫu, và `summary`."""
    run_dir = Path(run_dir)
    return {
        "details": {r["id"]: r for r in _rows(run_dir / "details.jsonl")},
        "trace": {r["id"]: r for r in _rows(run_dir / "trace.jsonl")},
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
    }


def run_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    """Các chỉ số của một lần chạy (xem docstring module)."""
    details, trace = run["details"], run["trace"]
    ids = sorted(details)
    scores = [float(details[i]["score"]) for i in ids]
    by_cat: dict[str, list[float]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    errors: Counter[str] = Counter()
    for i in ids:
        d = details[i]
        by_cat.setdefault(str(d["category"]), []).append(float(d["score"]))
        confusion[(str(d["category"]), str(d.get("predicted")))] += 1
        errors.update(str(e) for e in d.get("errors") or [])
    stops = Counter(str(trace[i]["stop_reason"]) for i in ids if i in trace)
    total_s, first_s, prefill_s, steps, completion, max_prompt = [], [], [], [], [], []
    guard_codes: Counter[str] = Counter()
    warn_codes: Counter[str] = Counter()
    for i in ids:
        t = trace.get(i)
        if t is None:
            continue
        timing = t.get("timing") or {}
        total_s.append(float(timing.get("total_s") or 0.0))
        prefill_s.append(float(timing.get("prefill_s") or 0.0))
        st = t.get("steps") or []
        steps.append(len(st))
        if st:
            first_s.append(float(st[0].get("llm_s") or 0.0))
        completion.append(int((t.get("tokens") or {}).get("completion_total") or 0))
        max_prompt.append(int((t.get("tokens") or {}).get("max_prompt") or 0))
        for s in st:
            if s.get("error"):
                guard_codes[str(s["error"])] += 1
            warn_codes.update(str(w) for w in s.get("warnings") or [])
    n = len(ids)
    correct = sum(v for (gold, pred), v in confusion.items() if gold == pred)
    summary_score = (run.get("summary") or {}).get("score") or {}
    return {
        "n": n,
        "score": 100 * sum(scores) / n if n else 0.0,
        "score_canon": (summary_score.get("by_mode") or {}).get("canon"),
        "perfect": sum(s >= 1.0 for s in scores),
        "zero": sum(s == 0.0 for s in scores),
        "by_category": {c: {"n": len(v), "score": 100 * sum(v) / len(v)} for c, v in sorted(by_cat.items())},
        "decision_accuracy": 100 * correct / n if n else 0.0,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(confusion.items())},
        "error_types": dict(errors.most_common()),
        "stop_reason": dict(stops.most_common()),
        "timeouts": stops.get("timeout", 0),
        "seconds": {
            "p50": _pct(total_s, 50),
            "p95": _pct(total_s, 95),
            "max": max(total_s, default=0.0),
            "over_25": sum(x > 25 for x in total_s),
            "over_29": sum(x > 29 for x in total_s),
        },
        "first_step_s": {"p50": _pct(first_s, 50), "p95": _pct(first_s, 95), "max": max(first_s, default=0.0)},
        "prefill_s": {"p50": _pct(prefill_s, 50), "max": max(prefill_s, default=0.0)},
        "steps": {
            "mean": statistics.fmean(steps) if steps else 0.0,
            "p50": _pct(steps, 50),
            "max": max(steps, default=0),
        },
        "completion_tokens_mean": statistics.fmean(completion) if completion else 0.0,
        "max_prompt_tokens": {"p50": _pct(max_prompt, 50), "max": max(max_prompt, default=0)},
        "guard_blocks": dict(guard_codes.most_common()),
        "guard_warnings": dict(warn_codes.most_common()),
    }


def paired(
    base: Mapping[str, Mapping[str, Any]], other: Mapping[str, Mapping[str, Any]], rounds: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Chênh lệch điểm `other` − `base` (thang 0–100), CI 95% bootstrap ghép cặp theo mẫu, số mẫu tăng / giảm.

    Lỗi:
        ValueError: khi hai run không có cùng tập mẫu.
    """
    ids = sorted(base)
    if sorted(other) != ids:
        raise ValueError(f"runs differ in samples: {len(set(base) ^ set(other))} ids not shared")
    a = [float(other[i]["score"]) for i in ids]
    b = [float(base[i]["score"]) for i in ids]
    rng = random.Random(seed)
    diffs = []
    for _ in range(rounds):
        pick = [rng.randrange(len(ids)) for _ in ids]
        diffs.append(100 * sum(a[k] - b[k] for k in pick) / len(pick))
    diffs.sort()
    return {
        "diff": 100 * (sum(a) - sum(b)) / len(ids),
        "ci": [diffs[int(0.025 * rounds)], diffs[min(rounds - 1, int(0.975 * rounds))]],
        "gained": sum(x > y for x, y in zip(a, b, strict=True)),
        "lost": sum(x < y for x, y in zip(a, b, strict=True)),
    }


def markdown(res: Mapping[str, Any]) -> str:
    """Bảng tóm tắt các run (một dòng mỗi run) và bảng loại lỗi."""
    base = res["base"]
    lines = [
        f"| Run | Điểm | So với {base} (CI 95%) | Tăng / giảm | Thực thi / hỏi lại / ngoài phạm vi | Quyết định đúng"
        " | Hết giờ | Giây p50 / p95 | Bước đầu p50 / p95 | Bước TB | Guard chặn |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in res["runs"].items():
        p = res["paired"].get(name)
        vs = f"{p['diff']:+.2f} [{p['ci'][0]:+.2f}; {p['ci'][1]:+.2f}]" if p else "mốc"
        gl = f"{p['gained']} / {p['lost']}" if p else ""
        cat = " / ".join(f"{m['by_category'].get(c, {}).get('score', 0):.2f}" for c in DECISIONS)
        blocks = ", ".join(f"{k} {v}" for k, v in m["guard_blocks"].items()) or "0"
        lines.append(
            f"| {name} | {m['score']:.2f} | {vs} | {gl} | {cat} | {m['decision_accuracy']:.1f}% | {m['timeouts']}"
            f" | {m['seconds']['p50']:.1f} / {m['seconds']['p95']:.1f}"
            f" | {m['first_step_s']['p50']:.1f} / {m['first_step_s']['p95']:.1f} | {m['steps']['mean']:.2f}"
            f" | {blocks} |"
        )
    kinds = sorted({k for m in res["runs"].values() for k in m["error_types"]})
    lines += ["", "| Run | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    for name, m in res["runs"].items():
        lines.append(f"| {name} | " + " | ".join(str(m["error_types"].get(k, 0)) for k in kinds) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: xem docstring module."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.agent_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    specs = [args.base, *args.run]
    runs = {s.split("=", 1)[0]: load_run(Path(s.split("=", 1)[1])) for s in specs}
    base = args.base.split("=", 1)[0]
    res: dict[str, Any] = {
        "base": base,
        "runs": {n: run_metrics(r) for n, r in runs.items()},
        "paired": {n: paired(runs[base]["details"], r["details"]) for n, r in runs.items() if n != base},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

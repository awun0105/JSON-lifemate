"""Text reports over the files every run and training job already writes (SPEC 4.4, 6.2.6).

    python -m lifemate.report runs  [runs/index.csv]         # one line per agent run, newest first
    python -m lifemate.report train <work dir> [--png out]   # loss / eval curves from metrics.jsonl

No plotting dependency: tables are plain text; `--png` draws with matplotlib only when it is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

NUMERIC_SKIP = frozenset({"step", "epoch", "time"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All rows of a JSONL file (blank lines ignored)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Fixed-width text table; numbers are rendered with 4 decimals."""

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    body = [[cell(v) for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in body]
    return "\n".join(out)


def runs_table(index_csv: Path, limit: int | None = None) -> str:
    """runs/index.csv newest first, as text."""
    with open(index_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.reverse()
    if limit is not None:
        rows = rows[:limit]
    headers = ["run", "date", "git_commit", "model", "split", "n", "score", "p95_seconds"]
    return format_table(headers, [[r.get(h, "") for h in headers] for r in rows])


def metric_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Numeric metric keys in first-seen order (step/epoch/time excluded)."""
    cols: list[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in NUMERIC_SKIP and isinstance(v, int | float) and not isinstance(v, bool) and k not in cols:
                cols.append(k)
    return cols


def train_table(rows: Sequence[Mapping[str, Any]], every: int = 1) -> str:
    """metrics.jsonl as text: one line per logged step (every n-th), all numeric columns."""
    cols = metric_columns(rows)
    picked = [r for i, r in enumerate(rows) if i % max(1, every) == 0 or i == len(rows) - 1]
    return format_table(["step", *cols], [[r.get("step", ""), *(r.get(c) for c in cols)] for r in picked])


def best_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> list[Mapping[str, Any]]:
    """Rows that carry `metric`, lowest first."""
    have = [r for r in rows if isinstance(r.get(metric), int | float)]
    return sorted(have, key=lambda r: float(r[metric]))


def plot_png(rows: Sequence[Mapping[str, Any]], out: Path) -> bool:
    """Draw every numeric metric against step; False when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cols = metric_columns(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in cols:
        pts = [(r["step"], r[c]) for r in rows if "step" in r and isinstance(r.get(c), int | float)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=c, marker="." if len(pts) < 50 else None)
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def train_report(work_dir: Path, *, every: int, png: Path | None) -> str:
    """Report for one training work dir (`<work>/sft` or `<work>/asr` or a folder holding metrics.jsonl)."""
    candidates = [work_dir / "metrics.jsonl", work_dir / "sft" / "metrics.jsonl", work_dir / "asr" / "metrics.jsonl"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no metrics.jsonl under {work_dir}")
    rows = read_jsonl(path)
    parts = [f"{path}: {len(rows)} rows", train_table(rows, every)]
    for metric in ("eval_unseen_loss", "eval_seen_loss", "eval_loss", "loss"):
        best = best_rows(rows, metric)
        if best:
            parts.append(f"best {metric}: {float(best[0][metric]):.4f} at step {best[0].get('step')}")
    progress = work_dir / "pipeline_progress.json"
    if progress.is_file():
        parts.append(f"pipeline progress: {progress.read_text(encoding='utf-8').strip()}")
    if png is not None:
        parts.append(f"png: {png}" if plot_png(rows, png) else "png skipped: matplotlib not installed")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lifemate.report", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs", help="table of agent runs from runs/index.csv")
    r.add_argument("index", nargs="?", default="runs/index.csv", type=Path)
    r.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("train", help="loss / eval table from a training work dir")
    t.add_argument("work_dir", type=Path)
    t.add_argument("--every", type=int, default=1, help="print every n-th logged step")
    t.add_argument("--png", type=Path, default=None, help="also draw the curves (needs matplotlib)")
    args = ap.parse_args(argv)
    if args.cmd == "runs":
        print(runs_table(args.index, args.limit))
    else:
        print(train_report(args.work_dir, every=args.every, png=args.png))
    return 0


if __name__ == "__main__":
    sys.exit(main())

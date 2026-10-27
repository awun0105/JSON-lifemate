"""Explain why a run lost points, call by call: missing or extra calls, redundant reads, argument differences.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/evaluation/audit_labels.py@c703bd9. Changes from
the source: predicted and gold calls are paired by tool name, not by position (the scorer ignores call order);
a read call is recognised by its catalog `kind`, not by words in its name; out-of-scope answers that called tools
get their own type. The score itself stays in `lifemate.eval.score`; this script only lists differences, so it can
also point at doubtful gold labels (for example a value that differs only in letter case).

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
    return any(turn.get("tool_results") for turn in sample.get("history") or [])


def _pair(pred: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]) -> tuple[list, list[int], list[int]]:
    """(pairs of (pred, gold) with the same name, unmatched gold indices, unmatched pred indices)."""
    used: set[int] = set()
    pairs, missing = [], []
    for gi, g in enumerate(gold):
        pi = next((i for i, p in enumerate(pred) if i not in used and p.get("name") == g.get("name")), None)
        if pi is None:
            missing.append(gi)
        else:
            used.add(pi)
            pairs.append((pred[pi], g))
    return pairs, missing, [i for i in range(len(pred)) if i not in used]


def _arg_issues(tool: str, p_args: Mapping[str, Any], g_args: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in sorted(set(p_args) | set(g_args)):
        if k not in p_args:
            out.append({"type": "MISSING_ARGUMENT", "tool": tool, "param": k, "gold": g_args[k]})
        elif k not in g_args:
            out.append({"type": "EXTRA_ARGUMENT", "tool": tool, "param": k, "pred": p_args[k]})
        else:
            p, g = p_args[k], g_args[k]
            if type(p) is not type(g):
                out.append({"type": "TYPE_MISMATCH", "tool": tool, "param": k, "pred": p, "gold": g})
            elif p != g:
                kind = "CASING_ONLY" if str(p).strip().lower() == str(g).strip().lower() else "VALUE_MISMATCH"
                out.append({"type": kind, "tool": tool, "param": k, "pred": p, "gold": g})
    return out


def audit_sample(
    sample: Mapping[str, Any], pred: Sequence[Mapping[str, Any]], tool_kind: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Differences between the predicted calls and the gold calls of one labelled sample."""
    gold = list((sample.get("label") or {}).get("tool_calls") or [])
    if not gold and pred:
        return [{"type": "CALLED_ON_EMPTY_LABEL", "tools": [p.get("name") for p in pred]}]
    issues: list[dict[str, Any]] = []
    pairs, missing, extra = _pair(pred, gold)
    for gi in missing:
        issues.append({"type": "MISSING_CALL", "tool": gold[gi].get("name")})
    for pi in extra:
        name = str(pred[pi].get("name"))
        redundant = tool_kind.get(name) == "read" and _history_has_results(sample)
        issues.append({"type": "REDUNDANT_READ" if redundant else "EXTRA_CALL", "tool": name})
    for p, g in pairs:
        if g.get("name") != ASK_TOOL:
            issues += _arg_issues(str(g.get("name")), p.get("arguments") or {}, g.get("arguments") or {})
    return issues


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts by issue type, by (type, tool) and by (type, tool, param), plus samples with no issue."""
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
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path, help="submission.jsonl of a run on a labelled split")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding the split folders")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, help="only these ids (default: every id in the submission)")
    ap.add_argument("--out", type=Path, help="write one jsonl row per sample with its issues")
    args = ap.parse_args(argv)

    with open(args.root / args.split / f"{args.split}.jsonl", encoding="utf-8") as f:
        samples = {s["id"]: s for s in map(json.loads, filter(str.strip, f))}
    with open(args.root / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8") as f:
        tool_kind = {t["name"]: t["kind"] for app in json.load(f)["apps"] for t in app["tools"]}
    with open(args.submission, encoding="utf-8") as f:
        preds = {r["id"]: r.get("tool_calls") or [] for r in map(json.loads, filter(str.strip, f))}
    ids = read_ids(args.ids) if args.ids else sorted(preds)
    rows = [{"id": i, "issues": audit_sample(samples[i], preds.get(i, []), tool_kind)} for i in ids]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

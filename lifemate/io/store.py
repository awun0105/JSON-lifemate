"""Run directory: trace, submission, summary, config, resume, atomic writes (SPEC 4.11, 4.12)."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import SampleResult, StopReason

log = structlog.get_logger(__name__)

MAX_ERROR_IDS = 10
OVER_S = (30.0, 25.0)
OVER_TOKENS = 14_000
SHRINK_LEVELS = 4
INDEX_FIELDS = ("run", "date", "git_commit", "model", "prompt_hash", "split", "ids_file", "n", "score", "p95_seconds")


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via `<path>.tmp` + fsync + os.replace; the tmp file is removed on failure."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (`q` in [0, 1]); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[k]


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """SPEC 4.12 summary fields from trace records; `score` stays None until scoring."""
    recs = list(records)
    stop = {r.value: 0 for r in StopReason}
    shrink = {str(i): 0 for i in range(SHRINK_LEVELS)}
    errors: dict[str, dict[str, Any]] = {}
    seconds: list[float] = []
    max_prompts: list[int] = []
    for r in recs:
        reason = str(r.get("stop_reason"))
        stop[reason] = stop.get(reason, 0) + 1
        level = str(r.get("prompt_shrink_level", 0))
        shrink[level] = shrink.get(level, 0) + 1
        for code in r.get("errors") or []:
            entry = errors.setdefault(str(code), {"count": 0, "ids": []})
            entry["count"] += 1
            if len(entry["ids"]) < MAX_ERROR_IDS:
                entry["ids"].append(r["id"])
        seconds.append(float((r.get("timing") or {}).get("total_s", 0.0)))
        max_prompts.append(int((r.get("tokens") or {}).get("max_prompt", 0)))
    return {
        "n": len(recs),
        "n_exception": stop[StopReason.EXCEPTION.value],
        "stop_reason": stop,
        "prompt_shrink_level": shrink,
        "errors": errors,
        "seconds": {
            "p50": round(percentile(seconds, 0.5), 3),
            "p95": round(percentile(seconds, 0.95), 3),
            "max": round(max(seconds, default=0.0), 3),
            "over_30": sum(s > OVER_S[0] for s in seconds),
            "over_25": sum(s > OVER_S[1] for s in seconds),
        },
        "tokens": {"max_prompt": max(max_prompts, default=0), "over_14k": sum(t > OVER_TOKENS for t in max_prompts)},
        "score": None,
    }


def append_index(index_csv: Path, row: Mapping[str, Any]) -> None:
    """Append one row to artifacts/runs/index.csv, writing the header when the file is new."""
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not index_csv.exists()
    with index_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


class RunStore:
    """Reads and writes one run directory; never runs the agent."""

    def __init__(self, run_dir: Path, *, resume: bool) -> None:
        self.run_dir = Path(run_dir)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.submission_path = self.run_dir / "submission.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        if self.trace_path.exists() and not resume:
            raise FileExistsError(f"{self.trace_path} exists; pass --resume to continue or pick another --name/--out")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume and self.trace_path.exists():
            self._drop_corrupt_tail()

    def _drop_corrupt_tail(self) -> None:
        """Remove a trailing line cut by an interrupted write so later appends start on a clean line."""
        text = self.trace_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines:
            return
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            log.warning("trace_corrupt_tail_dropped", path=str(self.trace_path))
            kept = lines[:-1]
            atomic_write_text(self.trace_path, "".join(ln + "\n" for ln in kept))
            return
        if not text.endswith("\n"):
            atomic_write_text(self.trace_path, text + "\n")

    def records(self) -> list[dict[str, Any]]:
        """Parsed trace lines; a corrupt last line is skipped, a corrupt middle line raises ValueError."""
        if not self.trace_path.exists():
            return []
        lines = [ln for ln in self.trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                if i == len(lines) - 1:
                    log.warning("trace_corrupt_tail_skipped", path=str(self.trace_path))
                    break
                raise ValueError(f"{self.trace_path}: corrupt line {i + 1}") from e
        return out

    def done_ids(self) -> set[str]:
        """Ids already in trace.jsonl."""
        return {str(r["id"]) for r in self.records()}

    def append(self, result: SampleResult) -> None:
        """Append one sample to trace.jsonl, flushed and fsynced."""
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(result.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Write config.json atomically."""
        atomic_write_text(self.config_path, json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def write_ids(self, ids: Sequence[str]) -> Path:
        """Write the ids of this run (one per line) to ids.txt; returns its path."""
        path = self.run_dir / "ids.txt"
        atomic_write_text(path, "".join(i + "\n" for i in ids))
        return path

    def finalize(self, split_ids: Sequence[str]) -> Path:
        """Rebuild submission.jsonl from the trace in split order, write summary.json; return the submission."""
        by_id = {str(r["id"]): r for r in self.records()}
        lines: list[str] = []
        missing: list[str] = []
        for sid in split_ids:
            r = by_id.get(sid)
            if r is None:
                missing.append(sid)
                continue
            row = {"id": sid, "tool_calls": r.get("tool_calls") or [], "state_hash": r.get("state_hash") or ""}
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
        if missing:
            log.error("submission_missing_ids", n=len(missing), first=missing[:5])
        atomic_write_text(self.submission_path, "".join(lines))
        wanted = set(split_ids)
        self.write_summary(summarize(r for sid, r in by_id.items() if sid in wanted))
        return self.submission_path

    def read_summary(self) -> dict[str, Any]:
        """Parsed summary.json."""
        data: dict[str, Any] = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return data

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Write summary.json atomically."""
        atomic_write_text(self.summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

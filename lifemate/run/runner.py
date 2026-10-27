"""Runner: warm-up, per-sample loop with the only `except Exception`, finalize, score or check (SPEC 4.11)."""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from lifemate.agent.agent import Agent, transcript_source
from lifemate.core.types import ErrorCode, Sample, SampleResult, StopReason, ToolEnvLike
from lifemate.io.check_submission import check_submission
from lifemate.io.store import RunStore, append_index

log = structlog.get_logger(__name__)

_FINAL_RE = re.compile(r"^Final score(?: \[(\w+)\])?: ([\d.]+|nan)", re.MULTILINE)
_CATEGORY_RE = re.compile(r"^(execute|ask_missing_info|out_of_scope)\s+(\d+)\s+([\d.]+|nan)", re.MULTILINE)


class EnvLoader(Protocol):
    """What the Runner needs from `ZaicLoader`."""

    root: Path
    split: str

    def open_env(self, sample_id: str) -> ToolEnvLike: ...


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Where the outputs of one run went and the headline numbers."""

    run_dir: Path
    submission: Path
    summary: dict[str, Any]
    score: dict[str, Any] | None
    report: Path | None


def parse_score(text: str) -> dict[str, Any] | None:
    """`Final score` lines (one per match mode with `--match both`) and per-category scores of score.py output."""
    finals = _FINAL_RE.findall(text)
    if not finals:
        return None
    modes = {(mode or "strict"): float(val) for mode, val in finals}
    by_category = {name: float(val) for name, _n, val in _CATEGORY_RE.findall(text)}
    return {"total": modes.get("strict", next(iter(modes.values()))), "by_mode": modes, "by_category": by_category}


class Runner:
    """Runs every sample once, isolates per-sample failures and writes the run directory; no agent logic."""

    def __init__(
        self,
        agent: Agent,
        loader: EnvLoader,
        store: RunStore,
        *,
        repo_root: Path,
        index_csv: Path | None = None,
        score_script: Path | None = None,
        train_mode: bool = False,
    ) -> None:
        self.agent = agent
        self.loader = loader
        self.store = store
        self.repo_root = Path(repo_root)
        self.index_csv = index_csv
        self.score_script = score_script  # None: run `python -m lifemate.eval.score` from repo_root
        self.train_mode = train_mode

    def warmup(self) -> dict[str, float]:
        """Warm up the LLM and the transcriber; returns seconds of each."""
        t0 = time.perf_counter()
        self.agent.llm.warmup()
        t1 = time.perf_counter()
        self.agent.transcriber.warmup()
        t2 = time.perf_counter()
        log.info("warmup_done", llm_s=round(t1 - t0, 3), transcriber_s=round(t2 - t1, 3))
        return {"llm_s": t1 - t0, "transcriber_s": t2 - t1}

    def run_one(self, sample: Sample) -> SampleResult:
        """Run one sample; any Exception becomes an `exception` result (the only catch-all in lifemate)."""
        env: ToolEnvLike | None = None
        t0 = time.monotonic()
        try:
            env = self.loader.open_env(sample.id)
            result = self.agent.run(sample, env, train_mode=self.train_mode)
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("sample_exception", sample_id=sample.id, error=repr(exc), traceback=tb)
            calls: tuple[dict[str, Any], ...] = ()
            state_hash = ""
            if env is not None:
                calls = tuple(env.get_tool_calls())
                state_hash = env.get_state_hash()
            result = SampleResult(
                id=sample.id,
                stop_reason=StopReason.EXCEPTION,
                errors=(ErrorCode.EXCEPTION,),
                transcript=None,
                steps=(),
                tool_calls=calls,
                state_hash=state_hash,
                timing={"total_s": time.monotonic() - t0},
                transcript_source=transcript_source(self.agent.transcriber),
                traceback=tb,
            )
        return result

    def run(
        self,
        samples: Sequence[Sample],
        *,
        ids_file: Path | None,
        split_has_labels: bool,
        config_record: Mapping[str, Any],
    ) -> RunSummary:
        """Warm up, run all samples not yet in the trace, finalize, then score (labels) or check (no labels)."""
        record = dict(config_record)
        record["warmup_s"] = self.warmup()
        self.store.write_config(record)

        ordered = sorted(samples, key=lambda s: s.id)
        done = self.store.done_ids()
        if done:
            log.info("resume_skip", n_done=len(done))
        for sample in ordered:
            if sample.id in done:
                continue
            result = self.run_one(sample)
            self.store.append(result)
            log.info("sample_done", sample_id=sample.id, stop_reason=result.stop_reason.value)

        split_ids = [s.id for s in samples]
        submission = self.store.finalize(split_ids)
        summary = self.store.read_summary()
        score: dict[str, Any] | None = None
        report: Path | None
        if split_has_labels:
            report, score = self._score(submission, self.store.write_ids(split_ids))
            summary["score"] = score
            self.store.write_summary(summary)
        else:
            report = self._check(submission)
        if self.index_csv is not None:
            append_index(
                self.index_csv,
                {
                    "run": self.store.run_dir.name,
                    "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "git_commit": record.get("env_info", {}).get("code_version", ""),
                    "model": record.get("config", {}).get("llm", {}).get("model", ""),
                    "prompt_hash": record.get("prompt_sha256", ""),
                    "split": self.loader.split,
                    "ids_file": str(ids_file) if ids_file is not None else "",
                    "n": summary["n"],
                    "score": "" if score is None else score["total"],
                    "p95_seconds": summary["seconds"]["p95"],
                },
            )
        return RunSummary(self.store.run_dir, submission, summary, score, report)

    def _score(self, submission: Path, ids_path: Path) -> tuple[Path, dict[str, Any] | None]:
        """Score in a subprocess (importing score.py here would load zaic_tools a second time)."""
        run_dir = self.store.run_dir
        scorer = [str(self.score_script)] if self.score_script is not None else ["-m", "lifemate.eval.score"]
        cmd = [
            sys.executable,
            *scorer,
            str(submission.resolve()),
            "--root",
            str(self.loader.root.resolve()),
            "--split",
            self.loader.split,
            "--ids",
            str(ids_path.resolve()),
            "--match",
            "both",
            "--details",
            str((run_dir / "details.jsonl").resolve()),
        ]
        proc = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True, check=False)
        out = run_dir / "score.txt"
        text = proc.stdout + (("\n== stderr\n" + proc.stderr) if proc.returncode != 0 else "")
        out.write_text(text, encoding="utf-8")
        if proc.returncode != 0:
            log.error("score_failed", returncode=proc.returncode, stderr=proc.stderr[-2000:])
        score = parse_score(proc.stdout)
        log.info("score_done", score=None if score is None else score["by_mode"])
        return out, score

    def _check(self, submission: Path) -> Path:
        """Run check_submission and write check.txt."""
        report = check_submission(submission, self.loader.root, self.loader.split, self.agent.index)
        out = self.store.run_dir / "check.txt"
        out.write_text(report.to_text(), encoding="utf-8")
        return out

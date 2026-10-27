"""Runner: khởi động, vòng lặp từng mẫu với `except Exception` duy nhất, chốt kết quả, chấm hoặc kiểm tra (SPEC 4.11).

Cách xây dựng:
- `Runner` không chứa logic agent: chỉ khởi động mô hình, chạy từng mẫu một lần theo thứ tự id, ghi kết quả qua
  `RunStore` và chốt file nộp bài.
- Ranh giới ngoại lệ theo mẫu: `run_one` là nơi duy nhất trong lifemate bắt mọi Exception; lỗi của một mẫu thành
  kết quả `exception` (kèm traceback, các lời gọi tool đã thực hiện và hash trạng thái) thay vì làm dừng cả run.
- Chạy tiếp (resume): mẫu đã có trong trace được bỏ qua.
- Split có nhãn thì chấm điểm bằng `lifemate.eval.score` trong tiến trình con và phân tích dòng `Final score`;
  split không nhãn thì chạy `check_submission`. Cuối cùng ghi một dòng vào index.csv.
"""

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

# Dòng tổng điểm (có thể kèm chế độ so khớp) và dòng điểm theo nhóm trong đầu ra của score.py
_FINAL_RE = re.compile(r"^Final score(?: \[(\w+)\])?: ([\d.]+|nan)", re.MULTILINE)
_CATEGORY_RE = re.compile(r"^(execute|ask_missing_info|out_of_scope)\s+(\d+)\s+([\d.]+|nan)", re.MULTILINE)


class EnvLoader(Protocol):
    """Phần giao diện của `ZaicLoader` mà Runner cần."""

    root: Path
    split: str

    def open_env(self, sample_id: str) -> ToolEnvLike: ...


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Vị trí các đầu ra của một run và các con số chính."""

    run_dir: Path
    submission: Path
    summary: dict[str, Any]
    score: dict[str, Any] | None
    report: Path | None


def parse_score(text: str) -> dict[str, Any] | None:
    """Đọc dòng `Final score` (mỗi chế độ so khớp một dòng khi dùng `--match both`) và điểm theo nhóm của score.py.

    Trả None khi không có dòng tổng điểm nào. `total` là điểm chế độ strict, hoặc chế độ đầu tiên nếu thiếu strict.
    """
    finals = _FINAL_RE.findall(text)
    if not finals:
        return None
    # Dòng không ghi chế độ được hiểu là strict
    modes = {(mode or "strict"): float(val) for mode, val in finals}
    by_category = {name: float(val) for name, _n, val in _CATEGORY_RE.findall(text)}
    return {"total": modes.get("strict", next(iter(modes.values()))), "by_mode": modes, "by_category": by_category}


class Runner:
    """Chạy mỗi mẫu một lần, cô lập lỗi theo từng mẫu và ghi thư mục run; không chứa logic agent."""

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
        self.score_script = score_script  # None: chạy `python -m lifemate.eval.score` từ repo_root
        self.train_mode = train_mode

    def warmup(self) -> dict[str, float]:
        """Khởi động LLM và transcriber; trả số giây của từng bước."""
        t0 = time.perf_counter()
        self.agent.llm.warmup()
        t1 = time.perf_counter()
        self.agent.transcriber.warmup()
        t2 = time.perf_counter()
        log.info("warmup_done", llm_s=round(t1 - t0, 3), transcriber_s=round(t2 - t1, 3))
        return {"llm_s": t1 - t0, "transcriber_s": t2 - t1}

    def run_one(self, sample: Sample) -> SampleResult:
        """Chạy một mẫu; mọi Exception thành kết quả `exception` (chỗ bắt-tất-cả duy nhất trong lifemate).

        Khi môi trường tool đã mở, kết quả lỗi vẫn giữ các lời gọi tool đã thực hiện và hash trạng thái, để bộ chấm
        điểm đánh giá đúng những gì đã xảy ra trước lỗi.
        """
        env: ToolEnvLike | None = None
        t0 = time.monotonic()
        try:
            env = self.loader.open_env(sample.id)
            result = self.agent.run(sample, env, train_mode=self.train_mode)
        except Exception as exc:  # ranh giới ngoại lệ theo mẫu: một mẫu lỗi không làm dừng cả run
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
        """Khởi động, chạy mọi mẫu chưa có trong trace, chốt kết quả, rồi chấm (có nhãn) hoặc kiểm tra (không nhãn).

        Bản ghi cấu hình (kèm thời gian khởi động) được ghi trước vòng lặp. Mỗi kết quả được ghi nối ngay sau khi
        mẫu xong, nên run bị ngắt có thể chạy tiếp bằng `--resume`.
        """
        record = dict(config_record)
        record["warmup_s"] = self.warmup()
        self.store.write_config(record)

        # Thứ tự theo id cố định, để hai lần chạy cùng tập mẫu cho cùng trình tự
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
        """Chấm điểm trong tiến trình con (import score.py ở đây sẽ nạp zaic_tools lần thứ hai).

        Đầu ra chuẩn được ghi vào score.txt (kèm stderr khi lỗi); chi tiết từng mẫu ghi vào details.jsonl.
        """
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
        """Chạy check_submission và ghi check.txt."""
        report = check_submission(submission, self.loader.root, self.loader.split, self.agent.index)
        out = self.store.run_dir / "check.txt"
        out.write_text(report.to_text(), encoding="utf-8")
        return out

"""Đọc và ghi thư mục của một lần chạy (run): trace, submission, summary, config, chạy tiếp (SPEC 4.11, 4.12).

Nội dung chính:
- `RunStore`: đọc và ghi một thư mục run (`trace.jsonl`, `submission.jsonl`, `summary.json`, `config.json`,
  `ids.txt`).
- `atomic_write_text`: ghi file nguyên tử, không bao giờ để lại file ghi dở.
- `percentile`, `summarize`: tính thống kê cho `summary.json`.
- `append_index`: nối một dòng vào bảng tổng hợp các run.

Cách hoạt động:
- `trace.jsonl` là nguồn dữ liệu gốc duy nhất. Mỗi mẫu xong được nối thêm một dòng, có flush và fsync, nên lần
  chạy bị ngắt vẫn giữ được các mẫu đã xong. `submission.jsonl` và `summary.json` luôn được dựng lại từ trace.
- Chạy tiếp (`resume`) bỏ qua các id đã có trong trace; dòng cuối bị cắt dở do ghi gián đoạn được loại bỏ.
- Khi chạy tiếp, config mới phải khớp config cũ ở các trường `RESUME_KEYS` (`check_resume`), và `config.json` giữ
  bản ghi gốc, chỉ nối thêm thông tin của lần chạy tiếp vào `resumes` (`resumed_config`).
- Các file còn lại được ghi qua file `.tmp` rồi `os.replace`, nên không ai đọc phải file đang ghi dở.

Lưu ý:
- `RunStore` không chạy agent, chỉ lo phần lưu trữ.
"""

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
# Ngưỡng (giây) để đếm mẫu chậm: 30 giây là giới hạn của cuộc thi, 25 giây là mức cảnh báo.
OVER_S = (30.0, 25.0)
OVER_TOKENS = 14_000
# Các trường của bản ghi config quyết định kết quả từng mẫu; chạy tiếp mà khác ở đây thì trace bị trộn hai cấu hình.
# Không so `dtype` đã chọn hay thông tin máy: lần chạy tiếp trên GPU khác vẫn hợp lệ, nhất là khi container nộp bài.
RESUME_KEYS = ("config", "prompt_sha256", "split", "train_mode")
# Thông tin riêng của từng lần chạy tiếp, nối vào `config.json["resumes"]`.
_RESUME_INFO_KEYS = ("started_at", "warmup_s", "env_info", "dtype")
SHRINK_LEVELS = 4
INDEX_FIELDS = ("run", "date", "git_commit", "model", "prompt_hash", "split", "ids_file", "n", "score", "p95_seconds")


def atomic_write_text(path: Path, text: str) -> None:
    """Ghi `text` vào `path` một cách nguyên tử: ghi `<path>.tmp`, fsync, rồi `os.replace`.

    `os.replace` là thao tác nguyên tử trên cùng hệ thống file, nên `path` hoặc giữ nội dung cũ, hoặc có đủ nội dung
    mới. Khi có lỗi, file tmp bị xoá và lỗi vẫn được ném tiếp ra ngoài.

    Tham số:
        path: file đích.
        text: nội dung cần ghi, mã hoá UTF-8.
    """
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
    """Tính phân vị theo hạng gần nhất.

    Tham số:
        values: dãy số cần tính.
        q: mức phân vị, trong khoảng [0, 1]; ví dụ 0.95 cho p95.

    Trả về:
        Giá trị phân vị; 0.0 khi `values` rỗng.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # Kẹp chỉ số vào [0, n-1] để q = 0 hoặc q = 1 không vượt biên.
    k = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[k]


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Tính các trường của `summary.json` (SPEC 4.12) từ các bản ghi trace.

    Mọi `StopReason` và mọi mức rút gọn prompt đều có mặt, kể cả khi bằng 0. Nhờ vậy summary của các run có cùng
    khuôn dạng và so sánh được với nhau. Mỗi mã lỗi chỉ giữ tối đa `MAX_ERROR_IDS` id mẫu làm ví dụ.

    Tham số:
        records: các bản ghi trace đã parse.

    Trả về:
        Dict summary. Trường `score` để `None` cho tới khi chấm điểm.
    """
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
    """Nối một dòng vào `artifacts/runs/index.csv`; ghi thêm dòng tiêu đề khi file mới được tạo.

    Chỉ giữ các cột trong `INDEX_FIELDS`: khoá thừa bị bỏ, khoá thiếu thành chuỗi rỗng.

    Tham số:
        index_csv: đường dẫn file CSV.
        row: dữ liệu của dòng mới.
    """
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not index_csv.exists()
    with index_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in INDEX_FIELDS})


class RunStore:
    """Đọc và ghi một thư mục run; không bao giờ chạy agent.

    Tham số:
        run_dir: thư mục run; được tạo nếu chưa có.
        resume: `True` để chạy tiếp trên trace sẵn có.

    Thuộc tính:
        run_dir: thư mục run.
        trace_path: đường dẫn `trace.jsonl`.
        submission_path: đường dẫn `submission.jsonl`.
        summary_path: đường dẫn `summary.json`.
        config_path: đường dẫn `config.json`.

    Lỗi:
        FileExistsError: khi `trace.jsonl` đã có mà không bật `resume`, để không ghi đè run cũ.
    """

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
        """Bỏ dòng cuối bị cắt dở do ghi gián đoạn, để các lần nối sau bắt đầu trên một dòng sạch."""
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
        # Dòng cuối hợp lệ nhưng thiếu ký tự xuống dòng: thêm vào, để lần nối sau không dính liền vào dòng này.
        if not text.endswith("\n"):
            atomic_write_text(self.trace_path, text + "\n")

    def records(self) -> list[dict[str, Any]]:
        """Đọc và parse các dòng của `trace.jsonl`.

        Dòng cuối hỏng là dấu hiệu bình thường của lần chạy bị ngắt, nên được bỏ qua. Dòng hỏng ở giữa nghĩa là file bị
        hư thật, nên báo lỗi.

        Trả về:
            Danh sách bản ghi; danh sách rỗng khi chưa có trace.

        Lỗi:
            ValueError: khi có dòng hỏng không phải dòng cuối.
        """
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
        """Trả về các id đã có trong `trace.jsonl`, để bỏ qua khi chạy tiếp."""
        return {str(r["id"]) for r in self.records()}

    def append(self, result: SampleResult) -> None:
        """Nối kết quả một mẫu vào `trace.jsonl`, có flush và fsync."""
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(canonical_json(result.to_dict()) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Ghi `config.json` theo cách nguyên tử."""
        atomic_write_text(self.config_path, json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _old_config(self) -> dict[str, Any] | None:
        """Đọc `config.json` của run đang chạy tiếp; `None` khi chưa có trace hoặc chưa có config (run mới)."""
        if not (self.trace_path.exists() and self.config_path.exists()):
            return None
        return dict(json.loads(self.config_path.read_text(encoding="utf-8")))

    def check_resume(self, record: Mapping[str, Any]) -> None:
        """Báo lỗi khi chạy tiếp một run bằng cấu hình khác với cấu hình đã sinh ra trace.

        Chỉ so các trường trong `RESUME_KEYS` mà config cũ có, để run tạo bởi phiên bản cũ vẫn chạy tiếp được.

        Tham số:
            record: bản ghi config của lần chạy này (cùng dạng với `config.json`).

        Lỗi:
            ValueError: khi một trường trong `RESUME_KEYS` khác với config cũ.
        """
        old = self._old_config()
        if old is None:
            return
        # Đi qua JSON để tuple và list (hoặc kiểu số) được so như nhau, giống cách chúng nằm trong `config.json`.
        new = json.loads(json.dumps({k: record.get(k) for k in RESUME_KEYS}, ensure_ascii=False))
        diff = sorted(k for k in RESUME_KEYS if k in old and old[k] != new[k])
        if diff:
            raise ValueError(
                f"{self.run_dir}: --resume with a different {', '.join(diff)} than the run's config.json; "
                "use another --name/--out for a new configuration"
            )

    def resumed_config(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Trả về bản ghi cần ghi vào `config.json`.

        Run mới thì là chính `record`. Chạy tiếp thì giữ bản ghi gốc (cấu hình đã sinh ra trace, thời điểm bắt đầu) và
        nối thông tin của lần này (`_RESUME_INFO_KEYS`) vào danh sách `resumes`.
        """
        old = self._old_config()
        if old is None:
            return dict(record)
        info = {k: record[k] for k in _RESUME_INFO_KEYS if k in record}
        return {**old, "resumes": [*old.get("resumes", []), info]}

    def write_ids(self, ids: Sequence[str]) -> Path:
        """Ghi các id của lần chạy này vào `ids.txt`, mỗi dòng một id, và trả về đường dẫn file."""
        path = self.run_dir / "ids.txt"
        atomic_write_text(path, "".join(i + "\n" for i in ids))
        return path

    def finalize(self, split_ids: Sequence[str]) -> Path:
        """Dựng lại `submission.jsonl` từ trace theo thứ tự của split, rồi ghi `summary.json`.

        Id trong `split_ids` chưa có trong trace bị bỏ khỏi submission và được ghi log lỗi. Summary chỉ tính các id
        thuộc `split_ids`, kể cả khi trace còn bản ghi của id khác.

        Tham số:
            split_ids: các id của split, theo thứ tự cần ghi.

        Trả về:
            Đường dẫn file submission.
        """
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
        """Đọc và parse `summary.json`."""
        data: dict[str, Any] = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return data

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        """Ghi `summary.json` theo cách nguyên tử."""
        atomic_write_text(self.summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

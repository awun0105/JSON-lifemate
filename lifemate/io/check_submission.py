"""Kiểm tra một file submission cho split không có nhãn trước khi nộp (SPEC 4.11, B3).

Nội dung chính:
- `CheckReport`: kết quả kiểm tra, gồm lỗi (chặn việc nộp) và cảnh báo (chỉ cần xem lại).
- `check_submission`: chạy toàn bộ các bước kiểm tra B3.

Cách hoạt động:
1. Đọc từng dòng JSON. Dòng hỏng được ghi lại; với id trùng thì chỉ giữ dòng đầu.
2. So id với split: thiếu, trùng hoặc lạ đều là lỗi (`errors`), tức file không nộp được.
3. Tên tool không có trong catalog chỉ là cảnh báo (`warnings`).
4. Chạy `zaic_tools --validate` trong tiến trình con, với `PYTHONPATH` trỏ vào thư mục split.
5. So phân bố (tỉ lệ câu trả lời rỗng, tỉ lệ có lời gọi hỏi lại, số lời gọi trung bình) với tập train; lệch quá
   `MAX_RATIO` lần thì cảnh báo.
6. Phát lại mọi lời gọi trong một `ToolEnv` mới và đếm số kết quả `ok:false`.

Lưu ý:
- Mọi bước đều chạy hết, kể cả khi bước trước đã có lỗi, để báo cáo đầy đủ trong một lần.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import structlog

from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.io.data import load_split
from lifemate.io.zaic import EnvRegistry, load_catalog, load_zaic, open_env

log = structlog.get_logger(__name__)

# Phân bố tham chiếu của tập train (SPEC 4.11 B3): tỉ lệ câu trả lời rỗng, tỉ lệ có lời gọi hỏi lại, và số lời
# gọi trung bình.
TRAIN_EMPTY_RATE = 0.07
TRAIN_ASK_RATE = 0.14
TRAIN_MEAN_CALLS = 2.3
MAX_RATIO = 2.0
MAX_LISTED_IDS = 20
_VALIDATE_RE = re.compile(r"(\d+) lines checked, (\d+) schema errors")


@dataclass(slots=True)
class CheckReport:
    """Kết quả của `check_submission`: có `errors` thì file không nộp được, còn `warnings` chỉ cần xem lại.

    Thuộc tính:
        split: tên split được kiểm tra.
        n_expected: số mẫu của split.
        n_lines: số dòng không trống trong file submission.
        errors: các lỗi chặn việc nộp.
        warnings: các cảnh báo cần xem lại.
        missing_ids: id của split không có trong file.
        duplicate_ids: id xuất hiện nhiều lần.
        unknown_ids: id trong file nhưng không thuộc split.
        invalid_lines: số thứ tự (từ 1) của các dòng không hợp lệ.
        validate_output: stdout và stderr của `zaic_tools --validate`.
        validate_errors: số lỗi schema mà `--validate` báo; `None` khi không đọc được.
        stats: phân bố lời gọi của submission (`empty_rate`, `ask_rate`, `mean_calls`).
        rejected: id mẫu tới số lời gọi trả `ok:false` khi phát lại.
    """

    split: str
    n_expected: int
    n_lines: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    invalid_lines: list[int] = field(default_factory=list)
    validate_output: str = ""
    validate_errors: int | None = None
    stats: dict[str, float] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """`True` khi không có lỗi nào."""
        return not self.errors

    def to_text(self) -> str:
        """Trả về báo cáo dạng văn bản dễ đọc, dùng để ghi `check.txt`."""
        out = [f"check_submission split={self.split} expected={self.n_expected} lines={self.n_lines}"]
        out.append("status: " + ("OK" if self.ok else "FAIL"))
        out += [f"ERROR: {e}" for e in self.errors]
        out += [f"WARNING: {w}" for w in self.warnings]
        if self.stats:
            out.append("distribution: " + ", ".join(f"{k}={v:.3f}" for k, v in self.stats.items()))
        if self.validate_output:
            out.append("== zaic_tools --validate")
            out.append(self.validate_output.rstrip())
        return "\n".join(out) + "\n"


def _ids_text(ids: list[str]) -> str:
    """Nối danh sách id để in: chỉ liệt kê tối đa `MAX_LISTED_IDS` id đầu, kèm số id còn lại."""
    more = f" (+{len(ids) - MAX_LISTED_IDS} more)" if len(ids) > MAX_LISTED_IDS else ""
    return ", ".join(ids[:MAX_LISTED_IDS]) + more


def _run_validate(report: CheckReport, submission: Path, root: Path, split: str) -> None:
    """Chạy `zaic_tools --validate` trên `submission` và ghi kết quả vào `report`.

    Số lỗi schema được đọc từ stdout bằng `_VALIDATE_RE`. Lệnh không chạy được hoặc kết thúc bất thường chỉ thành
    cảnh báo; có lỗi schema thì thành lỗi.

    Tham số:
        report: báo cáo cần ghi thêm kết quả.
        submission: file submission cần kiểm tra.
        root: thư mục dữ liệu.
        split: tên split.
    """
    split_dir = (root / split).resolve()
    env = dict(os.environ)
    # Mỗi split có bản `zaic_tools` riêng: đặt thư mục split lên đầu `PYTHONPATH` để import đúng bản.
    env["PYTHONPATH"] = str(split_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        "-m",
        "zaic_tools",
        "--split",
        split,
        "--root",
        str(root.resolve()),
        "--validate",
        str(submission.resolve()),
    ]
    try:
        proc = subprocess.run(cmd, cwd=split_dir, env=env, capture_output=True, text=True, check=False)
    except OSError as e:
        report.warnings.append(f"zaic_tools --validate could not run: {e}")
        return
    report.validate_output = proc.stdout + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    match = _VALIDATE_RE.search(proc.stdout)
    # Không thấy dòng tổng kết thì không biết số lỗi schema, nên chỉ cảnh báo chứ không kết luận.
    if proc.returncode != 0 or match is None:
        report.warnings.append(f"zaic_tools --validate did not finish normally (exit {proc.returncode})")
        return
    report.validate_errors = int(match.group(2))
    if report.validate_errors:
        report.errors.append(f"zaic_tools --validate: {report.validate_errors} schema errors")


def _distribution(report: CheckReport, rows: list[dict[str, Any]]) -> None:
    """Tính phân bố lời gọi của `rows` và cảnh báo khi lệch quá `MAX_RATIO` lần so với tập train."""
    if not rows:
        return
    calls = [r.get("tool_calls") or [] for r in rows]
    stats = {
        "empty_rate": sum(not c for c in calls) / len(calls),
        "ask_rate": sum(any(x.get("name") == ASK_TOOL for x in c) for c in calls) / len(calls),
        "mean_calls": sum(len(c) for c in calls) / len(calls),
    }
    report.stats = stats
    refs = {"empty_rate": TRAIN_EMPTY_RATE, "ask_rate": TRAIN_ASK_RATE, "mean_calls": TRAIN_MEAN_CALLS}
    for key, ref in refs.items():
        val = stats[key]
        # Đo độ lệch theo tỉ lệ (gấp hoặc chia `MAX_RATIO`), không theo hiệu số, vì các chỉ số có thang đo khác nhau.
        if val > ref * MAX_RATIO or val < ref / MAX_RATIO:
            report.warnings.append(f"{key}={val:.3f} is more than {MAX_RATIO:g}x off the train reference {ref}")


def _replay(report: CheckReport, rows: list[dict[str, Any]], zaic: ModuleType, root: Path, split: str) -> None:
    """Phát lại mọi lời gọi trong một `ToolEnv` mới và đếm số kết quả `ok:false`.

    Dùng `EnvRegistry` riêng, tách khỏi registry của lần chạy. Số lời gọi bị từ chối theo từng id được ghi vào
    `report.rejected`, rồi gộp thành một cảnh báo.

    Tham số:
        report: báo cáo cần ghi thêm kết quả.
        rows: các dòng submission có id thuộc split.
        zaic: module `zaic_tools` đã nạp.
        root: thư mục dữ liệu.
        split: tên split.
    """
    registry = EnvRegistry()
    for row in rows:
        env = open_env(zaic, root, split, str(row["id"]), registry=registry)
        bad = 0
        for call in row.get("tool_calls") or []:
            result = env.call(str(call.get("name")), call.get("arguments") or {})
            bad += not result.get("ok", False)
        if bad:
            report.rejected[str(row["id"])] = bad
    if report.rejected:
        ids = sorted(report.rejected)
        report.warnings.append(
            f"replay: {sum(report.rejected.values())} calls returned ok:false in {len(ids)} samples: {_ids_text(ids)}"
        )


def _read_rows(report: CheckReport, submission: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Đọc file submission, giữ dòng đầu tiên của mỗi id và đếm số lần mỗi id xuất hiện.

    Dòng không phải JSON, không phải object, thiếu `id` hoặc có `tool_calls` không phải list được ghi số dòng vào
    `report.invalid_lines`. Dòng trống bị bỏ qua và không tính vào `report.n_lines`.

    Tham số:
        report: báo cáo cần ghi thêm kết quả.
        submission: file submission.

    Trả về:
        Cặp (các dòng hợp lệ đầu tiên của mỗi id, số lần xuất hiện của mỗi id).
    """
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for n, line in enumerate(submission.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        report.n_lines += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            report.invalid_lines.append(n)
            continue
        if not isinstance(row, dict) or "id" not in row or not isinstance(row.get("tool_calls", []), list):
            report.invalid_lines.append(n)
            continue
        sid = str(row["id"])
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] == 1:
            rows.append(row)
    return rows, seen


def check_submission(
    submission_path: Path,
    root: Path,
    split: str,
    index: CatalogIndex | None = None,
    *,
    zaic: ModuleType | None = None,
) -> CheckReport:
    """Chạy các bước kiểm tra B3: id, JSON, `zaic_tools --validate`, phân bố, phát lại.

    Tham số:
        submission_path: file `submission.jsonl` cần kiểm tra.
        root: thư mục dữ liệu chứa `<split>/`.
        split: tên split.
        index: catalog tool đã dựng sẵn; `None` thì đọc từ catalog của split.
        zaic: module `zaic_tools` đã import; `None` thì import từ thư mục split.

    Trả về:
        `CheckReport`; `report.ok` là `False` khi có lỗi chặn việc nộp.
    """
    root = Path(root)
    expected = [s.id for s in load_split(root, split)]
    report = CheckReport(split=split, n_expected=len(expected))
    rows, seen = _read_rows(report, Path(submission_path))
    expected_set = set(expected)
    report.missing_ids = [i for i in expected if i not in seen]
    report.duplicate_ids = sorted(i for i, c in seen.items() if c > 1)
    report.unknown_ids = sorted(i for i in seen if i not in expected_set)
    if report.invalid_lines:
        report.errors.append(f"invalid JSON lines: {report.invalid_lines[:MAX_LISTED_IDS]}")
    if report.missing_ids:
        report.errors.append(f"{len(report.missing_ids)} missing ids: {_ids_text(report.missing_ids)}")
    if report.duplicate_ids:
        report.errors.append(f"{len(report.duplicate_ids)} duplicated ids: {_ids_text(report.duplicate_ids)}")
    if report.unknown_ids:
        report.errors.append(f"{len(report.unknown_ids)} ids not in split: {_ids_text(report.unknown_ids)}")
    if index is None:
        index = CatalogIndex(load_catalog(root, split))
    unknown_tools = sorted(
        {
            str(c.get("name"))
            for r in rows
            for c in r.get("tool_calls") or []
            if str(c.get("name")) != ASK_TOOL and not index.has_tool(str(c.get("name")))
        }
    )
    if unknown_tools:
        report.warnings.append(f"tools not in the catalog: {unknown_tools}")
    if report.invalid_lines:
        # `zaic_tools --validate` dừng ở dòng JSON hỏng đầu tiên, nên chỉ đưa vào các dòng parse được.
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "submission.jsonl"
            clean.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            _run_validate(report, clean, root, split)
    else:
        _run_validate(report, Path(submission_path), root, split)
    # Id lạ đã thành lỗi ở trên. Bỏ chúng khỏi thống kê và phát lại, vì không mở được `ToolEnv` cho id đó.
    known_rows = [r for r in rows if str(r["id"]) in expected_set]
    _distribution(report, known_rows)
    _replay(report, known_rows, zaic if zaic is not None else load_zaic(root, split), root, split)
    log.info("check_submission_done", ok=report.ok, errors=len(report.errors), warnings=len(report.warnings))
    return report

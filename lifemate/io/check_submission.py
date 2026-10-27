"""Kiểm tra hợp lệ một file submission cho split không có nhãn (SPEC 4.11, B3).

Nội dung file: `CheckReport` (kết quả kiểm tra) và `check_submission` (chạy toàn bộ các bước kiểm tra B3).

Cách xây dựng, theo thứ tự các bước:
1. Đọc từng dòng JSON; dòng hỏng được ghi lại, với id trùng thì chỉ giữ dòng đầu.
2. So id với split: thiếu, trùng hoặc lạ đều là lỗi (`errors`), tức file không nộp được.
3. Tên công cụ không có trong catalog chỉ là cảnh báo (`warnings`).
4. Chạy `zaic_tools --validate` trong tiến trình con, với PYTHONPATH trỏ vào thư mục split.
5. So phân bố (tỉ lệ câu trả lời rỗng, tỉ lệ có lời gọi hỏi lại, số lời gọi trung bình) với tập train;
   lệch quá `MAX_RATIO` lần thì cảnh báo.
6. Phát lại mọi lời gọi trong một ToolEnv mới và đếm kết quả ok:false.
Lỗi chặn việc nộp, còn cảnh báo chỉ cần xem lại; mọi bước đều chạy hết để báo cáo đầy đủ trong một lần.
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

# Phân bố tham chiếu của tập train (SPEC 4.11 B3): tỉ lệ câu trả lời rỗng, tỉ lệ có lời gọi hỏi lại, số lời gọi
# trung bình.
TRAIN_EMPTY_RATE = 0.07
TRAIN_ASK_RATE = 0.14
TRAIN_MEAN_CALLS = 2.3
MAX_RATIO = 2.0
MAX_LISTED_IDS = 20
_VALIDATE_RE = re.compile(r"(\d+) lines checked, (\d+) schema errors")


@dataclass(slots=True)
class CheckReport:
    """Kết quả của `check_submission`; có `errors` thì file không nộp được, `warnings` chỉ cần xem lại."""

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
        """True khi không tìm thấy lỗi nào."""
        return not self.errors

    def to_text(self) -> str:
        """Báo cáo dạng văn bản dễ đọc, ghi vào check.txt."""
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
    """Nối danh sách id để in, chỉ liệt kê tối đa `MAX_LISTED_IDS` id đầu và ghi số id còn lại."""
    more = f" (+{len(ids) - MAX_LISTED_IDS} more)" if len(ids) > MAX_LISTED_IDS else ""
    return ", ".join(ids[:MAX_LISTED_IDS]) + more


def _run_validate(report: CheckReport, submission: Path, root: Path, split: str) -> None:
    """Chạy `zaic_tools --validate` trên `submission` và ghi kết quả vào `report`.

    Số lỗi schema đọc từ stdout qua `_VALIDATE_RE`. Lệnh không chạy được hoặc kết thúc bất thường chỉ thành
    cảnh báo; có lỗi schema thì thành lỗi.
    """
    split_dir = (root / split).resolve()
    env = dict(os.environ)
    # Mỗi split có bản zaic_tools riêng: đặt thư mục split lên đầu PYTHONPATH để import đúng bản.
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
    # Không thấy dòng tổng kết thì không biết số lỗi schema, nên chỉ cảnh báo thay vì kết luận.
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
        # Lệch theo tỉ lệ (gấp hoặc chia `MAX_RATIO`), không theo hiệu số, vì các chỉ số có thang khác nhau.
        if val > ref * MAX_RATIO or val < ref / MAX_RATIO:
            report.warnings.append(f"{key}={val:.3f} is more than {MAX_RATIO:g}x off the train reference {ref}")


def _replay(report: CheckReport, rows: list[dict[str, Any]], zaic: ModuleType, root: Path, split: str) -> None:
    """Phát lại mọi lời gọi trong một ToolEnv mới (EnvRegistry riêng, tách khỏi registry của lượt chạy) và đếm ok:false.

    Số lời gọi bị từ chối theo từng id được ghi vào `report.rejected`; tổng hợp lại thành một cảnh báo.
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
    """Đọc submission: trả về dòng đầu tiên của mỗi id và số lần xuất hiện của mỗi id.

    Dòng không phải JSON, không phải object, thiếu `id` hoặc có `tool_calls` không phải list được ghi số dòng
    vào `report.invalid_lines`. Dòng trống bị bỏ qua và không tính vào `report.n_lines`.
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
        submission_path: file submission.jsonl cần kiểm tra.
        root: thư mục dữ liệu chứa `<split>/`.
        split: tên split.
        index: catalog công cụ đã dựng sẵn; None thì đọc từ catalog của split.
        zaic: module zaic_tools đã import; None thì import từ thư mục split.

    Trả về `CheckReport`; `report.ok` là False khi có lỗi chặn việc nộp.
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
        # zaic_tools --validate dừng ở dòng JSON hỏng đầu tiên; vì vậy chỉ validate các dòng parse được.
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "submission.jsonl"
            clean.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            _run_validate(report, clean, root, split)
    else:
        _run_validate(report, Path(submission_path), root, split)
    # Id lạ đã thành lỗi ở trên; bỏ chúng khỏi thống kê và phát lại vì ToolEnv không mở được cho id đó.
    known_rows = [r for r in rows if str(r["id"]) in expected_set]
    _distribution(report, known_rows)
    _replay(report, known_rows, zaic if zaic is not None else load_zaic(root, split), root, split)
    log.info("check_submission_done", ok=report.ok, errors=len(report.errors), warnings=len(report.warnings))
    return report
